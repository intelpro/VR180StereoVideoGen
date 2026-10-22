"""Download Stereo4D source VR180 videos from YouTube + compact per-clip metadata.

For each selected YouTube video:
  raw/<video_id>.webm       full VR180 SBS video (YouTube 3840x2160 "mesh" format;
                            each eye is treated as a 180x180 equirect, verified to
                            rectify as well as the official equirect demo)
  meta/<clip>.npz           timestamps, camera2world, rectified2rig, fov_bounds
                            (only these members are range-read from the released
                            npz; the 3D tracks, ~95% of each file, are skipped)

Resumable: progress is appended to <root>/state.jsonl; rerun the same command.

Usage:
  python download_stereo4d.py --root /group/video-tempo-control/VRStereoVideoGen/data/stereo4d \
      --num_videos 1000 --workers 4
"""
import argparse
import io
import json
import os
import os.path as osp
import random
import subprocess
import threading
import time
import urllib.parse
import urllib.request
import zipfile
from collections import defaultdict

import numpy as np

BUCKET = 'stereo4d'
LIST_URL = f'https://storage.googleapis.com/storage/v1/b/{BUCKET}/o'
OBJ_URL = f'https://storage.googleapis.com/{BUCKET}/'
META_KEYS = ('timestamps', 'camera2world', 'rectified2rig', 'fov_bounds')
FORMAT = ('bv[vcodec^=vp9][height>=1920][height<=2160]/'
          'bv[height>=1920][height<=2160]')
BOT_MARKERS = ('Sign in to confirm', "confirm you're not a bot", 'HTTP Error 429')
GONE_MARKERS = ('Video unavailable', 'Private video', 'has been removed',
                'account associated with this video has been terminated',
                'This video is not available')


# --------------------------------------------------------------------------
# Bucket listing
# --------------------------------------------------------------------------
def list_clips(split, cache_path):
  """Return {video_id: [clip_name, ...]} for <split>/*.npz, cached to json."""
  if osp.exists(cache_path):
    return json.load(open(cache_path))
  by_video, token, n = defaultdict(list), None, 0
  while True:
    q = {'prefix': f'{split}/', 'maxResults': 1000, 'fields': 'items(name),nextPageToken'}
    if token:
      q['pageToken'] = token
    page = json.load(urllib.request.urlopen(f'{LIST_URL}?{urllib.parse.urlencode(q)}', timeout=60))
    for it in page.get('items', []):
      clip = osp.basename(it['name'])[:-len('.npz')]
      # clip = <11-char youtube id>_<first timestamp>; ids may contain '_'
      by_video[clip[:11]].append(clip)
      n += 1
    token = page.get('nextPageToken')
    if not token:
      break
  print(f'listed {n} clips from {len(by_video)} videos in {split}/')
  json.dump(by_video, open(cache_path, 'w'))
  return by_video


# --------------------------------------------------------------------------
# Range-read only the needed members of a remote npz
# --------------------------------------------------------------------------
class HttpRangeFile(io.RawIOBase):
  """Seekable read-only file over HTTP Range requests with a block cache."""

  BLOCK = 1 << 16

  def __init__(self, url):
    self.url, self.pos, self.cache = url, 0, {}
    req = urllib.request.Request(url, method='HEAD')
    self.size = int(urllib.request.urlopen(req, timeout=60).headers['Content-Length'])

  def readable(self):
    return True

  def seekable(self):
    return True

  def tell(self):
    return self.pos

  def seek(self, off, whence=io.SEEK_SET):
    self.pos = {io.SEEK_SET: off, io.SEEK_CUR: self.pos + off, io.SEEK_END: self.size + off}[whence]
    return self.pos

  def _block(self, i):
    if i not in self.cache:
      lo, hi = i * self.BLOCK, min(self.size, (i + 1) * self.BLOCK) - 1
      req = urllib.request.Request(self.url, headers={'Range': f'bytes={lo}-{hi}'})
      self.cache[i] = urllib.request.urlopen(req, timeout=60).read()
    return self.cache[i]

  def read(self, n=-1):
    end = self.size if n is None or n < 0 else min(self.size, self.pos + n)
    out = bytearray()
    while self.pos < end:
      i, o = divmod(self.pos, self.BLOCK)
      chunk = self._block(i)[o:o + end - self.pos]
      out += chunk
      self.pos += len(chunk)
    return bytes(out)

  def readinto(self, b):
    data = self.read(len(b))
    b[:len(data)] = data
    return len(data)


def fetch_meta(clip, split, out_path):
  with zipfile.ZipFile(HttpRangeFile(f'{OBJ_URL}{split}/{clip}.npz')) as z:
    arrays = {k: np.lib.format.read_array(io.BytesIO(z.read(f'{k}.npy'))) for k in META_KEYS}
  tmp = out_path + '.tmp.npz'
  np.savez(tmp, **arrays)
  os.replace(tmp, out_path)


# --------------------------------------------------------------------------
# Downloader
# --------------------------------------------------------------------------
class Downloader:

  def __init__(self, args):
    self.args = args
    self.raw_dir = osp.join(args.root, 'raw')
    self.meta_dir = osp.join(args.root, 'meta')
    os.makedirs(self.raw_dir, exist_ok=True)
    os.makedirs(self.meta_dir, exist_ok=True)
    self.state_path = osp.join(args.root, 'state.jsonl')
    self.lock = threading.Lock()
    self.cooldown_until = 0.0
    self.bot_strikes = 0
    self.stop = threading.Event()

    self.final = {}  # video_id -> final status (done / skipped / gone)
    if osp.exists(self.state_path):
      for line in open(self.state_path):
        r = json.loads(line)
        if r['status'] in ('done', 'skipped', 'gone'):
          self.final[r['vid']] = r['status']
    self.n_done = sum(s == 'done' for s in self.final.values())

  def log(self, **rec):
    rec['time'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    with self.lock:
      with open(self.state_path, 'a') as f:
        f.write(json.dumps(rec) + '\n')
      if rec['status'] in ('done', 'skipped', 'gone'):
        self.final[rec['vid']] = rec['status']
      if rec['status'] == 'done':
        self.n_done += 1
      counts = defaultdict(int)
      for s in self.final.values():
        counts[s] += 1
    print(f"[{rec['time']}] {rec['vid']} {rec['status']:8s} {rec.get('msg', '')[:80]} "
          f"| done={counts['done']}/{self.args.num_videos} skipped={counts['skipped']} gone={counts['gone']}",
          flush=True)

  def wait_cooldown(self):
    while not self.stop.is_set():
      left = self.cooldown_until - time.time()
      if left <= 0:
        return
      time.sleep(min(left, 30))

  def ytdlp(self, vid):
    out_tmpl = osp.join(self.raw_dir, '%(id)s.%(ext)s')
    cmd = self.args.ytdlp.split() + [
        '--no-progress', '--no-warnings', '-f', FORMAT,
        '--match-filter', f'duration <= {self.args.max_duration}',
        '--retries', '5', '--fragment-retries', '10',
        '-o', out_tmpl, '--', vid]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=self.args.timeout)
    return p.returncode, p.stdout + p.stderr

  def video_file(self, vid):
    for ext in ('webm', 'mp4'):
      path = osp.join(self.raw_dir, f'{vid}.{ext}')
      if osp.exists(path):
        return path
    return None

  def process(self, vid, clips):
    if not self.video_file(vid):
      self.wait_cooldown()
      code, out = self.ytdlp(vid)
      if any(m in out for m in BOT_MARKERS):
        with self.lock:
          self.bot_strikes += 1
          self.cooldown_until = time.time() + self.args.bot_cooldown
          strikes = self.bot_strikes
        self.log(vid=vid, status='error', msg=f'bot check (strike {strikes}), cooling down')
        if strikes >= self.args.max_bot_strikes:
          print('too many bot checks in a row, stopping', flush=True)
          self.stop.set()
        return
      if any(m in out for m in GONE_MARKERS):
        self.log(vid=vid, status='gone', msg=out.strip().splitlines()[-1] if out.strip() else '')
        return
      if 'does not pass filter' in out:
        self.log(vid=vid, status='skipped', msg='longer than max_duration')
        return
      if code != 0 or not self.video_file(vid):
        self.log(vid=vid, status='error', msg=(out.strip().splitlines() or ['?'])[-1])
        return
      with self.lock:
        self.bot_strikes = 0

    try:
      for clip in clips:
        path = osp.join(self.meta_dir, f'{clip}.npz')
        if not osp.exists(path):
          fetch_meta(clip, self.args.split, path)
    except Exception as e:  # keep the video; meta is retried on the next run
      self.log(vid=vid, status='error', msg=f'meta: {e!r}')
      return
    size = osp.getsize(self.video_file(vid)) / 2**20
    self.log(vid=vid, status='done', msg=f'{size:.0f}MB, {len(clips)} clips', clips=len(clips))

  def worker(self, queue):
    while not self.stop.is_set():
      with self.lock:
        if self.n_done >= self.args.num_videos or not queue:
          return
        vid, clips = queue.pop()
      try:
        self.process(vid, clips)
      except subprocess.TimeoutExpired:
        self.log(vid=vid, status='error', msg='yt-dlp timeout')
      except Exception as e:
        self.log(vid=vid, status='error', msg=repr(e))
      time.sleep(self.args.sleep * random.uniform(0.5, 1.5))

  def run(self, by_video):
    vids = sorted(by_video)
    random.Random(self.args.seed).shuffle(vids)
    # pop() takes from the end, so reverse to keep the seeded order
    queue = [(v, by_video[v]) for v in reversed(vids) if v not in self.final]
    print(f'{self.n_done} videos already done, {len(queue)} candidates left', flush=True)
    threads = [threading.Thread(target=self.worker, args=(queue,)) for _ in range(self.args.workers)]
    for t in threads:
      t.start()
    for t in threads:
      t.join()
    print(f'finished: {self.n_done} videos done', flush=True)


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--root', required=True)
  p.add_argument('--split', default='train', choices=['train', 'test'])
  p.add_argument('--num_videos', type=int, default=1000)
  p.add_argument('--max_duration', type=int, default=600, help='seconds; longer videos are skipped')
  p.add_argument('--workers', type=int, default=4)
  p.add_argument('--sleep', type=float, default=5, help='mean pause per worker between videos (s)')
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--timeout', type=int, default=3600, help='per-video yt-dlp timeout (s)')
  p.add_argument('--bot_cooldown', type=int, default=1800, help='pause after a bot check (s)')
  p.add_argument('--max_bot_strikes', type=int, default=3)
  p.add_argument('--ytdlp', default=osp.expanduser('~/.local/bin/uvx yt-dlp@latest'))
  args = p.parse_args()

  os.makedirs(args.root, exist_ok=True)
  by_video = list_clips(args.split, osp.join(args.root, f'{args.split}_clips.json'))
  Downloader(args).run(by_video)


if __name__ == '__main__':
  main()
