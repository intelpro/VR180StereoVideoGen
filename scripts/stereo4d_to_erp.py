"""Stereo4D clips -> rectified 180x180 half-equirect (ERP) left/right videos.

For each clip (meta/<clip>.npz from download_stereo4d.py):
  1. decode only the clip's time range of raw/<video_id>.webm with ffmpeg (fast
     seek, frame pts read back from the showinfo filter), area-downscaled so the
     source matches the output angular resolution;
  2. match every annotated timestamp to the nearest decoded frame; split the
     clip wherever a timestamp is missing / the annotation has a gap;
  3. rectify each eye with rectified2rig and resample it to a square
     180x180-degree half-ERP (lon, lat in [-90, 90], lat increasing downwards).
     The raw YouTube VR180 eye covers fov_bounds (yaw is often wider than
     +-90, some clips only +-75); pixels outside it are left black.

Outputs, one set per contiguous segment (resumable; <clip>.done marks a clip):
  <out>/<video_id>/<clip>_<seg>_left.mp4, _right.mp4   (H.264, native fps)
  <out>/<video_id>/<clip>_<seg>.npz                    timestamps [us], camera2world,
                                                       rectified2rig, fov_bounds, fps, ...

Usage:
  python stereo4d_to_erp.py --root /group/video-tempo-control/VRStereoVideoGen/data/stereo4d \
      --size 1024 --workers 4 [--shard 0 --num_shards 16] [--clips a,b,c]
"""
import argparse
import glob
import hashlib
import json
import os
import os.path as osp
import re
import subprocess
import threading
import time
from collections import defaultdict
from multiprocessing import Pool

import cv2
import numpy as np

SHOWINFO_RE = re.compile(r'\bn:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:\s*(-?[\d.]+)')


def probe(path):
  out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                        '-show_entries', 'stream=width,height', '-of', 'json', path],
                       capture_output=True, text=True, check=True).stdout
  s = json.loads(out)['streams'][0]
  return s['width'], s['height']


def erp_to_src_map(R, size, fov_bounds, src_w, src_h):
  """Output ERP pixel (rectified frame) -> pixel in one source eye (x right, y down, z fwd)."""
  ang = (np.arange(size) + 0.5) / size * np.pi - np.pi / 2
  lon, lat = np.meshgrid(ang, ang)
  ray = np.stack([np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon)])
  ray = np.einsum('ij,jhw->ihw', R, ray)
  src_lon = np.arctan2(ray[0], ray[2])
  src_lat = np.arcsin(np.clip(ray[1], -1, 1))
  yaw0, yaw1, tilt0, tilt1 = np.radians(fov_bounds)
  mx = (src_lon - yaw0) / (yaw1 - yaw0) * src_w - 0.5
  my = (src_lat - tilt0) / (tilt1 - tilt0) * src_h - 0.5
  return mx.astype(np.float32), my.astype(np.float32)


def valid_fractions(erp, thresh=12):
  """Non-black fraction of a left-eye ERP frame: whole image, and |lon|,|lat| < 60 deg."""
  valid = erp.max(axis=2) > thresh
  s = erp.shape[0]
  lo, hi = s // 6, s - s // 6
  return float(valid.mean()), float(valid[lo:hi, lo:hi].mean())


def open_writer(path, size, fps, crf):
  return subprocess.Popen(
      ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
       '-s', f'{size}x{size}', '-r', f'{fps:.6f}', '-i', '-', '-c:v', 'libx264', '-threads', '2',
       '-crf', str(crf), '-preset', 'fast', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', path],
      stdin=subprocess.PIPE)


def decode_frames(path, t0, t1, out_w, out_h, threads):
  """Yield (pts_seconds, BGR frame) for frames in [t0, t1], scaled to out_w x out_h."""
  # accurate input seek resets output pts to 0 at t0, so absolute time = t0 + pts_time
  # (-copyts would keep absolute pts but then -t is applied to them and nothing is output)
  t0 = max(t0, 0.0)
  cmd = ['ffmpeg', '-v', 'info', '-nostats', '-threads', str(threads),
         '-ss', f'{t0:.6f}', '-i', path, '-t', f'{t1 - t0:.6f}',
         '-vf', f'scale={out_w}:{out_h}:flags=area,showinfo',
         '-fps_mode', 'passthrough', '-an', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-']
  p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, bufsize=0)
  times, err_tail = [], []
  cond = threading.Condition()

  def read_stderr():
    for line in iter(p.stderr.readline, b''):
      line = line.decode(errors='replace')
      m = SHOWINFO_RE.search(line)
      with cond:
        if m:
          times.append(t0 + float(m.group(3)))
        else:
          err_tail.append(line)
          del err_tail[:-20]
        cond.notify_all()
    with cond:
      times.append(None)  # EOF sentinel
      cond.notify_all()

  th = threading.Thread(target=read_stderr, daemon=True)
  th.start()
  nbytes = out_w * out_h * 3
  i = 0
  try:
    while True:
      buf = bytearray()
      while len(buf) < nbytes:
        chunk = p.stdout.read(nbytes - len(buf))
        if not chunk:
          break
        buf += chunk
      if len(buf) < nbytes:
        break
      with cond:
        cond.wait_for(lambda: len(times) > i)
        t = times[i]
      if t is None:
        raise RuntimeError('frame without showinfo line')
      yield t, np.frombuffer(bytes(buf), np.uint8).reshape(out_h, out_w, 3)
      i += 1
  finally:
    p.stdout.close()
    code = p.wait()
    th.join()
  if code != 0:
    raise RuntimeError(f'ffmpeg exited {code}: {"".join(err_tail[-5:]).strip()}')


class SegmentWriter:
  """Streams rectified frames of contiguous runs into <clip>_<seg>_{left,right}.mp4."""

  def __init__(self, out_dir, clip, meta, args, maps, eye_w, fps, src_wh):
    self.out_dir, self.clip, self.meta, self.args = out_dir, clip, meta, args
    self.maps, self.eye_w, self.fps, self.src_wh = maps, eye_w, fps, src_wh
    self.cur, self.saved = None, []

  def add(self, i, img):
    first = self.cur is None
    if first:
      name = f'{self.clip}_{len(self.saved)}'
      tmp = {e: osp.join(self.out_dir, f'.{name}_{e}.tmp.mp4') for e in ('left', 'right')}
      self.cur = {'name': name, 'start': i, 'end': i, 'tmp': tmp,
                  'w': {e: open_writer(tmp[e], self.args['size'], self.fps, self.args['crf']) for e in tmp}}
    for e, eye in enumerate(('left', 'right')):
      src = img[:, e * self.eye_w:(e + 1) * self.eye_w]
      out = cv2.remap(src, *self.maps[e], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
      self.cur['w'][eye].stdin.write(out.tobytes())
      if first and eye == 'left':
        self.cur['valid'] = valid_fractions(out)
    self.cur['end'] = i + 1

  def close(self):
    c, self.cur = self.cur, None
    if c is None:
      return
    for e, w in c['w'].items():
      w.stdin.close()
      if w.wait() != 0:
        raise RuntimeError(f"{c['name']}: ffmpeg encode failed ({e})")
    a, b = c['start'], c['end']
    if b - a < self.args['min_frames']:
      for f in c['tmp'].values():
        os.remove(f)
      return
    for e, f in c['tmp'].items():
      os.replace(f, osp.join(self.out_dir, f"{c['name']}_{e}.mp4"))
    m = self.meta
    np.savez(osp.join(self.out_dir, f"{c['name']}.npz"),
             timestamps=m['timestamps'][a:b], camera2world=m['camera2world'][a:b],
             rectified2rig=m['rectified2rig'], fov_bounds=m['fov_bounds'], fps=self.fps,
             size=self.args['size'], src_wh=np.array(self.src_wh), clip=self.clip,
             seg_range=np.array([a, b]), valid_all=c['valid'][0], valid_center=c['valid'][1])
    self.saved.append({'name': c['name'], 'frames': b - a,
                       'valid_all': round(c['valid'][0], 3), 'valid_center': round(c['valid'][1], 3)})

  def abort(self):
    if self.cur is not None:
      for w in self.cur['w'].values():
        w.stdin.close()
        w.wait()
      for f in self.cur['tmp'].values():
        if osp.exists(f):
          os.remove(f)
      self.cur = None


def process_clip(job):
  clip, args = job['clip'], job['args']
  vid = clip[:11]
  out_dir = osp.join(args['out'], vid)
  done_path = osp.join(out_dir, f'{clip}.done')
  if osp.exists(done_path):
    return clip, 'exists', None
  os.makedirs(out_dir, exist_ok=True)
  for f in glob.glob(osp.join(out_dir, f'.{glob.escape(clip)}_*.tmp.mp4')):
    os.remove(f)  # left over from a run that was killed (e.g. time limit)
  t_start = time.time()

  meta = np.load(osp.join(args['root'], 'meta', f'{clip}.npz'))
  ts = meta['timestamps'].astype(np.int64)
  r2r, fov = meta['rectified2rig'], meta['fov_bounds']
  video = osp.join(args['root'], 'raw', f'{vid}.webm')
  W, H = probe(video)
  if W // 2 < args['min_eye_width']:
    info = {'status': 'skipped', 'reason': f'low-res eye {W // 2}x{H}'}
    json.dump(info, open(done_path, 'w'))
    return clip, 'skipped', info['reason']

  # area-downscale the source to the output's angular resolution (+ a little margin)
  px_per_deg = args['size'] / 180 * args['oversample']
  eye_w = int(round(px_per_deg * (fov[1] - fov[0]) / 2) * 2)
  src_h = int(round(px_per_deg * (fov[3] - fov[2]) / 2) * 2)
  eye_w, src_h = min(eye_w, W // 2), min(src_h, H)
  maps = [erp_to_src_map(r2r[e], args['size'], fov, eye_w, src_h) for e in range(2)]

  dts = np.diff(ts)
  med_dt = float(np.median(dts)) if len(dts) else 33367.0
  tol, max_gap = 0.5 * med_dt / 1e6, 1.5 * med_dt
  fps = 1e6 / med_dt
  need = ts / 1e6
  writer = SegmentWriter(out_dir, clip, meta, args, maps, eye_w, fps, (W, H))
  # Sequential merge: each annotated timestamp gets the nearest decoded frame within
  # +-tol. A timestamp is final once decoding passes need[k] + tol; it is then appended
  # to the open segment, or closes it (no match / same frame as previous / annotation gap).
  best = {}  # k -> (abs err, decoded frame index, frame)
  state = {'k': 0, 'last_fi': None, 'matched': 0}

  def finalize_until(pts):
    while state['k'] < len(ts) and (pts is None or need[state['k']] + tol < pts):
      k = state['k']
      b = best.pop(k, None)
      if k > 0 and ts[k] - ts[k - 1] > max_gap:
        writer.close()
      if b is None or b[1] == state['last_fi']:
        writer.close()
      else:
        writer.add(k, b[2])
        state['matched'] += 1
      state['last_fi'] = b[1] if b else None
      state['k'] += 1

  try:
    frames = decode_frames(video, need[0] - 0.5, need[-1] + 0.5, 2 * eye_w, src_h, args['threads'])
    for fi, (pts, img) in enumerate(frames):
      finalize_until(pts)
      k = state['k']
      while k < len(ts) and need[k] - tol <= pts:
        err = abs(need[k] - pts)
        if k not in best or err < best[k][0]:
          best[k] = (err, fi, img)
        k += 1
      if state['k'] >= len(ts):
        break
    frames.close()  # stop ffmpeg early once every timestamp is final
    finalize_until(None)
    writer.close()
  except BaseException:
    writer.abort()
    raise

  written = writer.saved
  info = {'status': 'done', 'frames': len(ts), 'matched': state['matched'], 'fps': fps,
          'segments': written, 'fov_bounds': fov.tolist(), 'src_wh': [W, H],
          'sec': round(time.time() - t_start, 1)}
  json.dump(info, open(done_path, 'w'))
  return clip, 'done', f"{sum(s['frames'] for s in written)}/{len(ts)} frames, {len(written)} seg, {info['sec']}s"


def list_clips(root):
  done = set()
  for line in open(osp.join(root, 'state.jsonl')):
    try:  # the downloader may be appending a line right now
      r = json.loads(line)
    except json.JSONDecodeError:
      continue
    if r['status'] == 'done':
      done.add(r['vid'])
  by_video = json.load(open(osp.join(root, 'train_clips.json')))
  clips = []
  for vid in sorted(done):
    for clip in by_video.get(vid, []):
      if osp.exists(osp.join(root, 'meta', f'{clip}.npz')):
        clips.append(clip)
  return clips


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--root', required=True, help='stereo4d dir with raw/, meta/, state.jsonl')
  p.add_argument('--out', default=None, help='default: <root>/erp<size>')
  p.add_argument('--size', type=int, default=1024, help='output ERP side (px) per eye')
  p.add_argument('--oversample', type=float, default=1.25,
                 help='source is area-downscaled to size/180*oversample px per degree before remap')
  p.add_argument('--crf', type=int, default=18)
  p.add_argument('--min_frames', type=int, default=32, help='drop segments shorter than this')
  p.add_argument('--min_eye_width', type=int, default=1500, help='skip low-res uploads')
  p.add_argument('--workers', type=int, default=4, help='clips processed in parallel')
  p.add_argument('--threads', type=int, default=4, help='ffmpeg decode threads per clip')
  p.add_argument('--shard', type=int, default=0)
  p.add_argument('--num_shards', type=int, default=1)
  p.add_argument('--clips', default=None, help='comma-separated clip names (debug)')
  args = p.parse_args()
  args.out = args.out or osp.join(args.root, f'erp{args.size}')
  os.makedirs(args.out, exist_ok=True)

  clips = args.clips.split(',') if args.clips else list_clips(args.root)
  # shard by a stable hash of the video id, so shards never overlap even if the
  # download is still adding videos between shard starts
  clips = [c for c in clips
           if int(hashlib.md5(c[:11].encode()).hexdigest(), 16) % args.num_shards == args.shard]
  mine = {c[:11] for c in clips}
  print(f'shard {args.shard}/{args.num_shards}: {len(clips)} clips from {len(mine)} videos', flush=True)

  a = {k: getattr(args, k) for k in ('root', 'out', 'size', 'oversample', 'crf', 'min_frames',
                                     'min_eye_width', 'threads')}
  counts = defaultdict(int)
  t0 = time.time()
  with Pool(args.workers) as pool:
    for n, (clip, status, msg) in enumerate(pool.imap_unordered(
        _safe_process, [{'clip': c, 'args': a} for c in clips]), 1):
      counts[status] += 1
      if status != 'exists':
        print(f'[{n}/{len(clips)} {time.time() - t0:.0f}s] {clip} {status} {msg or ""}', flush=True)
  print(f'finished: {dict(counts)}', flush=True)


def _safe_process(job):
  try:
    return process_clip(job)
  except Exception as e:  # keep going; the clip is retried on the next run
    return job['clip'], 'error', repr(e)[:300]


if __name__ == '__main__':
  main()
