"""Stereo4D clip -> rectified pinhole perspective L/R videos.

Reads the YouTube VR180 SBS video (each eye treated as 180x180 equirect, see
fov_bounds), rectifies each eye with the npz rectified2rig, and renders a
forward-facing pinhole view.

Usage:
  python stereo4d_to_perspective.py --root stereo4d --clips stereo4d/clips.txt \
      --hfov 90 --size 512
Outputs: <root>/perspective_<hfov>/<clip>_{left,right,sbs}.mp4
"""
import argparse
import os
import os.path as osp
import subprocess

import cv2
import numpy as np


def pers_to_eq_map(R, eq_h, eq_w, fov_bounds, size, hfov):
  """Pinhole (y down, z forward) pixel -> rotated by R -> equirect pixel."""
  yaw0, yaw1, tilt0, tilt1 = np.radians(fov_bounds)
  f = size / 2 / np.tan(np.radians(hfov) / 2)
  u, v = np.meshgrid(np.arange(size), np.arange(size))
  ray = np.stack([(u - size / 2 + 0.5) / f, (v - size / 2 + 0.5) / f, np.ones((size, size))])
  ray /= np.linalg.norm(ray, axis=0)
  ray = np.einsum('ij,jhw->ihw', R, ray)
  lon = np.arctan2(ray[0], ray[2])
  lat = np.arcsin(np.clip(ray[1], -1, 1))
  mx = (lon - yaw0) / (yaw1 - yaw0) * (eq_w - 1)
  my = (lat - tilt0) / (tilt1 - tilt0) * (eq_h - 1)
  return mx.astype(np.float32), my.astype(np.float32)


def open_writer(path, w, h, fps=30):
  return subprocess.Popen(
      ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
       '-s', f'{w}x{h}', '-r', str(fps), '-i', '-', '-c:v', 'libx264',
       '-crf', '18', '-preset', 'fast', '-pix_fmt', 'yuv420p', path],
      stdin=subprocess.PIPE)


def process(clip, root, out_dir, hfov, size):
  d = np.load(osp.join(root, 'npz', f'{clip}.npz'))
  ts, r2r, fov_bounds = d['timestamps'], d['rectified2rig'], d['fov_bounds']
  cap = cv2.VideoCapture(osp.join(root, 'raw', f'{clip[:11]}.webm'))
  eq_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  eq_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2
  maps = [pers_to_eq_map(r2r[e], eq_h, eq_w, fov_bounds, size, hfov) for e in range(2)]

  writers = {k: open_writer(osp.join(out_dir, f'{clip}_{k}.mp4'), size * (2 if k == 'sbs' else 1), size)
             for k in ('left', 'right', 'sbs')}
  for t in ts:
    cap.set(cv2.CAP_PROP_POS_MSEC, t / 1000)
    ok, img = cap.read()
    if not ok:
      raise RuntimeError(f'{clip}: failed to read frame at {t}us')
    left, right = (cv2.remap(img[:, e * eq_w:(e + 1) * eq_w], *maps[e], cv2.INTER_LINEAR)
                   for e in range(2))
    writers['left'].stdin.write(left.tobytes())
    writers['right'].stdin.write(right.tobytes())
    writers['sbs'].stdin.write(np.concatenate([left, right], axis=1).tobytes())
  for w in writers.values():
    w.stdin.close()
    w.wait()
  print(f'{clip}: {len(ts)} frames')


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--root', required=True)
  p.add_argument('--clips', required=True)
  p.add_argument('--hfov', type=float, default=90)
  p.add_argument('--size', type=int, default=512)
  args = p.parse_args()
  out_dir = osp.join(args.root, f'perspective_{int(args.hfov)}')
  os.makedirs(out_dir, exist_ok=True)
  for clip in open(args.clips).read().split():
    process(clip, args.root, out_dir, args.hfov, args.size)


if __name__ == '__main__':
  main()
