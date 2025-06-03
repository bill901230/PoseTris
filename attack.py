#!/usr/bin/env python3
"""
skeleton_grid_image.py  (fixed global)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* 刪除 `global pose` 與先前的「先用後宣告」錯誤。
* `process_image()` 現在接受 `pose` 實例當參數，避免全域依賴。

九宮格索引 (row‑major)：
    0 1 2
    3 4 5
    6 7 8
"""
from __future__ import annotations
import argparse, cv2, numpy as np, mediapipe as mp
from pathlib import Path


def grid_indices_from_landmarks(h:int, w:int, landmarks) -> set[int]:
    occ = set()
    for lm in landmarks:
        if lm.visibility < 0.5:
            continue
        cx, cy = int(lm.x * w), int(lm.y * h)
        col = min(cx * 3 // w, 2)
        row = min(cy * 3 // h, 2)
        occ.add(row * 3 + col)
    return occ


def process_image(path: Path, pose: mp.solutions.pose.Pose, show: bool=False):
    img = cv2.imread(str(path))
    if img is None:
        print(f"[WARN] Cannot read {path}")
        return
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    occ = set()
    if results.pose_landmarks:
        occ = grid_indices_from_landmarks(h, w, results.pose_landmarks.landmark)
        if show:
            for lm in results.pose_landmarks.landmark:
                if lm.visibility < 0.5: continue
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (cx, cy), 3, (0,255,0), -1)

    if show:
        step_x, step_y = w//3, h//3
        for i in range(1,3):
            cv2.line(img, (0,i*step_y), (w,i*step_y), (200,200,200),1)
            cv2.line(img, (i*step_x,0), (i*step_x,h), (200,200,200),1)
        cv2.putText(img, str(sorted(occ)), (10,h-10), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
        cv2.imshow(str(path.name), img)
        cv2.waitKey(0)
        cv2.destroyWindow(str(path.name))

    print(sorted(occ))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Skeleton grid (image)")
    ap.add_argument("images", nargs='+', help="image file(s) or glob pattern")
    ap.add_argument("--show", action='store_true', help="display annotated image")
    args = ap.parse_args()

    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=False, model_complexity=0,
                      enable_segmentation=False,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        for pattern in args.images:
            for p in sorted(Path().glob(pattern)):
                process_image(p, pose, show=args.show)
