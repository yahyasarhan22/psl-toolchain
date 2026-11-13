"""PSL T1 / A2 landmark extractor (subset-v2).

Same models, settings and frame layout as the browser (parity/landmarks.js):
  pose 0-16 (x,y,z,visibility) | left hand 21x(x,y,z) | right hand 21x(x,y,z) | face 21x(x,y,z)
  = 257 values per frame. Missing block -> None (NaN in numpy).

Usage:
  python extract_landmarks.py --models ../models --out out_dir clip1.mp4 clip2.mp4 ...
"""
import argparse, hashlib, json, math, os, platform, sys, time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

SUBSET_VERSION = "subset-v2"
POSE_IDX = list(range(17))
FACE_IDX = [10, 1, 152, 234, 454, 70, 105, 107, 300, 334, 336, 159, 145, 386, 374, 61, 291, 0, 17, 13, 14]
FACE_EVERY = 3
FACE_CROP_SCALE = 3.0   # crop side = 3 x head width (ears), square, around the ear midpoint
FACE_CROP_MIN = 64      # px
EXPECTED_SHA256 = {
    "hand_landmarker.task": "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1",
    "pose_landmarker_lite.task": "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a",
    "face_landmarker.task": "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
}
N_VALUES = 17 * 4 + 21 * 3 * 2 + 21 * 3  # 257


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_models(models_dir):
    for name, want in EXPECTED_SHA256.items():
        got = sha256(os.path.join(models_dir, name))
        if got != want:
            sys.exit(f"Model {name} hash mismatch: {got} != {want}")


def make_tasks(models_dir):
    run = vision.RunningMode.VIDEO
    hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=os.path.join(models_dir, "hand_landmarker.task")),
        running_mode=run, num_hands=2,
        min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5, min_tracking_confidence=0.5))
    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=os.path.join(models_dir, "pose_landmarker_lite.task")),
        running_mode=run, num_poses=1, output_segmentation_masks=False,
        min_pose_detection_confidence=0.5, min_pose_presence_confidence=0.5, min_tracking_confidence=0.5))
    face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=os.path.join(models_dir, "face_landmarker.task")),
        running_mode=vision.RunningMode.IMAGE, num_faces=1, output_face_blendshapes=False,
        min_face_detection_confidence=0.5, min_face_presence_confidence=0.5, min_tracking_confidence=0.5))
    return hand, pose, face


def face_crop_box(pose, W, H):
    """Square crop around the head from pose points. Same integer maths as parity/landmarks.js."""
    fl = math.floor
    ex = lambda i: (pose[i].x * W, pose[i].y * H)
    (x7, y7), (x8, y8), (x3, y3), (x6, y6) = ex(7), ex(8), ex(3), ex(6)
    head = max(math.hypot(x7 - x8, y7 - y8), 1.5 * math.hypot(x3 - x6, y3 - y6))
    side = max(fl(FACE_CROP_SCALE * head + 0.5), FACE_CROP_MIN)
    side = min(side, W, H)
    cx, cy = (x7 + x8) / 2, (y7 + y8) / 2
    x0 = min(max(fl(cx - side / 2 + 0.5), 0), W - side)
    y0 = min(max(fl(cy - side / 2 + 0.5), 0), H - side)
    return x0, y0, side


class _P:  # landmark in full-frame coordinates
    __slots__ = ("x", "y", "z")

    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


def run_face(face_t, rgb, pose):
    if pose is None:
        return None
    H, W = rgb.shape[:2]
    x0, y0, side = face_crop_box(pose, W, H)
    crop = np.ascontiguousarray(rgb[y0:y0 + side, x0:x0 + side])
    r = face_t.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=crop))
    if not r.face_landmarks:
        return None
    return [_P((x0 + p.x * side) / W, (y0 + p.y * side) / H, p.z * side / W) for p in r.face_landmarks[0]]


def assign_hands(hands, pose):
    """Return (left, right) hand landmark lists using the nearest pose wrist (15 = signer's left, 16 = right)."""
    left = right = None
    if not hands:
        return left, right
    if pose is None:  # no wrists: fall back to MediaPipe handedness order
        return (hands[0] if len(hands) > 0 else None), (hands[1] if len(hands) > 1 else None)
    lw, rw = pose[15], pose[16]
    d = lambda h, w: math.hypot(h[0].x - w.x, h[0].y - w.y)
    scored = sorted(((d(h, lw) - d(h, rw), i) for i, h in enumerate(hands)))  # most "left-like" first
    if len(hands) == 1:
        h = hands[0]
        return (h, None) if d(h, lw) <= d(h, rw) else (None, h)
    left = hands[scored[0][1]]
    right = hands[scored[-1][1]]
    return left, right


def frame_vector(pose, left, right, face):
    v = []
    if pose is None:
        v += [None] * 68
    else:
        for i in POSE_IDX:
            p = pose[i]
            v += [p.x, p.y, p.z, p.visibility]
    for h in (left, right):
        if h is None:
            v += [None] * 63
        else:
            for p in h:
                v += [p.x, p.y, p.z]
    if face is None:
        v += [None] * 63
    else:
        for i in FACE_IDX:
            p = face[i]
            v += [p.x, p.y, p.z]
    assert len(v) == N_VALUES
    return v


def extract(path, models_dir):
    hand, pose_t, face_t = make_tasks(models_dir)
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames, i, last_face = [], 0, None
    t0 = time.time()
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        ts = round(i * 1000 / fps)
        rh = hand.detect_for_video(img, ts)
        rp = pose_t.detect_for_video(img, ts)
        pose = rp.pose_landmarks[0] if rp.pose_landmarks else None
        if i % FACE_EVERY == 0:
            last_face = run_face(face_t, rgb, pose)
        left, right = assign_hands(list(rh.hand_landmarks), pose)
        frames.append(frame_vector(pose, left, right, last_face))
        i += 1
    cap.release()
    for t in (hand, pose_t, face_t):
        t.close()
    return {
        "clip": os.path.basename(path), "side": "python", "subset": SUBSET_VERSION,
        "frames": len(frames), "fps": fps, "seconds": round(time.time() - t0, 2),
        "mediapipe": mp.__version__, "opencv": cv2.__version__, "python": platform.python_version(),
        "platform": platform.platform(), "data": frames,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("clips", nargs="+")
    a = ap.parse_args()
    check_models(a.models)
    os.makedirs(a.out, exist_ok=True)
    for c in a.clips:
        r = extract(c, a.models)
        name = os.path.splitext(os.path.basename(c))[0] + ".python.json"
        with open(os.path.join(a.out, name), "w") as f:
            json.dump(r, f, separators=(",", ":"))
        d = np.array([[np.nan if x is None else x for x in row] for row in r["data"]], dtype=float)
        pres = lambda s: int(np.mean(~np.isnan(d[:, s])) * 100) if len(d) else 0
        nose_above = np.nanmean(d[:, 1] < (d[:, 11 * 4 + 1] + d[:, 12 * 4 + 1]) / 2) * 100
        print(f"{r['clip']}: {r['frames']} frames in {r['seconds']}s | pose {pres(slice(0,68))}% "
              f"L {pres(slice(68,131))}% R {pres(slice(131,194))}% face {pres(slice(194,257))}% "
              f"| upright (nose above shoulders) {nose_above:.0f}%")


if __name__ == "__main__":
    main()
