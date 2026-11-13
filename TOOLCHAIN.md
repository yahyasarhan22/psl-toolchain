# TOOLCHAIN.md — PSL project, step T1 (draft)

Status: **in progress** (decisions 1–3 done 13 Nov 2026; Python extractor and parity test still to do).

Rule 3 of the roadmap: *one pinned MediaPipe Tasks model and one features.py for training and the live app.* This file is the single place that says which.

## 1. Versions (pinned)

| Side | Package | Version |
|---|---|---|
| Python (training, A2) | `mediapipe` | **1.2.3** |
| Browser (live app, C3) | `@mediapipe/tasks-vision` | **1.2.3** (wasm from the same version) |

Both released 11 Nov 2026. Never use `@latest` in app code.

## 2. Model choice: Hand Landmarker + Pose Landmarker (lite)

Roadmap rule: Holistic Landmarker if the phone holds ≥ 15 FPS, otherwise Hand + Pose.

Phone test, 13 Nov 2026 (`fps_test/`, results in `fps_test/results/`): iPhone, iOS 26.6.1, Chrome, front camera 480×640, GPU delegate, signing in front of the camera, 3 runs × 15 s:

| Setup | FPS (mean of 3) | ms / frame | p95 ms | both hands found |
|---|---|---|---|---|
| Holistic | **12.4** ✗ | 77 | 101–105 | 92–94% |
| Hand + Pose (full) | 16.2 | 58 | 78–82 | 85–86% |
| Hand + Pose (lite) | **18.0** ✓ | 52 | 73–74 | 86–94% |

Decision: **Hand Landmarker + Pose Landmarker lite.**
- Holistic misses the 15 FPS bar in all three runs.
- Pose *full* passes by only 1 FPS. The real app also draws the UI, builds features and sends requests, so it would drop under 15. Lite leaves ~3 FPS of headroom.
- Lite's lower pose accuracy matters little here: finger detail comes from the Hand model, and pose is only used for head, shoulders, elbows and wrists.
- Cost: no FaceMesh. The ~20 face points in the plan are replaced by the 11 face points of the pose model (nose, eyes, ears, mouth corners). Eyebrow and detailed mouth shapes are lost; this is a known limitation for PSL non-manual signals and goes in the final report.

| Model file | Bytes | SHA-256 |
|---|---|---|
| `hand_landmarker.task` (float16) | 7,819,105 | `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1` |
| `pose_landmarker_lite.task` (float16) | 5,777,746 | `59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a` |

Both files are served from the app's own static hosting and loaded in Python from the same files, never from the `latest` URL at run time.

Open item: the test phone is an iPhone. If the team plans to use an Android phone in the hospital, run the same page on it once; if Hand + Pose lite is under 15 FPS there, revisit.

## 3. Landmark subset (`landmark_subset.json`, subset-v1)

Per frame, 59 landmarks / 194 values:
- Pose 0–16 (x, y, z, visibility): face proxy 0–10, shoulders, elbows, wrists.
- Left hand 21 × (x, y, z), right hand 21 × (x, y, z).
- Hand slots are assigned by the nearest pose wrist, not by MediaPipe's handedness label.
- A missing hand is NaN; filling is decided in A3.

## 4. Task settings (same on both sides)

| Setting | Value |
|---|---|
| Running mode | VIDEO (timestamps in ms, strictly increasing) |
| Hand | `numHands: 2`, detection / presence / tracking confidence 0.5 |
| Pose | `numPoses: 1`, detection / presence / tracking confidence 0.5, no segmentation mask |
| Input frames | unmirrored, RGB; video preprocessed by A1 (rotation applied, SDR, 30 fps) |
| Delegate | browser: GPU, CPU fallback · Python: CPU |

## 5. Parity test (to do)

5 clips from dataset v1 → same frames through Python and the browser → compare per-landmark error and detection agreement. Pass rule: to be filled with the result.

## 6. Python environment (to do)

`requirements.txt` with exact versions and the Python version used.
