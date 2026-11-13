# TOOLCHAIN.md — PSL project, step T1 (draft)

Status: **in progress** (decisions 1–3 done 13 Nov 2026; Python extractor and parity test still to do).

Rule 3 of the roadmap: *one pinned MediaPipe Tasks model and one features.py for training and the live app.* This file is the single place that says which.

## 1. Versions (pinned)

| Side | Package | Version |
|---|---|---|
| Python (training, A2) | `mediapipe` | **1.2.3** |
| Browser (live app, C3) | `@mediapipe/tasks-vision` | **1.2.3** (wasm from the same version) |

Both released 11 Nov 2026. Never use `@latest` in app code.

## 2. Model choice: Hand Landmarker + Pose Landmarker lite + Face Landmarker every 3rd frame

Roadmap rule: Holistic Landmarker if the phone holds ≥ 15 FPS, otherwise Hand + Pose. Facial expressions carry part of PSL meaning, so a second test (13 Nov 2026) checked ways to keep the face.

Test phone: iPhone, iOS 26.6.1, Chrome, front camera 480×640, GPU delegate, signing continuously. Test page `fps_test/` (v1: 3 runs × 15 s; v2: 1 run × 30 s). Raw results in `fps_test/results/`.

| Setup | FPS | ms / frame | p95 ms | Face? |
|---|---|---|---|---|
| Holistic (v1 mean of 3, v2) | 12.4, 12.6 ✗ | 77 | 101–105 | full |
| Hand + Pose full (v1) | 16.2 | 58 | 78–82 | no |
| Hand + Pose lite (v1 mean, v2) | 18.0, 18.2 | 52 | 73–75 | no |
| Hand + Pose lite + Face every frame | 13.5 ✗ | 72 | 98 | full |
| Hand + Pose lite + Face every 2nd frame | 15.4 | 62 | 91 | full, ~8×/s |
| **Hand + Pose lite + Face every 3rd frame** | **16.0** ✓ | 60 | 90 | full, ~5×/s |

**Decision: Hand + Pose lite on every frame, Face Landmarker on every 3rd frame (last face held in between).**
- Holistic misses 15 FPS in all four runs.
- Hands move fast and need every frame; brows and mouth change more slowly, so ~5 face updates per second still catches a raised brow or mouth shape that lasts most of a sign.
- Every 2nd frame passes by only 0.4 FPS; every 3rd gives 1 FPS headroom, which is still thin.

**Safety valve (decided by data, not by guess):**
- A2 (offline, Python) extracts the face block for every take; speed does not matter offline.
- A4 trains the baseline with and without the face block on the LOSO folds.
- If the face adds less than ~2 points of top-1, the live app drops it and runs Hand + Pose lite at ~18 FPS.
- If A7's live prototype falls below 15 FPS with the face, it switches to every 4th frame or drops the face, and reports the fold accuracy without face.

| Model file | Bytes | SHA-256 |
|---|---|---|
| `hand_landmarker.task` (float16) | 7,819,105 | `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1` |
| `pose_landmarker_lite.task` (float16) | 5,777,746 | `59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a` |
| `face_landmarker.task` (float16) | 3,758,596 | `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff` |

The files are served from the app's own static hosting and loaded in Python from the same files, never from the `latest` URL at run time.

Open items:
- The test phone is an iPhone. If an Android phone will be used, run the same page on it once.
- **Frame-rate mismatch for A1:** training videos are 30 fps, but the live app gets ~16 landmark frames per second. A1 should resample training clips to 15 fps (or A3 resamples every sequence to a fixed length), so the model sees the same timing in training and live use.

## 3. Landmark subset (`landmark_subset.json`, subset-v2)

Per frame, 80 landmarks / 257 values:
- Pose 0–16 (x, y, z, visibility): coarse face points 0–10, shoulders, elbows, wrists.
- Left hand 21 × (x, y, z), right hand 21 × (x, y, z).
- Face 21 × (x, y, z) from FaceMesh: forehead, nose tip, chin, cheeks, 3 points per eyebrow, upper and lower eyelid per eye, mouth corners, outer and inner lips. Updated every 3rd frame, held in between. Optional block (see safety valve).
- Hand slots are assigned by the nearest pose wrist, not by MediaPipe's handedness label.
- Missing values are NaN; filling is decided in A3.

## 4. Task settings (same on both sides)

| Setting | Value |
|---|---|
| Running mode | VIDEO (timestamps in ms, strictly increasing) |
| Hand | `numHands: 2`, detection / presence / tracking confidence 0.5 |
| Pose | `numPoses: 1`, detection / presence / tracking confidence 0.5, no segmentation mask |
| Face | `numFaces: 1`, confidence 0.5, no blendshapes, run when `frame_index % 3 == 0` |
| Input frames | unmirrored, RGB; video preprocessed by A1 (rotation applied, SDR, 30 fps) |
| Delegate | browser: GPU, CPU fallback · Python: CPU |

## 5. Face from a head crop (found during parity)

Dataset v1 is filmed full-body: the face is only ~75–95 px wide in a 720 px frame. FaceLandmarker on the full frame found the face in **0% of frames in 3 of 5 parity clips**. Both extractors now cut a square around the head from the pose points (side = 3 × head width, min 64 px), run FaceLandmarker (IMAGE mode) on that crop and map the points back. Face found: 83–100% (the 83% clip is the sign "head", where the hand covers the face). Same integer maths in `python/extract_landmarks.py` and `web/landmarks.js`.

## 6. Parity test

5 clips from dataset v1, one per signer and category: pain (S8), where (S5), head (S2), letter ha (S3), number 7 (S6). Converted like A1 (rotation applied, HLG → SDR BT.709, 30 fps, 720×1280 H.264, muted).

**Run 1, 13 Nov 2026: headless Chrome in a Linux container, identical PNG frames on both sides** (`parity_results/2026-11-13_headless_chrome.json`). A frame matches when every block is found on both sides or missing on both, and each found block is within 1% of frame width (7.2 px) on average.

| Comparison | Frames matching (451 frames) | Hands err median | Face err median | Pose err median |
|---|---|---|---|---|
| Python ↔ browser **CPU** | **96.2%** ✓ (per clip 90.0–100) | 0.04–0.5 px | 0.1–0.4 px | 1.0–2.5 px |
| Python ↔ browser GPU (SwiftShader software emulation) | 83.2% | 0.3–2.7 px | 0.3–0.6 px | 3.6–5.2 px |

- On CPU the two runtimes give the same landmarks; remaining mismatches are borderline detections (a resting hand at the edge of the 0.5 threshold, the hand over the face in "head").
- The GPU delegate computes with lower precision, so borderline detections flip more often. SwiftShader is not a real GPU, so this number is only a warning. Run 2 measures real devices.

**Run 2 (to do): real browsers.** Laptop Chrome and the iPhone, CPU and GPU, with `web/parity/` (video frames decoded by the browser, as in the live app). Decision rule: if real-GPU parity is < 95%, measure the final pipeline on the phone with the CPU delegate; use CPU if it holds ≥ 15 FPS, otherwise keep GPU and report the parity number.

Note: headless Chrome in the container does not repaint video frames on seek (it fed the same frame every time); the parity page now refuses to run in a browser that does this.

## 7. Python environment

`python/requirements.txt`: `mediapipe==1.2.3`, `opencv-contrib-python==5.0.0.93`, `numpy==2.2.6` (Python 3.10–3.13). Run 1 used Python 3.13 on Linux x86-64. On a Linux machine without graphics, MediaPipe needs `libEGL.so.1` (package `libegl1`); Windows does not.
