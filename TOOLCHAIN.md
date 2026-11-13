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

## 5. Parity test (to do)

5 clips from dataset v1 → same frames through Python and the browser → compare per-landmark error and detection agreement. Pass rule: to be filled with the result.

## 6. Python environment (to do)

`requirements.txt` with exact versions and the Python version used.
