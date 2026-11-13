# psl-toolchain

T1 of the PSL hospital communication project (ENCS5200, Section 35): lock the landmark toolchain so training (Python) and the live app (browser) use the same MediaPipe Tasks model.

| Piece | Status |
|---|---|
| `fps_test/` – phone FPS test (Holistic vs Hand + Pose) | done: Hand + Pose lite + Face every 3rd frame |
| Python extractor `python/extract_landmarks.py` + browser twin `web/landmarks.js` | done |
| Landmark subset (`landmark_subset.json`) | done (subset-v2, 80 points) |
| Parity test `web/parity/` (Python vs browser, 5 clips) | done: CPU 90.9–96.2%, GPU 74.3% |
| `TOOLCHAIN.md` | **done – T1 closed 13 Nov 2026** |

Pinned: `mediapipe==1.2.3` (Python) and `@mediapipe/tasks-vision@1.2.3` (browser).

**Phone FPS test:** https://yahyasarhan22.github.io/psl-toolchain/fps_test/

**Parity test:** https://yahyasarhan22.github.io/psl-toolchain/web/parity/ (pick the clips and the Python JSON files; nothing is uploaded)

No dataset videos or signer data go in this repo.
