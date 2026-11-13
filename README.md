# psl-toolchain

T1 of the PSL hospital communication project (ENCS5200, Section 35): lock the landmark toolchain so training (Python) and the live app (browser) use the same MediaPipe Tasks model.

| Piece | Status |
|---|---|
| `fps_test/` – phone FPS test (Holistic vs Hand + Pose) | done: Hand + Pose lite chosen |
| Python extractor, pinned versions | to do |
| Landmark subset (`landmark_subset.json`) | done (subset-v1) |
| Parity test (Python vs browser, 5 clips) | to do |
| `TOOLCHAIN.md` | draft |

Pinned: `mediapipe==1.2.3` (Python) and `@mediapipe/tasks-vision@1.2.3` (browser).

**Phone FPS test:** https://yahyasarhan22.github.io/psl-toolchain/fps_test/

No dataset videos or signer data go in this repo.
