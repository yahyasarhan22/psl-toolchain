// PSL landmark extractor for the browser (subset-v2).
// Twin of python/extract_landmarks.py: same models, settings, face crop maths, hand slots and frame layout.
// Used by the parity test now and by the live app (C3) later.
//
//   const ex = await createExtractor({ lib, models, delegate: "GPU" });
//   const vec = ex.processFrame(sourceCanvas, frameIndex, timestampMs);  // 257 values, null = missing
//   ex.close();

export const SUBSET_VERSION = "subset-v2";
export const TASKS_VISION_VERSION = "1.2.3";
export const POSE_IDX = [...Array(17).keys()];
export const FACE_IDX = [10, 1, 152, 234, 454, 70, 105, 107, 300, 334, 336, 159, 145, 386, 374, 61, 291, 0, 17, 13, 14];
export const FACE_EVERY = 3;
export const FACE_CROP_SCALE = 3.0;
export const FACE_CROP_MIN = 64;
export const N_VALUES = 17 * 4 + 21 * 3 * 2 + 21 * 3; // 257
export const MODEL_SHA256 = {
  "hand_landmarker.task": "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1",
  "pose_landmarker_lite.task": "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a",
  "face_landmarker.task": "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
};

async function sha256Hex(buf) {
  const h = await crypto.subtle.digest("SHA-256", buf);
  return [...new Uint8Array(h)].map(b => b.toString(16).padStart(2, "0")).join("");
}

async function loadModel(modelsBase, name) {
  const r = await fetch(`${modelsBase}${name}`);
  if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`);
  const buf = await r.arrayBuffer();
  const got = await sha256Hex(buf);
  if (got !== MODEL_SHA256[name]) throw new Error(`${name}: hash mismatch ${got}`);
  return new Uint8Array(buf);
}

export function faceCropBox(pose, W, H) {
  const fl = Math.floor;
  const ex = i => [pose[i].x * W, pose[i].y * H];
  const [x7, y7] = ex(7), [x8, y8] = ex(8), [x3, y3] = ex(3), [x6, y6] = ex(6);
  const head = Math.max(Math.hypot(x7 - x8, y7 - y8), 1.5 * Math.hypot(x3 - x6, y3 - y6));
  let side = Math.max(fl(FACE_CROP_SCALE * head + 0.5), FACE_CROP_MIN);
  side = Math.min(side, W, H);
  const cx = (x7 + x8) / 2, cy = (y7 + y8) / 2;
  const x0 = Math.min(Math.max(fl(cx - side / 2 + 0.5), 0), W - side);
  const y0 = Math.min(Math.max(fl(cy - side / 2 + 0.5), 0), H - side);
  return [x0, y0, side];
}

export function assignHands(hands, pose) {
  if (!hands || hands.length === 0) return [null, null];
  if (!pose) return [hands[0] ?? null, hands[1] ?? null];
  const lw = pose[15], rw = pose[16];
  const d = (h, w) => Math.hypot(h[0].x - w.x, h[0].y - w.y);
  if (hands.length === 1) {
    const h = hands[0];
    return d(h, lw) <= d(h, rw) ? [h, null] : [null, h];
  }
  const scored = hands.map((h, i) => [d(h, lw) - d(h, rw), i]).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  return [hands[scored[0][1]], hands[scored[scored.length - 1][1]]];
}

export function frameVector(pose, left, right, face) {
  const v = [];
  if (!pose) for (let k = 0; k < 68; k++) v.push(null);
  else for (const i of POSE_IDX) { const p = pose[i]; v.push(p.x, p.y, p.z, p.visibility ?? null); }
  for (const h of [left, right]) {
    if (!h) for (let k = 0; k < 63; k++) v.push(null);
    else for (const p of h) v.push(p.x, p.y, p.z);
  }
  if (!face) for (let k = 0; k < 63; k++) v.push(null);
  else for (const i of FACE_IDX) { const p = face[i]; v.push(p.x, p.y, p.z); }
  if (v.length !== N_VALUES) throw new Error("bad vector length " + v.length);
  return v;
}

export async function createExtractor({ lib, models, delegate = "GPU" }) {
  const { FilesetResolver, HandLandmarker, PoseLandmarker, FaceLandmarker } = await import(`${lib}vision_bundle.mjs`);
  const vision = await FilesetResolver.forVisionTasks(`${lib}wasm`);
  const [hb, pb, fb] = await Promise.all(Object.keys(MODEL_SHA256).map(n => loadModel(models, n)));
  const hand = await HandLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetBuffer: hb, delegate }, runningMode: "VIDEO", numHands: 2,
    minHandDetectionConfidence: 0.5, minHandPresenceConfidence: 0.5, minTrackingConfidence: 0.5,
  });
  const pose = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetBuffer: pb, delegate }, runningMode: "VIDEO", numPoses: 1, outputSegmentationMasks: false,
    minPoseDetectionConfidence: 0.5, minPosePresenceConfidence: 0.5, minTrackingConfidence: 0.5,
  });
  const face = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetBuffer: fb, delegate }, runningMode: "IMAGE", numFaces: 1, outputFaceBlendshapes: false,
    minFaceDetectionConfidence: 0.5, minFacePresenceConfidence: 0.5, minTrackingConfidence: 0.5,
  });
  let lastFace = null;

  function runFace(src, p) {
    if (!p) return null;
    const W = src.width, H = src.height;
    const [x0, y0, side] = faceCropBox(p, W, H);
    const ctx = src.getContext("2d", { willReadFrequently: true });
    const crop = ctx.getImageData(x0, y0, side, side);
    const r = face.detect(crop);
    if (!r.faceLandmarks?.length) return null;
    return r.faceLandmarks[0].map(q => ({ x: (x0 + q.x * side) / W, y: (y0 + q.y * side) / H, z: q.z * side / W }));
  }

  return {
    delegate,
    // src: a canvas holding the current RGB frame (unmirrored). Returns 257 values.
    processFrame(src, i, ts) {
      const rh = hand.detectForVideo(src, ts);
      const rp = pose.detectForVideo(src, ts);
      const p = rp.landmarks?.[0] ?? null;
      if (i % FACE_EVERY === 0) lastFace = runFace(src, p);
      const [left, right] = assignHands(rh.landmarks ?? [], p);
      return frameVector(p, left, right, lastFace);
    },
    close() { hand.close(); pose.close(); face.close(); },
  };
}
