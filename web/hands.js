/**
 * MediaPipe Tasks Vision Wrapper untuk Jemari BISINDO Translator.
 * Mendukung deteksi hingga 2 tangan sekaligus (numHands: 2) untuk
 * pengenalan kosakata kata BISINDO yang membutuhkan gestur dua tangan.
 */

const MP_VERSION = "0.10.14";
const MP_CDN = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}`;
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

const PREFERRED_DELEGATE =
  new URLSearchParams(location.search).get("delegate") === "cpu" ? "CPU" : "GPU";

export async function createHandLandmarker(numHands = 2) {
  const vision = await import(/* @vite-ignore */ MP_CDN);
  const fileset = await vision.FilesetResolver.forVisionTasks(`${MP_CDN}/wasm`);

  const base = {
    baseOptions: { modelAssetPath: MODEL_URL },
    runningMode: "VIDEO",
    numHands: numHands,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  };

  try {
    return await vision.HandLandmarker.createFromOptions(fileset, {
      ...base,
      baseOptions: { ...base.baseOptions, delegate: PREFERRED_DELEGATE },
    });
  } catch (err) {
    console.warn(`${PREFERRED_DELEGATE} delegate gagal, fallback ke CPU:`, err);
    return await vision.HandLandmarker.createFromOptions(fileset, {
      ...base,
      baseOptions: { ...base.baseOptions, delegate: "CPU" },
    });
  }
}

export async function startCamera(videoEl) {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
    audio: false,
  });
  videoEl.srcObject = stream;
  await new Promise((resolve) => {
    videoEl.onloadedmetadata = () => {
      videoEl.play();
      resolve();
    };
  });
  return stream;
}

export function stopCamera(videoEl) {
  const stream = videoEl.srcObject;
  if (stream) stream.getTracks().forEach((t) => t.stop());
  videoEl.srcObject = null;
}

const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

export function drawMultiHands(ctx, multiLandmarks, handednesses, width, height) {
  ctx.clearRect(0, 0, width, height);
  if (!multiLandmarks || multiLandmarks.length === 0) return;

  for (let h = 0; h < multiLandmarks.length; h++) {
    const landmarks = multiLandmarks[h];
    const handedness = handednesses?.[h]?.[0]?.categoryName || "Right";
    const strokeColor = handedness === "Left" ? "#10b981" : "#3b82f6"; // Hijau untuk kiri, biru untuk kanan

    // Skeleton connections
    ctx.lineWidth = 3;
    ctx.strokeStyle = strokeColor;
    ctx.beginPath();
    for (const [a, b] of HAND_CONNECTIONS) {
      ctx.moveTo(landmarks[a].x * width, landmarks[a].y * height);
      ctx.lineTo(landmarks[b].x * width, landmarks[b].y * height);
    }
    ctx.stroke();

    // Landmark keypoints
    for (let i = 0; i < landmarks.length; i++) {
      const lm = landmarks[i];
      ctx.beginPath();
      ctx.arc(lm.x * width, lm.y * height, i === 0 ? 5 : 3.5, 0, 2 * Math.PI);
      ctx.fillStyle = i === 0 ? "#f59e0b" : "#ffffff";
      ctx.fill();
      ctx.strokeStyle = "#1e293b";
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  }
}
