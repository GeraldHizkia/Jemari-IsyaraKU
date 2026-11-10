/**
 * Jemari BISINDO Translator 2.0 — Orchestrator Utama
 * Menggabungkan Mode Kosakata (WL-BISINDO) dan Mode Alfabet (A-Z Jemari)
 */

import { createHandLandmarker, startCamera, stopCamera, drawMultiHands } from "./hands.js";
import { AlphabetRecognizer } from "./alphabet.js";
import { WordRecognizerClient } from "./wordRecognition.js";
import { TextToSpeechManager } from "./tts.js";

// DOM Elements
const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const captureWordBtn = document.getElementById("captureWordBtn");
const tabWord = document.getElementById("tabWord");
const tabAlphabet = document.getElementById("tabAlphabet");

const detectionChip = document.getElementById("detectionChip");
const bufferChip = document.getElementById("bufferChip");
const predModeTitle = document.getElementById("predModeTitle");
const currentPrediction = document.getElementById("currentPrediction");
const confidenceFill = document.getElementById("confidenceFill");
const confidenceText = document.getElementById("confidenceText");
const statusHint = document.getElementById("statusHint");

const transcriptBox = document.getElementById("transcriptBox");
const speakBtn = document.getElementById("speakBtn");
const copyBtn = document.getElementById("copyBtn");
const spaceBtn = document.getElementById("spaceBtn");
const backspaceBtn = document.getElementById("backspaceBtn");
const clearBtn = document.getElementById("clearBtn");

const backendDot = document.getElementById("backendDot");
const backendStatus = document.getElementById("backendStatus");
const vocabBtn = document.getElementById("vocabBtn");
const vocabModal = document.getElementById("vocabModal");
const closeVocabBtn = document.getElementById("closeVocabBtn");
const vocabGrid = document.getElementById("vocabGrid");

// State
const state = {
  mode: "word", // 'word' atau 'alphabet'
  running: false,
  landmarker: null,
  rafId: null,
  lastVideoTime: -1,
  transcript: "",
  alphabetRecognizer: new AlphabetRecognizer(),
  wordRecognizer: new WordRecognizerClient(""),
  tts: new TextToSpeechManager()
};

/* ---------------------------- Inisialisasi ---------------------------- */

async function checkBackendHealth() {
  try {
    const res = await fetch("/api/health");
    if (res.ok) {
      const data = await res.json();
      backendDot.classList.remove("offline");
      backendStatus.textContent = `Backend: Terhubung (${data.word_classes_count} kelas)`;
    } else {
      throw new Error("HTTP " + res.status);
    }
  } catch (e) {
    backendDot.classList.add("offline");
    backendStatus.textContent = "Backend: Offline / Standalone";
  }
}

async function initVocabularyModal() {
  const words = await state.wordRecognizer.loadVocabulary();
  vocabGrid.innerHTML = "";
  words.forEach((w) => {
    const tag = document.createElement("div");
    tag.className = "vocab-tag";
    tag.style.cursor = "pointer";
    tag.innerHTML = `
      <span class="vocab-tag-name">${w.display || w.gloss}</span>
      <span class="vocab-tag-cat">${w.category || "Kosakata"}</span>
    `;
    tag.title = "Klik untuk dengarkan lafal";
    tag.addEventListener("click", () => state.tts.speak(w.display || w.gloss));
    vocabGrid.appendChild(tag);
  });
}

/* -------------------------- Pergantian Mode --------------------------- */

function setMode(newMode) {
  state.mode = newMode;
  if (newMode === "word") {
    tabWord.classList.add("active");
    tabAlphabet.classList.remove("active");
    predModeTitle.textContent = "Kosakata Terdeteksi";
    bufferChip.style.display = "inline-block";
    captureWordBtn.style.display = "inline-flex";
    statusHint.textContent = "Mode Kosakata (WL-BISINDO)";
    state.alphabetRecognizer.reset();
  } else {
    tabAlphabet.classList.add("active");
    tabWord.classList.remove("active");
    predModeTitle.textContent = "Huruf Terdeteksi";
    bufferChip.style.display = "none";
    captureWordBtn.style.display = "none";
    statusHint.textContent = "Mode Alfabet (A–Z)";
    state.wordRecognizer.resetBuffer();
  }
  currentPrediction.textContent = "–";
  confidenceFill.style.width = "0%";
  confidenceText.textContent = "Keyakinan: 0%";
}

tabWord.addEventListener("click", () => setMode("word"));
tabAlphabet.addEventListener("click", () => setMode("alphabet"));

/* ------------------------- Alur Kamera & Loop ------------------------- */

function resizeOverlay() {
  overlay.width = video.videoWidth || 640;
  overlay.height = video.videoHeight || 480;
}

async function start() {
  startBtn.disabled = true;
  detectionChip.textContent = "Menyiapkan MediaPipe...";
  statusHint.textContent = "Memuat model...";

  try {
    if (!state.landmarker) {
      state.landmarker = await createHandLandmarker(2); // 2 tangan untuk WL-BISINDO
    }
    await state.alphabetRecognizer.load();
    await startCamera(video);
    resizeOverlay();

    state.running = true;
    stopBtn.disabled = false;
    detectionChip.textContent = "Kamera Aktif";
    statusHint.textContent = "Siap mendeteksi gestur";
    loop();
  } catch (err) {
    console.error("Gagal menyalakan kamera:", err);
    startBtn.disabled = false;
    detectionChip.textContent = "Gagal memulai kamera";
    alert(`Gagal mengakses kamera/MediaPipe: ${err.message}`);
  }
}

function stop() {
  state.running = false;
  if (state.rafId) cancelAnimationFrame(state.rafId);
  stopCamera(video);
  drawMultiHands(ctx, null, null, overlay.width, overlay.height);

  startBtn.disabled = false;
  stopBtn.disabled = true;
  detectionChip.textContent = "Kamera Berhenti";
  statusHint.textContent = "Standby";
  currentPrediction.textContent = "–";
  confidenceFill.style.width = "0%";
  confidenceText.textContent = "Keyakinan: 0%";
}

function loop() {
  if (!state.running) return;

  if (video.currentTime !== state.lastVideoTime && video.readyState >= 2) {
    state.lastVideoTime = video.currentTime;
    const timestamp = performance.now();
    const result = state.landmarker.detectForVideo(video, timestamp);

    const multiLandmarks = result.landmarks || [];
    const handednesses = result.handednesses || [];

    // Render skeleton pada canvas overlay
    drawMultiHands(ctx, multiLandmarks, handednesses, overlay.width, overlay.height);

    if (multiLandmarks.length > 0) {
      detectionChip.textContent = `Terdeteksi ${multiLandmarks.length} Tangan`;
      handleFrame(multiLandmarks, handednesses);
    } else {
      detectionChip.textContent = "Arahkan tangan ke kamera";
      if (state.mode === "word") {
        // Tangan keluar dari frame -> coba auto-predict buffer
        checkAutoWordTrigger();
      }
    }
  }

  state.rafId = requestAnimationFrame(loop);
}

function handleFrame(multiLandmarks, handednesses) {
  if (state.mode === "alphabet") {
    // Mode Alfabet: gunakan tangan dominan pertama
    const res = state.alphabetRecognizer.predict(multiLandmarks[0]);
    if (res && res.current) {
      currentPrediction.textContent = res.current.label;
      const confPct = Math.round(res.current.confidence * 100);
      confidenceFill.style.width = `${confPct}%`;
      confidenceText.textContent = `Keyakinan: ${confPct}%`;
    }
    if (res && res.stable) {
      appendTranscript(res.stable.label, false);
    }
  } else {
    // Mode Kosakata: buffer frame 126D
    state.wordRecognizer.pushFrame(multiLandmarks, handednesses);
    const bufLen = state.wordRecognizer.buffer.length;
    bufferChip.textContent = `Buffer: ${bufLen} frame`;
  }
}

async function checkAutoWordTrigger() {
  const predPromise = state.wordRecognizer.pushFrame([], []);
  if (predPromise) {
    statusHint.textContent = "Menganalisis kata...";
    const res = await predPromise;
    if (res && res.label) {
      onWordRecognized(res);
    }
  }
}

async function onWordRecognized(res) {
  currentPrediction.textContent = res.label;
  const confPct = Math.round(res.confidence * 100);
  confidenceFill.style.width = `${confPct}%`;
  confidenceText.textContent = `Keyakinan: ${confPct}%`;
  statusHint.textContent = `Kata terdeteksi: ${res.label}`;

  appendTranscript(res.label, true);
  state.tts.speak(res.label);
}

// Tombol manual "Tangkap Gestur Kata"
captureWordBtn.addEventListener("click", async () => {
  if (state.mode !== "word") return;
  statusHint.textContent = "Mengevaluasi buffer kata...";
  try {
    const res = await state.wordRecognizer.triggerManualPrediction();
    if (res && res.label) {
      onWordRecognized(res);
    }
  } catch (err) {
    alert(err.message);
  }
});

/* ----------------------- Manajemen Transkrip -------------------------- */

function appendTranscript(text, isWord = true) {
  if (isWord) {
    if (state.transcript.length > 0 && !state.transcript.endsWith(" ")) {
      state.transcript += " ";
    }
    state.transcript += text + " ";
  } else {
    state.transcript += text;
  }
  renderTranscript();
}

function renderTranscript() {
  if (!state.transcript.trim()) {
    transcriptBox.innerHTML = `
      <span class="transcript-placeholder">
        Gerakkan tangan di depan kamera. Kata atau huruf yang berhasil diterjemahkan akan tampil di sini...
      </span>`;
  } else {
    transcriptBox.textContent = state.transcript;
    transcriptBox.scrollTop = transcriptBox.scrollHeight;
  }
}

spaceBtn.addEventListener("click", () => {
  state.transcript += " ";
  renderTranscript();
});

backspaceBtn.addEventListener("click", () => {
  if (state.transcript.length > 0) {
    state.transcript = state.transcript.trimEnd().slice(0, -1);
    renderTranscript();
  }
});

clearBtn.addEventListener("click", () => {
  state.transcript = "";
  renderTranscript();
});

copyBtn.addEventListener("click", () => {
  if (!state.transcript.trim()) return;
  navigator.clipboard.writeText(state.transcript.trim()).then(() => {
    const orig = copyBtn.textContent;
    copyBtn.textContent = "✓ Tersalin!";
    setTimeout(() => (copyBtn.textContent = orig), 1500);
  });
});

speakBtn.addEventListener("click", () => {
  state.tts.speak(state.transcript);
});

// Modal Kosakata
vocabBtn.addEventListener("click", () => vocabModal.classList.add("open"));
closeVocabBtn.addEventListener("click", () => vocabModal.classList.remove("open"));
vocabModal.addEventListener("click", (e) => {
  if (e.target === vocabModal) vocabModal.classList.remove("open");
});

startBtn.addEventListener("click", start);
stopBtn.addEventListener("click", stop);

// Inisialisasi awal
checkBackendHealth();
initVocabularyModal();
setMode("word");
