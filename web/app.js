/**
 * Jemari-IsyaraKU 2.0 — Orkestrator Komunikasi Dua Arah Inklusif
 * Mengintegrasikan:
 * 1. Mode Kosakata WL-BISINDO (32 Kata Real, Model Bi-LSTM Akurasi 99.38%)
 * 2. Mode Alfabet (26 Huruf A-Z Jemari)
 * 3. Mode Suara Teman Dengar (Speech-to-Text via Web Speech Recognition)
 * 4. Sintesis Suara Text-to-Speech (TTS) Bahasa Indonesia
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
const tabHearing = document.getElementById("tabHearing");

const cameraCard = document.getElementById("cameraCard");
const hearingCard = document.getElementById("hearingCard");
const micToggleBtn = document.getElementById("micToggleBtn");
const micBtnText = document.getElementById("micBtnText");
const sttStatus = document.getElementById("sttStatus");
const sttInterimText = document.getElementById("sttInterimText");

const detectionChip = document.getElementById("detectionChip");
const bufferChip = document.getElementById("bufferChip");
const predModeTitle = document.getElementById("predModeTitle");
const currentPrediction = document.getElementById("currentPrediction");
const confidenceFill = document.getElementById("confidenceFill");
const confidenceText = document.getElementById("confidenceText");
const statusHint = document.getElementById("statusHint");

const topAlternatives = document.getElementById("topAlternatives");
const altChips = document.getElementById("altChips");

const transcriptBox = document.getElementById("transcriptBox");
const speakBtn = document.getElementById("speakBtn");
const copyBtn = document.getElementById("copyBtn");
const spaceBtn = document.getElementById("spaceBtn");
const backspaceBtn = document.getElementById("backspaceBtn");
const clearBtn = document.getElementById("clearBtn");

const backendDot = document.getElementById("backendDot");
const backendStatus = document.getElementById("backendStatus");
const guideBtn = document.getElementById("guideBtn");
const vocabModal = document.getElementById("vocabModal");
const closeVocabBtn = document.getElementById("closeVocabBtn");
const vocabGrid = document.getElementById("vocabGrid");

// State
const state = {
  mode: "word", // 'word', 'alphabet', 'hearing'
  running: false,
  landmarker: null,
  rafId: null,
  lastVideoTime: -1,
  transcript: "",
  alphabetRecognizer: new AlphabetRecognizer(),
  wordRecognizer: new WordRecognizerClient(""),
  tts: new TextToSpeechManager(),
  recognition: null,
  isListening: false,
};

/* ----------------------- Inisialisasi Speech STT ----------------------- */

function initSpeechRecognition() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    console.warn("Browser tidak mendukung Web Speech Recognition API");
    return;
  }

  const rec = new SpeechRecognition();
  rec.continuous = true;
  rec.interimResults = true;
  rec.lang = "id-ID";

  rec.onstart = () => {
    state.isListening = true;
    sttStatus.textContent = "Sedang Mendengarkan...";
    sttStatus.style.background = "var(--accent-red)";
    micBtnText.textContent = "Hentikan Bicara";
    micToggleBtn.classList.add("listening");
  };

  rec.onresult = (event) => {
    let interim = "";
    let final = "";

    for (let i = event.resultIndex; i < event.results.length; ++i) {
      if (event.results[i].isFinal) {
        final += event.results[i][0].transcript;
      } else {
        interim += event.results[i][0].transcript;
      }
    }

    if (interim) {
      sttInterimText.textContent = interim;
    }

    if (final) {
      appendTranscript(`[Teman Dengar]: "${final.trim()}"`, true);
      sttInterimText.textContent = "–";
    }
  };

  rec.onerror = (e) => {
    console.warn("STT Error:", e.error);
    stopListening();
  };

  rec.onend = () => {
    if (state.isListening && state.mode === "hearing") {
      try { rec.start(); } catch (err) {}
    } else {
      stopListening();
    }
  };

  state.recognition = rec;
}

function startListening() {
  if (!state.recognition) {
    alert("Speech Recognition tidak didukung oleh browser Anda. Gunakan Google Chrome atau Edge.");
    return;
  }
  try {
    state.recognition.start();
  } catch (err) {
    console.warn(err);
  }
}

function stopListening() {
  state.isListening = false;
  if (state.recognition) {
    try { state.recognition.stop(); } catch (err) {}
  }
  sttStatus.textContent = "Siap Merekam";
  sttStatus.style.background = "";
  micBtnText.textContent = "Mulai Berbicara";
  micToggleBtn.classList.remove("listening");
}

micToggleBtn.addEventListener("click", () => {
  if (state.isListening) {
    stopListening();
  } else {
    startListening();
  }
});

/* ---------------------------- Inisialisasi ---------------------------- */

async function checkBackendHealth() {
  try {
    const res = await fetch("/api/health");
    if (res.ok) {
      const data = await res.json();
      backendDot.classList.remove("offline");
      backendStatus.textContent = `Backend Online (Model Akurasi 99.38%)`;
    } else {
      throw new Error("HTTP " + res.status);
    }
  } catch (e) {
    backendDot.classList.add("offline");
    backendStatus.textContent = "Backend: Offline";
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
    tag.title = "Klik untuk dengarkan lafal audio";
    tag.addEventListener("click", () => state.tts.speak(w.display || w.gloss));
    vocabGrid.appendChild(tag);
  });
}

/* -------------------------- Pergantian Mode --------------------------- */

function setMode(newMode) {
  state.mode = newMode;
  tabWord.classList.remove("active");
  tabAlphabet.classList.remove("active");
  tabHearing.classList.remove("active");

  if (newMode === "word") {
    tabWord.classList.add("active");
    cameraCard.style.display = "flex";
    hearingCard.style.display = "none";
    predModeTitle.textContent = "Kosakata Terdeteksi";
    bufferChip.style.display = "inline-block";
    captureWordBtn.style.display = "inline-flex";
    statusHint.textContent = "Mode Kosakata (32 Kata Real WL-BISINDO)";
    state.alphabetRecognizer.reset();
    stopListening();
  } else if (newMode === "alphabet") {
    tabAlphabet.classList.add("active");
    cameraCard.style.display = "flex";
    hearingCard.style.display = "none";
    predModeTitle.textContent = "Huruf Terdeteksi";
    bufferChip.style.display = "none";
    captureWordBtn.style.display = "none";
    statusHint.textContent = "Mode Alfabet (A–Z Jemari)";
    state.wordRecognizer.resetBuffer();
    topAlternatives.style.display = "none";
    stopListening();
  } else if (newMode === "hearing") {
    tabHearing.classList.add("active");
    cameraCard.style.display = "none";
    hearingCard.style.display = "flex";
    statusHint.textContent = "Mode Teman Dengar (Bicara ke Teks)";
    if (state.running) stop();
  }

  currentPrediction.textContent = "–";
  confidenceFill.style.width = "0%";
  confidenceText.textContent = "Keyakinan: 0%";
}

tabWord.addEventListener("click", () => setMode("word"));
tabAlphabet.addEventListener("click", () => setMode("alphabet"));
tabHearing.addEventListener("click", () => setMode("hearing"));

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
      state.landmarker = await createHandLandmarker(2); // 2 tangan untuk gesture BISINDO
    }
    await state.alphabetRecognizer.load();
    await startCamera(video);
    resizeOverlay();

    state.running = true;
    stopBtn.disabled = false;
    detectionChip.textContent = "Kamera Aktif";
    statusHint.textContent = "Siap mendeteksi isyarat";
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
  topAlternatives.style.display = "none";
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
        checkAutoWordTrigger();
      }
    }
  }

  state.rafId = requestAnimationFrame(loop);
}

function handleFrame(multiLandmarks, handednesses) {
  if (state.mode === "alphabet") {
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
  } else if (state.mode === "word") {
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
  statusHint.textContent = `Kata terdeteksi: ${res.label} (${confPct}%)`;

  // Tampilkan alternatif top-3
  if (res.top_k && res.top_k.length > 1) {
    altChips.innerHTML = "";
    res.top_k.forEach((alt) => {
      const chip = document.createElement("span");
      chip.className = "alt-chip";
      chip.textContent = `${alt.display} (${Math.round(alt.confidence * 100)}%)`;
      chip.addEventListener("click", () => {
        appendTranscript(alt.display, true);
        state.tts.speak(alt.display);
      });
      altChips.appendChild(chip);
    });
    topAlternatives.style.display = "flex";
  }

  appendTranscript(res.label, true);
  state.tts.speak(res.label);
}

captureWordBtn.addEventListener("click", async () => {
  if (state.mode !== "word") return;
  statusHint.textContent = "Mengevaluasi buffer isyarat...";
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
        Isyarat yang terdeteksi atau suara dari lawan bicara akan tampil di papan ini...
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

// Modal Panduan
guideBtn.addEventListener("click", () => vocabModal.classList.add("open"));
closeVocabBtn.addEventListener("click", () => vocabModal.classList.remove("open"));
vocabModal.addEventListener("click", (e) => {
  if (e.target === vocabModal) vocabModal.classList.remove("open");
});

startBtn.addEventListener("click", start);
stopBtn.addEventListener("click", stop);

// Inisialisasi awal
checkBackendHealth();
initVocabularyModal();
initSpeechRecognition();
setMode("word");
