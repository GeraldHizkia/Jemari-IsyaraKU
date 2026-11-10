/**
 * Modul Pengenalan Kosakata BISINDO (32 Kata WL-BISINDO)
 * Mengumpulkan buffer sekuens temporal 126D dan berkomunikasi dengan backend inferensi.
 */

import { normalizeSingleHandLandmarks } from "./alphabet.js";

export class WordRecognizerClient {
  constructor(apiBaseUrl = "") {
    this.apiBaseUrl = apiBaseUrl;
    this.buffer = [];
    this.maxBufferFrames = 40;
    this.minBufferFrames = 15;
    this.isRecording = false;
    this.autoMode = true;
    this.lastPredictionTime = 0;
    this.cooldownMs = 1200;
    this.vocabulary = [];
    this.isPredicting = false;
  }

  async loadVocabulary() {
    try {
      const res = await fetch(`${this.apiBaseUrl}/api/vocab`);
      if (res.ok) {
        const data = await res.json();
        this.vocabulary = data.words || [];
        return this.vocabulary;
      }
    } catch (e) {
      console.warn("Gagal memuat kosakata dari backend, menggunakan file statis...", e);
      try {
        const res2 = await fetch("models/word/labels.json");
        if (res2.ok) {
          this.vocabulary = await res2.json();
          return this.vocabulary;
        }
      } catch (err2) {
        console.error("Gagal memuat labels.json statis:", err2);
      }
    }
    return [];
  }

  resetBuffer() {
    this.buffer = [];
  }

  extract126D(multiLandmarks, handednesses) {
    let leftFeatures = new Array(63).fill(0);
    let rightFeatures = new Array(63).fill(0);

    if (multiLandmarks && multiLandmarks.length > 0) {
      for (let i = 0; i < multiLandmarks.length; i++) {
        const lm = multiLandmarks[i];
        const handedness = handednesses?.[i]?.[0]?.categoryName || "Right";
        const norm = normalizeSingleHandLandmarks(lm);
        if (handedness === "Left") {
          leftFeatures = norm;
        } else {
          rightFeatures = norm;
        }
      }
    }

    return leftFeatures.concat(rightFeatures);
  }

  pushFrame(multiLandmarks, handednesses) {
    const hasHand = multiLandmarks && multiLandmarks.length > 0;
    if (!hasHand) {
      // Jika tangan tidak terlihat di kamera
      if (this.buffer.length > 0) {
        // Tangan baru saja keluar dari frame -> potensi akhir dari gestur kata
        if (this.autoMode && this.buffer.length >= this.minBufferFrames && !this.isPredicting) {
          const now = Date.now();
          if (now - this.lastPredictionTime > this.cooldownMs) {
            this.lastPredictionTime = now;
            const seqToPredict = [...this.buffer];
            this.buffer = [];
            return this.sendPrediction(seqToPredict);
          }
        }
      }
      return null;
    }

    const frameFeatures = this.extract126D(multiLandmarks, handednesses);
    this.buffer.push(frameFeatures);

    if (this.buffer.length > this.maxBufferFrames) {
      this.buffer.shift();
    }

    return null;
  }

  async sendPrediction(sequence) {
    if (!sequence || sequence.length < this.minBufferFrames) return null;
    this.isPredicting = true;
    try {
      const res = await fetch(`${this.apiBaseUrl}/api/predict/word`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sequence, top_k: 3 }),
      });
      if (!res.ok) {
        throw new Error(`Server returned ${res.status}`);
      }
      const data = await res.json();
      return data;
    } catch (err) {
      console.error("Gagal inferensi kata:", err);
      return null;
    } finally {
      this.isPredicting = false;
    }
  }

  async triggerManualPrediction() {
    if (this.buffer.length < this.minBufferFrames) {
      throw new Error(`Buffer gestur belum cukup (baru ${this.buffer.length}/${this.minBufferFrames} frame). Lakukan gerakan gestur di depan kamera.`);
    }
    const seq = [...this.buffer];
    this.buffer = [];
    return await this.sendPrediction(seq);
  }
}
