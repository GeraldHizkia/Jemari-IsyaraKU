/**
 * Modul Pengenalan Alfabet BISINDO (A-Z)
 * Menjalankan inferensi MLP 26 kelas secara client-side di browser.
 */

export function normalizeSingleHandLandmarks(landmarks) {
  if (!landmarks || landmarks.length === 0) return new Array(63).fill(0);

  const wrist = landmarks[0];
  const translated = landmarks.map((lm) => [lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z]);

  let scale = 0;
  for (const [x, y, z] of translated) {
    const d = Math.sqrt(x * x + y * y + z * z);
    if (d > scale) scale = d;
  }
  if (scale < 1e-6) scale = 1;

  const out = new Array(63);
  let i = 0;
  for (const [x, y, z] of translated) {
    out[i++] = x / scale;
    out[i++] = y / scale;
    out[i++] = z / scale;
  }
  return out;
}

function applyScaler(features, scaler) {
  if (!scaler) return features;
  return features.map((v, i) => (v - scaler.mean[i]) / (scaler.scale[i] || 1));
}

function relu(x) {
  return x > 0 ? x : 0;
}

function softmax(values) {
  const max = Math.max(...values);
  const exps = values.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

function mlpPredict(model, features) {
  let vec = applyScaler(features, model.scaler);

  for (let li = 0; li < model.layers.length; li++) {
    const layer = model.layers[li];
    const isLast = li === model.layers.length - 1;

    const out = layer.bias.slice();
    for (let i = 0; i < vec.length; i++) {
      const row = layer.weights[i];
      const v = vec[i];
      for (let o = 0; o < row.length; o++) out[o] += v * row[o];
    }

    if (!isLast) {
      for (let o = 0; o < out.length; o++) out[o] = relu(out[o]);
    }
    vec = out;
  }

  const probs = model.out_activation === "softmax" ? softmax(vec) : vec;
  let index = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[index]) index = i;

  return { index, confidence: probs[index] };
}

export class AlphabetRecognizer {
  constructor() {
    this.model = null;
    this.history = [];
    this.windowSize = 7;
    this.minConfidence = 0.55;
    this.lastEmitted = null;
    this.cooldown = 0;
  }

  async load(url = "models/alphabet/model.json") {
    try {
      const res = await fetch(url, { cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      this.model = await res.json();
      return this.model;
    } catch (err) {
      console.warn("Alphabet model gagal dimuat:", err);
      return null;
    }
  }

  reset() {
    this.history = [];
    this.lastEmitted = null;
    this.cooldown = 0;
  }

  predict(landmarks) {
    if (!this.model || !landmarks) return null;
    const features = normalizeSingleHandLandmarks(landmarks);
    const res = mlpPredict(this.model, features);
    const label = this.model.classes[res.index];

    // Temporal smoothing
    this.history.push({ label, confidence: res.confidence });
    if (this.history.length > this.windowSize) this.history.shift();

    let stableEmission = null;
    if (this.cooldown > 0) {
      this.cooldown--;
    } else if (this.history.length >= this.windowSize) {
      const counts = {};
      let totalConf = 0;
      for (const item of this.history) {
        counts[item.label] = (counts[item.label] || 0) + 1;
        totalConf += item.confidence;
      }
      const avgConf = totalConf / this.history.length;
      let majorityLabel = null;
      let majorityCount = 0;
      for (const [l, count] of Object.entries(counts)) {
        if (count > majorityCount) {
          majorityCount = count;
          majorityLabel = l;
        }
      }

      if (majorityCount >= Math.ceil(this.windowSize * 0.7) && avgConf >= this.minConfidence) {
        if (majorityLabel !== this.lastEmitted) {
          stableEmission = { label: majorityLabel, confidence: avgConf };
          this.lastEmitted = majorityLabel;
          this.cooldown = 12; // cegah duplikasi huruf berulang
        }
      }
    }

    return {
      current: { label, confidence: res.confidence },
      stable: stableEmission
    };
  }
}
