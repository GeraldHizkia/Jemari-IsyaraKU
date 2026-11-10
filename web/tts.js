/**
 * Modul Sintesis Suara Bahasa Indonesia (TTS)
 */

export class TextToSpeechManager {
  constructor() {
    this.voice = null;
    this.isSupported = "speechSynthesis" in window;
    this.rate = 0.95;
    this.pitch = 1.0;
    this.initVoices();
  }

  initVoices() {
    if (!this.isSupported) return;

    const findVoice = () => {
      const voices = window.speechSynthesis.getVoices() || [];
      // Prioritaskan suara Bahasa Indonesia
      this.voice =
        voices.find((v) => v.lang?.toLowerCase().replace("_", "-").startsWith("id")) ||
        voices.find((v) => v.lang?.toLowerCase().startsWith("id")) ||
        voices.find((v) => v.name?.toLowerCase().includes("indonesia")) ||
        voices.find((v) => v.lang?.toLowerCase().startsWith("en")) ||
        voices[0] ||
        null;
    };

    findVoice();
    if (window.speechSynthesis.onvoiceschanged !== undefined) {
      window.speechSynthesis.onvoiceschanged = findVoice;
    }
  }

  speak(text) {
    if (!this.isSupported || !text || !text.trim()) return;

    window.speechSynthesis.cancel(); // batalkan ucapan sebelumnya jika ada
    const utterance = new SpeechSynthesisUtterance(text.trim());
    utterance.lang = "id-ID";
    if (this.voice) {
      utterance.voice = this.voice;
    }
    utterance.rate = this.rate;
    utterance.pitch = this.pitch;
    window.speechSynthesis.speak(utterance);
  }

  stop() {
    if (this.isSupported) {
      window.speechSynthesis.cancel();
    }
  }
}
