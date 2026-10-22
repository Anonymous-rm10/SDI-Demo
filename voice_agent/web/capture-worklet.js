/* Streaming linear resampling to mono 16 kHz PCM16; no browser-rate assumption. */
class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.count = 0;
    this.next = 0;
    this.previous = 0;
    this.buffer = new Int16Array(1600);
    this.used = 0;
  }
  process(inputs) {
    const channels = inputs[0];
    if (!channels?.length) return true;
    for (let i = 0; i < channels[0].length; i++) {
      let x = 0;
      for (const channel of channels) x += channel[i] / channels.length;
      while (this.next <= this.count) {
        const fraction = this.next - this.count + 1;
        const value = this.previous + (x - this.previous) * fraction;
        this.buffer[this.used++] = Math.round(Math.max(-1, Math.min(1, value)) * 32767);
        this.next += sampleRate / 16000;
        if (this.used === this.buffer.length) {
          this.port.postMessage(this.buffer.buffer, [this.buffer.buffer]);
          this.buffer = new Int16Array(1600);
          this.used = 0;
        }
      }
      this.previous = x;
      this.count++;
    }
    return true;
  }
}
registerProcessor('pcm-capture', CaptureProcessor);
