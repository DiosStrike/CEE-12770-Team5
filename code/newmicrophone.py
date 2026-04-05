import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

sample_rate = 16000
channels = 1
device_index = 1
blocksize = 1024

audio_buffer = []
latest_chunk = np.zeros(blocksize, dtype=np.float32)

def callback(indata, frames, time, status):
    global latest_chunk
    if status:
        print(status)
    chunk = indata[:, 0].copy()
    audio_buffer.append(chunk.copy())
    latest_chunk = chunk

plt.ion()
fig, ax = plt.subplots()
x = np.arange(blocksize)
line, = ax.plot(x, latest_chunk)
ax.set_xlim(0, blocksize)
ax.set_ylim(-1, 1)
ax.set_title("Real-time Audio Waveform")
ax.set_xlabel("Samples")
ax.set_ylabel("Amplitude")

print("Recording... Press Ctrl+C to stop")

try:
    with sd.InputStream(
        samplerate=sample_rate,
        channels=channels,
        device=device_index,
        dtype='float32',
        callback=callback,
        blocksize=blocksize
    ):
        while True:
            line.set_ydata(latest_chunk)
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.01)

except KeyboardInterrupt:
    print("\nRecording stopped.")

full_audio = np.concatenate(audio_buffer)
write("realtime_recording.wav", sample_rate, (full_audio * 32767).astype(np.int16))
print("Saved to realtime_recording.wav")

plt.ioff()

time_axis = np.arange(len(full_audio)) / sample_rate
plt.figure(figsize=(12, 4))
plt.plot(time_axis, full_audio)
plt.title("Full Audio Waveform")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 4))
plt.specgram(full_audio, Fs=sample_rate)
plt.title("Spectrogram")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.colorbar(label="Intensity")
plt.tight_layout()
plt.show()