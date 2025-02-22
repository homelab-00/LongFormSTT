import pyaudio
import wave

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1  # Mono for Whisper
RATE = 16000  # Whisper's preferred sample rate

p = pyaudio.PyAudio()
stream = p.open(
    input_device_index=3,  # Replace with your index
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK
)

print("Recording...")
frames = []
for _ in range(100):  # Record 100 chunks (~10 seconds)
    data = stream.read(CHUNK)
    frames.append(data)

stream.stop_stream()
stream.close()
p.terminate()

# Save test file
with wave.open("test.wav", "wb") as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))

print("Saved to test.wav. Play this file to verify audio capture.")