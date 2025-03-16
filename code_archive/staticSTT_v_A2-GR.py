# Import necessary libraries
from transformers import pipeline
import torch

# Load the WhisperLargeV3 model with Greek language setting
model_name = "openai/whisper-large-v3"
transcriber = pipeline("automatic-speech-recognition", model=model_name, device=0 if torch.cuda.is_available() else -1)

# Specify the path to your audio file
audio_path = "recording.wav"

# Transcribe the audio file
print("Transcription in progress...")
result = transcriber(audio_path, chunk_length_s=60, language="el")

# Save the transcription to a text file
output_path = "transcription_output.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(result["text"])

print(f"Transcription completed. The output is saved at {output_path}")
