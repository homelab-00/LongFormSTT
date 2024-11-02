import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# Set device and data type
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Specify the model ID
model_id = "openai/whisper-large-v3"

# Load the model with the appropriate settings for accuracy
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True
)
model.to(device)

# Load the processor
processor = AutoProcessor.from_pretrained(model_id)

# Set up the pipeline for automatic speech recognition
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

# Transcribe the audio file with the specified parameters for accuracy
result = pipe(
    "recording.wav",
    generate_kwargs={
        "language": "greek",
        "task": "transcribe",
        "num_beams": 5,
        "temperature": 0.0
    }
)

# Print the transcription result
print(result["text"])
