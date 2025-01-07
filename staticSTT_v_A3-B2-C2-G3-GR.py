import torch
import torchaudio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

# Set device and data type
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Specify the model ID
model_id = "openai/whisper-large-v3"

# Load the model with the appropriate settings for accuracy
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True
).to(device)

# Load the processor
processor = AutoProcessor.from_pretrained(model_id)

# Set the language and task in the model's generation config
language = "greek"
task = "transcribe"
forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
model.config.forced_decoder_ids = forced_decoder_ids

# Load and process the audio file
audio_input, _ = torchaudio.load("recording.wav")
inputs = processor(audio_input.squeeze(), sampling_rate=16000, return_tensors="pt")
inputs = inputs.to(device)

# Adjust max_new_tokens to ensure total length <= 448
max_new_tokens = 445  # 448 (max_target_positions) - 3 (initial decoder_input_ids length)

# Set generation parameters
generate_kwargs = {
    "num_beams": 5,                  # Use beam search for better accuracy
    "temperature": 0.0,              # Use temperature fallback
    "max_new_tokens": max_new_tokens,  # Adjusted max_new_tokens
    "condition_on_prev_tokens": False, # Recommended for long-form transcription
    "compression_ratio_threshold": 2.4, # Adjusted threshold
    "logprob_threshold": -1.0,        # Adjusted logprob threshold
    "no_speech_threshold": 0.6        # Adjusted no_speech threshold
}

# Generate transcription
with torch.no_grad():
    generated_ids = model.generate(
        inputs["input_features"],
        **generate_kwargs
    )
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

# Print the transcription result
print(transcription)
