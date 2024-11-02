import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

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

# Set up the pipeline for automatic speech recognition
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
    chunk_length_s=30  # Set chunk length for better memory management during long audio
)

# Adjust parameters to prioritize accuracy for long-form audio
max_new_tokens = 445  # Keeping close to max_target_positions for thorough transcription

# Transcribe the audio file with specified parameters for accuracy
result = pipe(
    "recording.wav",
    return_timestamps=True,  # Required for long-form transcription
    generate_kwargs={
        "num_beams": 5,                   # Use beam search for better accuracy
        "temperature": 0.5,               # Moderate temperature for balanced output
        "max_new_tokens": max_new_tokens,  # Max tokens set for more detailed output
        "condition_on_prev_tokens": True, # Enables better context consideration
        "compression_ratio_threshold": 2.4, # Set based on Whisper's default values
        "logprob_threshold": -1.0,        # Lower logprob threshold for catching nuances
        "no_speech_threshold": 0.5        # Tweak no_speech threshold for better accuracy in pauses
    }
)

# Print the transcription result
print(result["text"])
