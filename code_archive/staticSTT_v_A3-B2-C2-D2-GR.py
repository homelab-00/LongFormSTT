import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import librosa

# Set device and data type
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Specify the model ID
model_id = "openai/whisper-large-v3"

# Load the model with appropriate settings for accuracy
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    use_safetensors=True
).to(device)

# Load the processor
processor = AutoProcessor.from_pretrained(model_id)

# Set the language and task in the model's generation config
language = "greek"
task = "transcribe"
forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
model.config.forced_decoder_ids = forced_decoder_ids

# Load and resample the audio file using librosa
audio_input, _ = librosa.load("recording.wav", sr=16000)

# Process the audio to get input_features
input_features = processor(
    audio_input,
    sampling_rate=16000,
    return_tensors="pt"
).input_features

# Move input_features to the device and set the correct dtype
input_features = input_features.to(device, dtype=torch_dtype)

# Adjust max_new_tokens to ensure total length <= 448
max_new_tokens = 445  # 448 (max_target_positions) - initial decoder input length (3)

# Generate tokens using the model's generate method with input_features
generated_tokens = model.generate(
    inputs=input_features,  # Use 'inputs' instead of 'input_features' as per the latest API
    max_new_tokens=max_new_tokens,
    num_beams=5,
    temperature=0.0,
    condition_on_prev_tokens=False,
    compression_ratio_threshold=2.4,
    logprob_threshold=-1.0,
    no_speech_threshold=0.6,
)

# Decode the tokens to get the transcription
transcription = processor.batch_decode(
    generated_tokens, skip_special_tokens=True
)[0]

# Print the transcription result
print(transcription)
