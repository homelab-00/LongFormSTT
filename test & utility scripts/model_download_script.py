from huggingface_hub import snapshot_download

# Define the model ID
model_id = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"

# Download the model
model_path = snapshot_download(repo_id=model_id)

print(f"Model downloaded to: {model_path}")
