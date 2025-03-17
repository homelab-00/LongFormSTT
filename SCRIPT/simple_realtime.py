import sys
import io
import time
from RealtimeSTT import AudioToTextRecorder

# Fix console encoding for Windows
if sys.platform == "win32":
    # Force UTF-8 encoding for console output
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def print_transcription(text):
    """Print real-time transcription to the terminal with proper encoding."""
    # Using a timestamp to show updates more clearly
    timestamp = time.strftime('%H:%M:%S')
    print(f"[{timestamp}] {text}")

if __name__ == "__main__":
    # Initialize the recorder with real-time transcription enabled
    # and Greek language support
    recorder = AudioToTextRecorder(
        # Set model to the one you prefer
        model="large-v3",
        # Set Greek language
        language="el",
        # Enable real-time transcription
        enable_realtime_transcription=True,
        # Set callback for real-time updates
        on_realtime_transcription_update=print_transcription,
        # Disable logging to file to avoid encoding issues
        no_log_file=True,
        # Use a smaller beam size for faster real-time responses
        beam_size_realtime=3
    )

    print("Real-time Greek transcription started. Speak into your microphone...")
    print("Press Ctrl+C to exit.")

    try:
        # Keep the program running until Ctrl+C is pressed
        while True:
            # Process audio in a continuous loop
            recorder.text()
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        # Clean up resources
        recorder.shutdown()