import time
import torch
import keyboard
import pyaudio
import wave
import os
import threading
import pyperclip
from rich.console import Console
from rich.panel import Panel
from faster_whisper import WhisperModel
import audioop
import re
import glob
import logging

# Set up logging (DEBUG level roughly equals 7/10 detail)
logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
USE_SYSTEM_AUDIO = True  # Set to True to capture system audio via stereomix
INPUT_DEVICE_INDEX = 2    # Device index for stereomix (usually 2 on Windows 10)

THRESHOLD = 500              # Amplitude threshold for silence vs. voice
SILENCE_LIMIT_SEC = 1.5      # Keep up to 1.5 seconds of silence
CHUNK_SPLIT_INTERVAL = 60    # How many seconds is each chunk, default 1 minute

# Hallucination filtering with regex (optional)
HALLUCINATIONS_REGEX = [
    re.compile(r"\bΥπότιτλοι\s+AUTHORWAVE\b[^\w]*", re.IGNORECASE),
    re.compile(r"\bΣας\s+ευχαριστώ\b[^\w]*", re.IGNORECASE),
    # Add more patterns if needed
]

# --------------------------------------------------------------------------------------
# Globals
# --------------------------------------------------------------------------------------
console = Console()
script_dir = os.path.dirname(os.path.abspath(__file__))

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "Systran/faster-whisper-large-v3"
model = WhisperModel(model_id, device=device, compute_type="float16" if device == "cuda" else "float32")

language = "el"
task = "transcribe"

paste_enabled = True

# Recording state
recording = False
recording_thread = None
stream = None
active_wave_file = None
active_filename = None

# Tracks partial transcriptions and transcription threads
partial_transcripts = []
transcription_threads = []

# For chunk logic
current_chunk_index = 1
record_start_time = 0
next_split_time = 0
chunk_split_requested = False  # Flag to indicate we want to split on next silence

# PyAudio parameters
audio = pyaudio.PyAudio()
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
chunks_per_second = RATE // CHUNK

# Audio buffer used in the record loop (renamed for clarity)
frame_buffer = []

# --------------------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------------------
def open_wave_file(filename):
    """
    Helper function to open a wave file for writing with the configured parameters.
    """
    try:
        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        logger.debug(f"Opened wave file: {filename}")
        return wf
    except Exception as e:
        console.print(f"[bold red]Failed to open wave file {filename}: {e}[/bold red]")
        logger.exception("Error opening wave file")
        return None

def cleanup_before_recording():
    """
    Delete all temp_audio_file*.wav files in the script directory.
    """
    temp_files = glob.glob(os.path.join(script_dir, "temp_audio_file*.wav"))
    for f in temp_files:
        try:
            os.remove(f)
            console.print(f"[yellow]Deleted file: {os.path.basename(f)}[/yellow]")
            logger.debug(f"Deleted temporary file: {f}")
        except Exception as e:
            console.print(f"[red]Failed to delete {os.path.basename(f)}: {e}[/red]")
            logger.exception("Error deleting temporary file")

# --------------------------------------------------------------------------------------
# Hotkey Handlers
# --------------------------------------------------------------------------------------
def toggle_paste():
    global paste_enabled
    paste_enabled = not paste_enabled
    status = "enabled" if paste_enabled else "disabled"
    console.print(f"[italic green]Paste is now {status}.[/italic green]")
    logger.debug(f"Paste toggled: {status}")

def start_recording():
    """
    Start a new recording session (triggered by F3).
    """
    global recording, recording_thread, stream, active_wave_file, active_filename
    global partial_transcripts, transcription_threads, frame_buffer, current_chunk_index
    global record_start_time, next_split_time, chunk_split_requested

    if recording:
        console.print("[bold yellow]Already recording![/bold yellow]")
        logger.debug("Recording already in progress. Ignoring start command.")
        return

    console.print("[bold green]Starting a new recording session[/bold green]")
    logger.debug("Starting new recording session")

    # Cleanup leftover files from previous sessions
    cleanup_before_recording()

    # Reset internal state
    partial_transcripts.clear()
    transcription_threads.clear()
    frame_buffer = []
    current_chunk_index = 1

    # Setup timing
    record_start_time = time.time()
    next_split_time = record_start_time + CHUNK_SPLIT_INTERVAL
    chunk_split_requested = False

    # Open the first chunk file
    first_file = os.path.join(script_dir, f"temp_audio_file{current_chunk_index}.wav")
    active_filename = first_file
    active_wave_file = open_wave_file(first_file)
    if not active_wave_file:
        recording = False
        return

    # Open audio stream
    try:
        stream_params = {
            'format': FORMAT,
            'channels': CHANNELS,
            'rate': RATE,
            'input': True,
            'frames_per_buffer': CHUNK,
            'input_device_index': INPUT_DEVICE_INDEX if USE_SYSTEM_AUDIO else None
        }
        stream = audio.open(**stream_params)
        logger.debug("Audio stream opened successfully")
    except Exception as e:
        console.print(f"[bold red]Failed to open audio stream: {e}[/bold red]")
        logger.exception("Error opening audio stream")
        recording = False
        return

    recording = True
    recording_thread = threading.Thread(target=record_audio, daemon=True)
    recording_thread.start()
    logger.debug("Recording thread started")

def record_audio():
    """
    Main loop for recording audio, handling silence detection and chunk splitting.
    """
    global recording, active_wave_file, active_filename, frame_buffer
    global record_start_time, next_split_time, current_chunk_index, chunk_split_requested
    global recording_thread

    chunk_count = 0
    silence_duration = 0.0

    try:
        while recording:
            data = stream.read(CHUNK, exception_on_overflow=False)
            peak = audioop.max(data, 2)
            chunk_time = float(CHUNK) / RATE
            now = time.time()
            elapsed = now - record_start_time

            # Request a chunk split if the interval is reached
            if not chunk_split_requested and (elapsed >= (next_split_time - record_start_time)):
                console.print(f"[yellow]Reached {int(elapsed)} seconds. Will split on next silence.[/yellow]")
                logger.debug(f"Elapsed time {elapsed:.2f}s reached, marking chunk split request")
                chunk_split_requested = True

            # Silence detection and buffering
            if peak < THRESHOLD:
                silence_duration += chunk_time
                if silence_duration <= SILENCE_LIMIT_SEC:
                    frame_buffer.append(data)
                    chunk_count += 1

                if chunk_split_requested and (silence_duration >= 0.1):
                    console.print("[bold green]Splitting now at silence...[/bold green]")
                    logger.debug("Silence detected and chunk split requested; splitting chunk")
                    split_current_chunk()
                    next_split_time += CHUNK_SPLIT_INTERVAL
                    chunk_split_requested = False
            else:
                silence_duration = 0.0
                frame_buffer.append(data)
                chunk_count += 1

            # Write frames to file once per second
            if chunk_count >= chunks_per_second:
                active_wave_file.writeframes(b''.join(frame_buffer))
                logger.debug(f"Wrote {len(frame_buffer)} frames to {os.path.basename(active_filename)}")
                frame_buffer = []
                chunk_count = 0

        # Write any remaining frames after recording stops
        if frame_buffer:
            active_wave_file.writeframes(b''.join(frame_buffer))
            logger.debug("Wrote remaining frames to file")
            frame_buffer = []
    except Exception as e:
        console.print(f"[bold red]Recording error: {e}[/bold red]")
        logger.exception("Error in recording loop")
    finally:
        if active_wave_file:
            active_wave_file.close()
        if stream:
            stream.stop_stream()
            stream.close()
        recording = False
        logger.debug("Recording stopped and audio stream closed")
        console.print("[green]Recording stopped.[/green]")

def split_current_chunk():
    """
    Split the current audio chunk: close the current file, spawn a partial transcription,
    and open a new chunk file.
    """
    global active_wave_file, active_filename, current_chunk_index, transcription_threads

    if active_wave_file:
        active_wave_file.close()
        logger.debug(f"Closed current chunk file: {os.path.basename(active_filename)}")

    # Spawn partial transcription for the closed chunk
    chunk_path = active_filename
    t = threading.Thread(target=partial_transcribe, args=(chunk_path,))
    t.start()
    transcription_threads.append(t)
    logger.debug(f"Started partial transcription thread for {os.path.basename(chunk_path)}")

    # Open a new chunk file
    current_chunk_index += 1
    new_filename = os.path.join(script_dir, f"temp_audio_file{current_chunk_index}.wav")
    active_filename = new_filename
    active_wave_file = open_wave_file(new_filename)
    if active_wave_file:
        console.print(f"[green]Opened new chunk file: {os.path.basename(new_filename)}[/green]")
        logger.debug(f"New chunk file opened: {new_filename}")

def partial_transcribe(chunk_path):
    """
    Transcribe an audio chunk, filter out hallucinations, and store the partial transcription.
    """
    global partial_transcripts
    try:
        segments, info = model.transcribe(chunk_path, language=language, task=task)
        text = "".join(segment.text for segment in segments)
        for pattern in HALLUCINATIONS_REGEX:
            text = pattern.sub("", text)
        console.print(f"[cyan]Partial transcription of {os.path.basename(chunk_path)}[/cyan]")
        console.print(f"[bold magenta]{text}[/bold magenta]\n")
        logger.debug(f"Transcription for {os.path.basename(chunk_path)}: {text[:50]}...")
        partial_transcripts.append(text)
    except Exception as e:
        console.print(f"[bold red]Partial transcription failed for {chunk_path}: {e}[/bold red]")
        logger.exception("Partial transcription error")

def stop_recording_and_transcribe():
    """
    Stop recording (triggered by F4), finalize the last chunk, wait for all partial transcriptions,
    combine them, then print and paste the final transcription.
    """
    global recording, recording_thread, active_wave_file, active_filename
    global partial_transcripts, transcription_threads

    if not recording:
        console.print("[italic bold yellow]Recording not in progress[/italic bold yellow]")
        logger.debug("Stop recording command issued, but no recording in progress")
        return

    console.print("[bold blue]Stopping recording and transcribing...[/bold blue]")
    logger.debug("Stop recording initiated")
    recording = False

    if recording_thread:
        recording_thread.join()
        logger.debug("Recording thread joined successfully")

    # Process the final chunk if it has valid data
    if active_filename and os.path.exists(active_filename):
        if os.path.getsize(active_filename) > 44:
            final_chunk = active_filename
            t = threading.Thread(target=partial_transcribe, args=(final_chunk,))
            t.start()
            transcription_threads.append(t)
            logger.debug(f"Final chunk {os.path.basename(final_chunk)} queued for transcription")

    console.print("[blue]Waiting for partial transcriptions...[/blue]")
    for t in transcription_threads:
        t.join()
    logger.debug("All partial transcription threads completed")

    full_text = "".join(partial_transcripts)
    panel = Panel(
        f"[bold magenta]Final Combined Transcription:[/bold magenta] {full_text}",
        title="Transcription",
        border_style="yellow"
    )
    console.print(panel)
    logger.debug(f"Final transcription combined length: {len(full_text)} characters")

    if paste_enabled:
        pyperclip.copy(full_text)
        keyboard.send('ctrl+v')
        logger.debug("Final transcription copied to clipboard and pasted")
    console.print("[italic green]Done.[/italic green]")

# --------------------------------------------------------------------------------------
# Hotkeys Setup
# --------------------------------------------------------------------------------------
def setup_hotkeys():
    keyboard.add_hotkey('F2', toggle_paste, suppress=True)
    keyboard.add_hotkey('F3', start_recording, suppress=True)
    keyboard.add_hotkey('F4', stop_recording_and_transcribe, suppress=True)
    logger.debug("Hotkeys F2, F3, F4 registered")

# --------------------------------------------------------------------------------------
# Startup and Main Loop
# --------------------------------------------------------------------------------------
def startup():
    setup_hotkeys()
    panel_content = (
        f"[bold yellow]Model[/bold yellow]: {model_id}\n"
        "[bold yellow]Hotkeys[/bold yellow]: "
        "[bold green]F2[/bold green] - Toggle typing | "
        "[bold green]F3[/bold green] - Start recording | "
        "[bold green]F4[/bold green] - Stop & Transcribe"
    )
    panel = Panel(panel_content, title="Information", border_style="green")
    console.print(panel)
    logger.debug("Startup complete")
    if paste_enabled:
        console.print("[italic green]Typing is enabled on start.[/italic green]")

def main():
    startup()
    try:
        keyboard.wait()
    except KeyboardInterrupt:
        console.print("[red]KeyboardInterrupt received. Exiting...[/red]")
        logger.debug("KeyboardInterrupt received. Exiting program.")
    finally:
        if recording:
            stop_recording_and_transcribe()
        audio.terminate()
        logger.debug("Audio terminated, program exit.")

if __name__ == "__main__":
    main()
