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
import webrtcvad  # New import for VAD

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
USE_SYSTEM_AUDIO = False  # Set to True to capture system audio via stereomix
INPUT_DEVICE_INDEX = 2    # Device index for stereomix (usually 2 on Windows 10)

MIN_CHUNK_LENGTH_SEC = 60   # Minimum length before we allow a chunk split
# Increase the silence frames a bit to have larger padding on chunk boundaries
SILENCE_FRAMES_REQUIRED = 12  # Was 8, now 12 to retain extra silence padding

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

# Keep track of partial transcriptions
partial_transcripts = []
transcription_threads = []

current_chunk_index = 1
chunk_start_time = 0.0  # We will set this when we open each new chunk

# PyAudio parameters
audio = pyaudio.PyAudio()
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024  # We'll read ~64ms chunks
chunks_per_second = RATE // CHUNK  # ~16 reads per second

# This buffer will hold *all* audio data (including silence) for the current chunk
buffer = []

def cleanup_before_recording():
    """
    Whenever F3 is pressed, we delete all temp_audio_file*.wav (including the plain one).
    This ensures a clean start each session.
    """
    temp_files = glob.glob(os.path.join(script_dir, "temp_audio_file*.wav"))
    for f in temp_files:
        try:
            os.remove(f)
            console.print(f"[yellow]Deleted file: {os.path.basename(f)}[/yellow]")
        except Exception as e:
            console.print(f"[red]Failed to delete {os.path.basename(f)}: {e}[/red]")

def toggle_paste():
    global paste_enabled
    paste_enabled = not paste_enabled
    status = "enabled" if paste_enabled else "disabled"
    console.print(f"[italic green]Paste is now {status}.[/italic]")

def start_recording():
    """
    Press F3 to start a brand new recording session.
      1) Cleanup leftover temp_audio_file*.wav
      2) Reset chunk indexing to 1
      3) Open temp_audio_file1.wav for writing
      4) Launch record_audio() in a thread
    """
    global recording, recording_thread
    global partial_transcripts, transcription_threads
    global buffer, current_chunk_index
    global chunk_start_time
    global active_filename, active_wave_file, stream

    if recording:
        console.print("[bold yellow]Already recording![/bold yellow]")
        return

    console.print("[bold green]Starting a new recording session[/bold green]")
    cleanup_before_recording()

    # Reset state
    partial_transcripts.clear()
    transcription_threads.clear()
    buffer = []
    current_chunk_index = 1

    # Open mic stream
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
    except Exception as e:
        console.print(f"[bold red]Failed to open audio stream: {e}[/bold red]")
        return

    # Open first chunk file
    active_filename = os.path.join(
        script_dir, f"temp_audio_file{current_chunk_index}.wav"
    )
    try:
        active_wave_file = wave.open(active_filename, 'wb')
        active_wave_file.setnchannels(CHANNELS)
        active_wave_file.setsampwidth(audio.get_sample_size(FORMAT))
        active_wave_file.setframerate(RATE)
    except Exception as e:
        console.print(f"[bold red]Failed to open wave file: {e}[/bold red]")
        return

    # We mark the time when we started this chunk
    chunk_start_time = time.time()

    recording = True
    recording_thread = threading.Thread(target=record_audio, daemon=True)
    recording_thread.start()

def record_audio():
    """
    Main recording loop with VAD:
      - Reads from mic
      - Writes *all* audio (including silence) to the current chunk file
      - Uses webrtcvad to detect speech frames only to see if we should split after
        MIN_CHUNK_LENGTH_SEC has passed.
    """
    global recording, active_wave_file, active_filename
    global partial_transcripts, transcription_threads
    global buffer, current_chunk_index
    global chunk_start_time
    global stream

    vad = webrtcvad.Vad(1)  # Low aggressiveness (0-3)
    leftover = b""
    chunk_count = 0
    silence_frame_count = 0

    try:
        while recording:
            data = stream.read(CHUNK, exception_on_overflow=False)
            # We always store the raw data (including silence)
            buffer.append(data)

            combined = leftover + data
            offset = 0
            speech_detected_in_this_read = False

            # Break the read into 20ms frames for VAD
            while len(combined) - offset >= 640:  # 20ms @ 16kHz, 16-bit mono
                frame = combined[offset:offset+640]
                offset += 640
                if vad.is_speech(frame, RATE):
                    speech_detected_in_this_read = True

            leftover = combined[offset:]

            # Flush buffer to file ~1x/second
            chunk_count += 1
            if chunk_count >= chunks_per_second:
                if buffer:
                    # Write all accumulated audio to the wave file
                    active_wave_file.writeframes(b"".join(buffer))
                    buffer = []
                chunk_count = 0

            # ---------------------------------------------
            # CHUNK SPLIT LOGIC
            # ---------------------------------------------
            now = time.time()
            elapsed_in_chunk = now - chunk_start_time

            # We only consider splitting if we are beyond MIN_CHUNK_LENGTH_SEC
            if elapsed_in_chunk >= MIN_CHUNK_LENGTH_SEC:
                if not speech_detected_in_this_read:
                    # Potential silent frame
                    silence_frame_count += 1
                else:
                    silence_frame_count = 0

                # If we've reached enough consecutive frames with no speech => finalize chunk
                if silence_frame_count >= SILENCE_FRAMES_REQUIRED:
                    split_current_chunk()
                    chunk_start_time = time.time()  # reset for new chunk
                    current_chunk_index += 1
                    silence_frame_count = 0

        # End of while => user pressed F4
        # Write leftover buffer
        if buffer:
            active_wave_file.writeframes(b"".join(buffer))
            buffer = []

    except Exception as e:
        console.print(f"[bold red]Recording error: {e}[/bold red]")
    finally:
        if active_wave_file:
            active_wave_file.close()
        if stream:
            stream.stop_stream()
            stream.close()

        recording = False
        console.print("[green]Recording stopped.[/green]")

def split_current_chunk():
    """
    Closes the current chunk file, transcribes it in a background thread,
    and opens a new file for the next chunk.
    """
    global active_wave_file, active_filename
    global transcription_threads, current_chunk_index

    # 1) Close the existing file
    if active_wave_file:
        active_wave_file.close()

    # 2) Kick off transcription
    chunk_path = active_filename
    t = threading.Thread(target=partial_transcribe, args=(chunk_path,))
    t.start()
    transcription_threads.append(t)

    # 3) Open a new chunk file
    new_filename = os.path.join(
        script_dir, f"temp_audio_file{current_chunk_index + 1}.wav"
    )
    active_filename = new_filename

    try:
        wf = wave.open(new_filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        console.print(f"[green]Opened new chunk file: {os.path.basename(new_filename)}[/green]")
    except Exception as e:
        console.print(f"[bold red]Failed to open new chunk file {new_filename}: {e}[/bold red]")
        return

    active_wave_file = wf

def partial_transcribe(chunk_path):
    """
    Transcribe the given chunk, remove hallucinations, and store partial text.
    Print partial (but do NOT paste).
    """
    global partial_transcripts
    try:
        segments, info = model.transcribe(
            chunk_path,
            language=language,
            task=task,
            beam_size=10,
            best_of=10,
            temperature=0.0
        )
        text = "".join(s.text for s in segments)

        # Remove hallucinations
        for pattern in HALLUCINATIONS_REGEX:
            text = pattern.sub("", text)

        console.print(f"[cyan]Partial transcription of {os.path.basename(chunk_path)}[/cyan]")
        console.print(f"[bold magenta]{text}[/bold magenta]\n")
        partial_transcripts.append(text)

    except Exception as e:
        console.print(f"[bold red]Partial transcription failed for {chunk_path}: {e}[/bold red]")

def stop_recording_and_transcribe():
    """
    Press F4 to stop recording, finalize last chunk, wait for partials, combine them,
    print & paste the final text.
    """
    global recording, recording_thread, active_wave_file, active_filename
    global partial_transcripts, transcription_threads

    if not recording:
        console.print("[italic bold yellow]Recording[/italic bold yellow] [italic]not in progress[/italic]")
        return

    console.print("[bold blue]Stopping recording and transcribing...[/bold blue]")
    recording = False

    if recording_thread:
        recording_thread.join()

    # Transcribe the last chunk if it has data
    if active_filename and os.path.exists(active_filename):
        if os.path.getsize(active_filename) > 44:  # bigger than just WAV header
            final_chunk = active_filename
            t = threading.Thread(target=partial_transcribe, args=(final_chunk,))
            t.start()
            transcription_threads.append(t)

    console.print("[blue]Waiting for partial transcriptions...[/blue]")
    for t in transcription_threads:
        t.join()

    # Combine all partial transcripts
    full_text = "".join(partial_transcripts)

    panel = Panel(
        f"[bold magenta]Final Combined Transcription:[/bold magenta] {full_text}",
        title="Transcription",
        border_style="yellow"
    )
    console.print(panel)

    if paste_enabled:
        pyperclip.copy(full_text)
        keyboard.send('ctrl+v')

    console.print("[italic green]Done.[/italic green]")

# --------------------------------------------------------------------------------------
# Hotkeys
# --------------------------------------------------------------------------------------
def setup_hotkeys():
    keyboard.add_hotkey('F2', toggle_paste, suppress=True)
    keyboard.add_hotkey('F3', start_recording, suppress=True)
    keyboard.add_hotkey('F4', stop_recording_and_transcribe, suppress=True)

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

    if paste_enabled:
        console.print("[italic green]Typing is enabled on start.[/italic green]")

if __name__ == "__main__":
    startup()
    keyboard.wait()
