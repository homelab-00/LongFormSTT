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

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
THRESHOLD = 500              # Amplitude threshold for silence vs. voice
SILENCE_LIMIT_SEC = 1.5      # Keep up to 1.5 seconds of silence
CHUNK_SPLIT_INTERVAL = 60    # 1 minute in seconds

# Hallucination filtering with regex (optional)
HALLUCINATIONS_REGEX = [
    re.compile(r"\bΥπότιτλοι\s+AUTHORWAVE\b", re.IGNORECASE),
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
recording = False

recording_thread = None
transcription_threads = []
partial_transcripts = []

# PyAudio parameters
audio = pyaudio.PyAudio()
stream = None

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
chunks_per_second = RATE // CHUNK

# Filenames & chunk counters
current_chunk_index = 1       # We'll increment each time we finish a 1-min chunk
active_wave_file = None       # The wave.Wave_write object for the currently recording file
active_filename = None        # e.g. "temp_audio_file.wav"
buffer = []                   # Audio buffer for partial chunk writes

# Timing
record_start_time = 0
next_split_time = 0

# --------------------------------------------------------------------------------------
# Utility Functions
# --------------------------------------------------------------------------------------
def clean_up_temp_files():
    """
    Deletes any existing temp_audio_file*.wav files at the start to prevent naming conflicts.
    """
    temp_files = glob.glob(os.path.join(script_dir, "temp_audio_file*.wav"))
    for file in temp_files:
        try:
            os.remove(file)
            console.print(f"[green]Deleted old temp file: {os.path.basename(file)}[/green]")
        except Exception as e:
            console.print(f"[red]Failed to delete old temp file {os.path.basename(file)}: {e}[/red]")

# --------------------------------------------------------------------------------------
# Hotkey Handlers
# --------------------------------------------------------------------------------------
def toggle_paste():
    global paste_enabled
    paste_enabled = not paste_enabled
    status = "enabled" if paste_enabled else "disabled"
    console.print(f"[italic green]Paste is now {status}.[/italic]")

def start_recording():
    """
    Press F3 to start recording. Opens 'temp_audio_file.wav',
    resets timing and chunk counters, etc.
    """
    global recording, record_start_time, next_split_time
    global current_chunk_index, partial_transcripts, transcription_threads
    global active_wave_file, active_filename, stream, buffer

    if recording:
        console.print("[bold yellow]Recording already in progress.[/bold yellow]")
        return

    console.print("[bold green]Starting recording[/bold green]")
    recording = True
    partial_transcripts.clear()
    transcription_threads.clear()
    buffer = []

    # Reset timing
    record_start_time = time.time()
    next_split_time = record_start_time + CHUNK_SPLIT_INTERVAL
    current_chunk_index = 1

    # Prepare first filename: 'temp_audio_file.wav'
    active_filename = os.path.join(script_dir, "temp_audio_file.wav")

    # Open PyAudio input
    try:
        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
    except Exception as e:
        console.print(f"[bold red]Failed to open audio stream: {e}[/bold red]")
        recording = False
        return

    # Open wave file for writing (overwrite if exists)
    try:
        active_wave_file = wave.open(active_filename, 'wb')
        active_wave_file.setnchannels(CHANNELS)
        active_wave_file.setsampwidth(audio.get_sample_size(FORMAT))
        active_wave_file.setframerate(RATE)
    except Exception as e:
        console.print(f"[bold red]Failed to open wave file: {e}[/bold red]")
        recording = False
        return

    # Launch recording thread
    global recording_thread
    recording_thread = threading.Thread(target=record_audio, daemon=True)
    recording_thread.start()

def record_audio():
    """
    Main recording loop:
      - Reads chunks from the mic.
      - Trims silence.
      - Splits into new files every 1 minute of *wall-clock* time.
    """
    global recording, active_wave_file, active_filename, buffer
    global record_start_time, next_split_time, current_chunk_index
    chunk_count = 0
    silence_duration = 0.0

    try:
        while recording:
            data = stream.read(CHUNK, exception_on_overflow=False)
            peak = audioop.max(data, 2)
            chunk_time = float(CHUNK) / RATE

            # Silence detection
            if peak < THRESHOLD:
                silence_duration += chunk_time
                if silence_duration <= SILENCE_LIMIT_SEC:
                    buffer.append(data)
                    chunk_count += 1
            else:
                silence_duration = 0.0
                buffer.append(data)
                chunk_count += 1

            # Write to wave once per second of audio
            if chunk_count >= chunks_per_second:
                active_wave_file.writeframes(b''.join(buffer))
                buffer = []
                chunk_count = 0

            # Check if it's time to split
            now = time.time()
            if now >= next_split_time:
                # We reached 1 minute of real time
                split_current_chunk()
                current_chunk_index += 1
                next_split_time += CHUNK_SPLIT_INTERVAL

        # Loop ends when user presses F4
        if buffer:
            active_wave_file.writeframes(b''.join(buffer))
            buffer = []

    except Exception as e:
        console.print(f"[bold red]Recording error: {e}[/bold red]")
    finally:
        # Close wave file, close stream
        if active_wave_file:
            active_wave_file.close()
        if stream:
            stream.stop_stream()
            stream.close()
        console.print("[green]Recording finished.[/green]")
        recording = False

def split_current_chunk():
    """
    Closes the current wave file (e.g. temp_audio_file.wav),
    spawns a thread to transcribe it,
    then opens temp_audio_file2.wav, temp_audio_file3.wav, etc. for new recording.
    """
    global active_wave_file, active_filename, current_chunk_index

    # 1) Close the active wave file
    active_wave_file.close()

    # 2) Define the new chunk name: 'temp_audio_file1.wav', 'temp_audio_file2.wav', etc.
    chunk_final_name = os.path.join(
        script_dir,
        f"temp_audio_file{current_chunk_index}.wav"
    )

    # 3) If the chunk_final_name already exists, delete it to prevent WinError 183
    if os.path.exists(chunk_final_name):
        try:
            os.remove(chunk_final_name)
            console.print(f"[yellow]Deleted existing chunk file: {os.path.basename(chunk_final_name)}[/yellow]")
        except Exception as e:
            console.print(f"[bold red]Failed to delete existing chunk file {chunk_final_name}: {e}[/bold red]")
            return  # Skip transcription to avoid conflicts

    # 4) Rename the active file to the final chunk name
    try:
        os.rename(active_filename, chunk_final_name)
        console.print(f"[green]Renamed {os.path.basename(active_filename)} to {os.path.basename(chunk_final_name)}[/green]")
    except Exception as e:
        console.print(f"[bold red]Failed to rename {active_filename} to {chunk_final_name}: {e}[/bold red]")
        return  # Skip transcription to avoid conflicts

    # 5) Transcribe the chunk in a separate thread
    t = threading.Thread(target=partial_transcribe, args=(chunk_final_name,))
    t.start()
    transcription_threads.append(t)

    # 6) Open a new wave file for the next chunk
    new_filename = os.path.join(script_dir, f"temp_audio_file{current_chunk_index + 1}.wav")
    active_filename = new_filename

    try:
        active_wave_file = wave.open(new_filename, 'wb')
        active_wave_file.setnchannels(CHANNELS)
        active_wave_file.setsampwidth(audio.get_sample_size(FORMAT))
        active_wave_file.setframerate(RATE)
        console.print(f"[green]Opened new chunk file: {os.path.basename(new_filename)}[/green]")
    except Exception as e:
        console.print(f"[bold red]Failed to open new chunk file {new_filename}: {e}[/bold red]")
        recording = False

def partial_transcribe(chunk_path):
    """
    Transcribe chunk_path, remove known hallucinations, store in partial_transcripts.
    Prints partial result (but does NOT paste).
    """
    global partial_transcripts
    try:
        segments, info = model.transcribe(chunk_path, language=language, task=task)
        chunk_text = "".join(seg.text for seg in segments)

        # Remove hallucinations
        for pattern in HALLUCINATIONS_REGEX:
            chunk_text = pattern.sub("", chunk_text)

        console.print(f"[cyan]Partial transcription of {os.path.basename(chunk_path)}[/cyan]")
        console.print(f"[bold magenta]{chunk_text}[/bold magenta]\n")

        partial_transcripts.append(chunk_text)

    except Exception as e:
        console.print(f"[bold red]Partial transcription failed for {chunk_path}: {e}[/bold red]")

def stop_recording_and_transcribe():
    """
    Press F4 to:
      1) Stop the record_audio loop.
      2) If there's a partial chunk not yet split, rename & transcribe it.
      3) Wait for all partial transcription threads to finish.
      4) Combine them all and print + paste.
    """
    global recording, recording_thread, transcription_threads
    global active_filename, active_wave_file, current_chunk_index, partial_transcripts

    if not recording:
        console.print("[bold yellow]No recording in progress.[/bold yellow]")
        return

    console.print("[bold blue]Stopping recording...[/bold blue]")
    recording = False

    # Wait for record_audio() to exit
    if recording_thread:
        recording_thread.join()

    # Handle the final chunk if it's not yet split
    if os.path.exists(active_filename):
        # wave_file is closed in finally: block. So let's rename the final chunk
        # only if it has more than just a header.
        if os.path.getsize(active_filename) > 44:  # ~44 bytes is the header of an empty WAV
            final_chunk_name = os.path.join(
                script_dir,
                f"temp_audio_file{current_chunk_index}.wav"
            )
            # Delete if exists to prevent rename error
            if os.path.exists(final_chunk_name):
                try:
                    os.remove(final_chunk_name)
                    console.print(f"[yellow]Deleted existing final chunk file: {os.path.basename(final_chunk_name)}[/yellow]")
                except Exception as e:
                    console.print(f"[bold red]Failed to delete existing final chunk file {final_chunk_name}: {e}[/bold red]")
                    return

            try:
                os.rename(active_filename, final_chunk_name)
                console.print(f"[green]Renamed final chunk to {os.path.basename(final_chunk_name)}[/green]")
                # Transcribe final chunk
                t = threading.Thread(target=partial_transcribe, args=(final_chunk_name,))
                t.start()
                transcription_threads.append(t)
            except Exception as e:
                console.print(f"[bold red]Could not finalize last chunk: {e}[/bold red]")

    console.print("[blue]Waiting for all partial transcriptions to finish...[/blue]")
    for t in transcription_threads:
        t.join()

    console.print("[green]All chunks transcribed.[/green]")

    # Combine partial transcripts
    full_text = "".join(partial_transcripts)

    # Print final text & paste
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
# Hotkeys Registration
# --------------------------------------------------------------------------------------
def setup_hotkeys():
    keyboard.add_hotkey('F2', toggle_paste, suppress=True)
    keyboard.add_hotkey('F3', start_recording, suppress=True)
    keyboard.add_hotkey('F4', stop_recording_and_transcribe, suppress=True)

# --------------------------------------------------------------------------------------
# Startup Actions
# --------------------------------------------------------------------------------------
def startup():
    # Clean up old temp_audio_file*.wav files
    clean_up_temp_files()

    # Set up hotkeys
    setup_hotkeys()

    # Display information panel
    panel_content = (
        f"[bold yellow]Model[/bold yellow]: {model_id}\n"
        "[bold yellow]Hotkeys[/bold yellow]: "
        "[bold green]F2[/bold green] - Toggle typing | "
        "[bold green]F3[/bold green] - Start recording | "
        "[bold green]F4[/bold green] - Stop & Transcribe\n"
        "[bold yellow]Split[/bold yellow]: Every 1 minute => new temp_audio_fileN.wav\n"
        "[bold yellow]Transcripts[/bold yellow]: Partial shown (no paste), final combined with paste"
    )
    panel = Panel(panel_content, title="Information", border_style="green")
    console.print(panel)

    if paste_enabled:
        console.print("[italic green]Typing is enabled on start.[/italic green]")

# --------------------------------------------------------------------------------------
# Main Execution
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    startup()
    # Keep script alive
    keyboard.wait()
