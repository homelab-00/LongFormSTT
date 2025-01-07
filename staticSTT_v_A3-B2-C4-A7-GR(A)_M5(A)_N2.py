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
import re  # Added for regex

# Silence detection parameters (adjust as needed)
THRESHOLD = 500         # Amplitude threshold for silence vs. voice
SILENCE_LIMIT_SEC = 1.5 # Keep up to 1.5s of silence before trimming

# Define hallucinations with regex patterns
HALLUCINATIONS_REGEX = [
    re.compile(r"\bΥπότιτλοι\s+AUTHORWAVE\b", re.IGNORECASE),
    # Add more patterns here if needed
]

# Initialize Rich console
console = Console()

# Get the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Set device and data type
device = "cuda" if torch.cuda.is_available() else "cpu"
# 'faster-whisper' handles dtype internally

# Specify the model ID
model_id = "Systran/faster-whisper-large-v3"  # Ensure this model is in CTranslate2 format

# Load the faster-whisper model
model = WhisperModel(model_id, device=device, compute_type="float16" if device == "cuda" else "float32")

# Set the language and task
language = "el"  # ISO code for Greek
task = "transcribe"

# Transcription paste toggle
paste_enabled = True  # Start with paste enabled
temp_file = None
recording = False
transcribing = False
lock = threading.Lock()
recording_thread = None  # Keep track of the recording thread
transcription_thread = None  # Keep track of the transcription thread
wave_file = None  # Add a new global variable to handle the wave file

# PyAudio parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
audio = pyaudio.PyAudio()
stream = None

buffer = []  # Add a buffer to accumulate audio data
chunks_per_second = RATE // CHUNK  # Calculate chunks that make up one second

def toggle_paste():
    global paste_enabled
    paste_enabled = not paste_enabled
    status = "enabled" if paste_enabled else "disabled"
    console.print(f"[italic green]Paste[/italic green] [italic]{status}.[/italic]")

def start_recording():
    global recording, temp_file, stream, wave_file, recording_thread
    if recording:
        console.print("[italic bold yellow]Recording[/italic bold yellow] [italic]in progress.[/italic]")
        return
    recording = True
    console.print("[italic bold green]Starting[/italic bold green] [italic]recording[/italic]")
    temp_file = os.path.join(script_dir, "temp_audio_file.wav")
    try:
        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        wave_file = wave.open(temp_file, 'wb')  # Open wave file for writing
        wave_file.setnchannels(CHANNELS)
        wave_file.setsampwidth(audio.get_sample_size(FORMAT))
        wave_file.setframerate(RATE)
    except Exception as e:
        console.print(f"[bold red]Failed to start recording: {e}[/bold red]")
        recording = False
        return
    recording_thread = threading.Thread(target=record_audio, args=(), daemon=True)
    recording_thread.start()

def record_audio():
    """
    Continuously read audio chunks from the mic.
    Write them to 'temp_audio_file.wav',
    but trim silence (beyond ~1.5s) to reduce file size and unneeded silent segments.
    """
    global recording, wave_file, buffer
    chunk_count = 0  # For writing to the wave file once per second

    # --- Silence detection additions below ---
    silence_duration = 0.0
    # We removed "Silence detected" prints to avoid spam
    # --- End of silence detection additions ---

    try:
        while recording:
            data = stream.read(CHUNK, exception_on_overflow=False)

            # --- Silence detection logic ---
            peak = audioop.max(data, 2)
            chunk_time = float(CHUNK) / RATE  

            if peak < THRESHOLD:
                # It's silence
                silence_duration += chunk_time
                if silence_duration <= SILENCE_LIMIT_SEC:
                    buffer.append(data)
                    chunk_count += 1
            else:
                # We detected voice, reset silence
                silence_duration = 0.0
                buffer.append(data)
                chunk_count += 1
            # --- End of silence detection logic ---

            # Write once per second of accumulated chunks
            if chunk_count >= chunks_per_second:
                wave_file.writeframes(b''.join(buffer))
                buffer = []
                chunk_count = 0
    except Exception as e:
        console.print(f"[bold red]Recording error: {e}[/bold red]")
    finally:
        # Write any leftover data
        if buffer:
            wave_file.writeframes(b''.join(buffer))
            buffer = []
        if stream:
            stream.stop_stream()
            stream.close()
        if wave_file:
            wave_file.close()
        console.print("[italic green]Recording[/italic green] [italic]saved to temp audio file[/italic]")
        recording = False

def stop_recording_and_transcribe():
    global recording, transcribing, temp_file, recording_thread, transcription_thread
    if not recording:
        console.print("[italic bold yellow]Recording[/italic bold yellow] [italic]not in progress[/italic]")
        return
    recording = False
    console.print("[italic bold blue]Stopping[/italic bold blue] [italic]recording and transcribing...[/italic]")

    if recording_thread is not None:
        recording_thread.join()  # Wait for the recording thread to finish

    transcribing = True
    transcription_thread = threading.Thread(target=transcribe, args=(temp_file,), daemon=True)
    transcription_thread.start()

def transcribe(filename):
    global transcribing, temp_file
    try:
        segments, info = model.transcribe(filename, language=language, task=task)
        text = ''.join([segment.text for segment in segments])

        # --- Remove known hallucinations using regex ---
        for pattern in HALLUCINATIONS_REGEX:
            text = pattern.sub("", text)
        # ------------------------------------

        transcription_panel = Panel(
            f"[bold magenta]Transcription:[/bold magenta] {text}",
            title="Transcription",
            border_style="yellow"
        )
        console.print(transcription_panel)
        if paste_enabled:
            pyperclip.copy(text)
            keyboard.send('ctrl+v')  # Paste the transcribed text
    except Exception as e:
        console.print(f"[bold red]Transcription failed: {e}[/bold red]")
    finally:
        transcribing = False
        temp_file = None

# Set up global hotkeys
keyboard.add_hotkey('F2', toggle_paste, suppress=True)
keyboard.add_hotkey('F3', start_recording, suppress=True)
keyboard.add_hotkey('F4', stop_recording_and_transcribe, suppress=True)

panel_content = (
    f"[bold yellow]Model[/bold yellow]  : {model_id}\n"
    "[bold yellow]Hotkeys[/bold yellow]: [bold green]F2[/bold green] - Toggle typing "
    "| [bold green]F3[/bold green] - Start recording "
    "| [bold green]F4[/bold green] - Stop & Transcribe\n"
    "[bold yellow]Silence detection[/bold yellow]: RMS/peak value-based trimming"
)
panel = Panel(panel_content, title="Information", border_style="green")
console.print(panel)

# If paste_enabled is True at start, ensure it's enabled
if paste_enabled:
    console.print("[italic green]Typing[/italic green] [italic]is enabled on start.[/italic]")

# Keep the script running
keyboard.wait()
