import torch
import keyboard
import pyaudio
import wave
import os
import threading
import pyperclip
from rich import print
from rich.console import Console
from faster_whisper import WhisperModel

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

# PyAudio parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
audio = pyaudio.PyAudio()
stream = None
frames = []

def toggle_paste():
    global paste_enabled
    paste_enabled = not paste_enabled
    status = "enabled" if paste_enabled else "disabled"
    console.print(f"[bold yellow]Paste {status}.[/bold yellow]")

def start_recording():
    global recording, temp_file, stream, frames, recording_thread
    if recording:
        console.print("[bold yellow]Already recording.[/bold yellow]")
        return
    recording = True
    console.print("[bold green]Recording started...[/bold green]")
    temp_file = os.path.join(script_dir, "temp_audio_file.wav")
    frames = []
    try:
        stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    except Exception as e:
        console.print(f"[bold red]Failed to start recording: {e}[/bold red]")
        recording = False
        return
    recording_thread = threading.Thread(target=record_audio, args=(temp_file,), daemon=True)
    recording_thread.start()

def record_audio(filename):
    global recording, frames
    try:
        while recording:
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
    except Exception as e:
        console.print(f"[bold red]Recording error: {e}[/bold red]")
    finally:
        if stream:
            stream.stop_stream()
            stream.close()
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
        console.print(f"[bold green]Recording saved to {filename}[/bold green]")
        recording = False

def stop_recording_and_transcribe():
    global recording, transcribing, temp_file, recording_thread
    if not recording:
        console.print("[bold yellow]Not currently recording.[/bold yellow]")
        return
    recording = False
    console.print("[bold blue]Stopping recording and transcribing...[/bold blue]")
    
    if recording_thread is not None:
        recording_thread.join()  # Wait for the recording thread to finish
    
    transcribing = True
    threading.Thread(target=transcribe, args=(temp_file,), daemon=True).start()

def transcribe(filename):
    global transcribing, temp_file
    try:
        segments, info = model.transcribe(filename, language=language, task=task)
        text = ''.join([segment.text for segment in segments])
        console.print(f"[bold magenta]Transcription:[/bold magenta] {text}")
        if paste_enabled:
            pyperclip.copy(text)
            keyboard.send('ctrl+v')  # Paste the transcribed text
            console.print("[bold green]Transcription pasted to active field.[/bold green]")
        os.remove(filename)
    except Exception as e:
        console.print(f"[bold red]Transcription failed: {e}[/bold red]")
    finally:
        transcribing = False
        temp_file = None

def reset():
    global recording, transcribing, temp_file, frames, stream, recording_thread
    try:
        if recording:
            recording = False
            if stream is not None:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception as e:
                    console.print(f"[bold red]Failed to stop/close stream: {e}[/bold red]")
                stream = None
            if recording_thread is not None and recording_thread.is_alive():
                recording_thread.join()
            console.print("[bold yellow]Recording stopped.[/bold yellow]")
        if transcribing and temp_file:
            try:
                os.remove(temp_file)
                console.print("[bold yellow]Temp audio file deleted.[/bold yellow]")
            except Exception as e:
                console.print(f"[bold red]Failed to delete temp file: {e}[/bold red]")
            transcribing = False
            temp_file = None
        frames = []
        console.print("[bold yellow]Reset completed.[/bold yellow]")
    except Exception as e:
        console.print(f"[bold red]Reset failed: {e}[/bold red]")

# Set up global hotkeys
keyboard.add_hotkey('F2', toggle_paste, suppress=True)
keyboard.add_hotkey('F3', start_recording, suppress=True)
keyboard.add_hotkey('F4', stop_recording_and_transcribe, suppress=True)
keyboard.add_hotkey('F5', reset, suppress=True)

console.print("[bold cyan]Hotkeys:[/bold cyan] F2: Toggle paste | F3: Start recording | F4: Stop & Transcribe | F5: Reset")

# If paste_enabled is True at start, ensure it's enabled
if paste_enabled:
    console.print("[bold green]Paste is enabled on start.[/bold green]")

# Keep the script running
keyboard.wait()