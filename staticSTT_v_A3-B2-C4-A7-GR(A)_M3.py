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
import webrtcvad
import time

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
wave_file = None  # Handle the wave file

# PyAudio parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
FRAME_DURATION_MS = 30  # Frame size for VAD
CHUNK = int(RATE * FRAME_DURATION_MS / 1000)  # Number of samples per frame
audio = pyaudio.PyAudio()
stream = None

buffer = []  # Buffer to accumulate audio data
chunks_per_second = RATE // CHUNK  # Calculate chunks that make up one second

# Initialize VAD
vad = webrtcvad.Vad(2)  # Aggressiveness mode: 0-3

# Silence detection parameters
SILENCE_DURATION = 1.0  # seconds of silence to consider as a pause
SILENCE_CHUNKS = int(SILENCE_DURATION * RATE / CHUNK)
FIXED_SILENCE_DURATION = 1.0  # seconds of fixed silence to insert
FIXED_SILENCE_CHUNKS = int(FIXED_SILENCE_DURATION * RATE / CHUNK)
FIXED_SILENCE = b'\x00\x00' * CHUNK * FIXED_SILENCE_CHUNKS  # 16-bit silence

# Pre-silence buffer to prevent clipping
PRE_SILENCE_DURATION = 0.5  # seconds
PRE_SILENCE_CHUNKS = int(PRE_SILENCE_DURATION * RATE / CHUNK)
pre_silence_buffer = []

# Flag to prevent multiple silence logs
silence_logged = False

def toggle_paste():
    global paste_enabled
    paste_enabled = not paste_enabled
    status = "enabled" if paste_enabled else "disabled"
    console.print(f"[italic green]Paste[/italic green] [italic]{status}.[/italic]")

def start_recording():
    global recording, temp_file, stream, wave_file, recording_thread, silence_logged
    if recording:
        console.print("[italic bold yellow]Recording[/italic bold yellow] [italic]in progress.[/italic]")
        return
    recording = True
    silence_logged = False  # Reset silence log flag
    console.print("[italic bold green]Starting[/italic bold green] [italic]recording[/italic]")
    temp_file = os.path.join(script_dir, "temp_audio_file.wav")
    try:
        stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
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
    global recording, wave_file, buffer, silence_logged
    silence_counter = 0
    try:
        while recording:
            data = stream.read(CHUNK, exception_on_overflow=False)
            is_speech = vad.is_speech(data, RATE)
            
            if is_speech:
                # Reset silence counter and log flag
                silence_counter = 0
                silence_logged = False

                # Append data to buffer
                buffer.append(data)

                # Handle pre-silence buffer
                if pre_silence_buffer:
                    wave_file.writeframes(b''.join(pre_silence_buffer))
                    pre_silence_buffer.clear()

            else:
                silence_counter += 1

                if silence_counter >= SILENCE_CHUNKS:
                    if not silence_logged:
                        console.print("[italic blue]Silence detected. Inserting fixed silence.[/italic blue]")
                        silence_logged = True
                    # Insert fixed silence
                    wave_file.writeframes(FIXED_SILENCE)
                    silence_counter = 0  # Reset counter after inserting silence

                else:
                    # Optional: Store data in pre-silence buffer to prevent clipping
                    pre_silence_buffer.append(data)

            # Write to wave file if buffer accumulates enough data (e.g., 1 second)
            if len(buffer) >= chunks_per_second:
                wave_file.writeframes(b''.join(buffer))
                buffer.clear()
    except Exception as e:
        console.print(f"[bold red]Recording error: {e}[/bold red]")
    finally:
        # Write any remaining data in buffer to file
        if buffer:
            wave_file.writeframes(b''.join(buffer))
            buffer = []
        if pre_silence_buffer:
            wave_file.writeframes(b''.join(pre_silence_buffer))
            pre_silence_buffer.clear()
        if stream:
            stream.stop_stream()
            stream.close()
        if wave_file:
            wave_file.close()  # Close the wave file after recording
        console.print("[italic green]Recording[/italic green] [italic]saved to temp audio file[/italic]")
        recording = False
        # Automatically start transcription if recording stopped by user
        if not transcribing:
            stop_recording_and_transcribe()

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
        transcription_panel = Panel(f"[bold magenta]Transcription:[/bold magenta] {text}", title="Transcription", border_style="yellow")
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
    "[bold yellow]Hotkeys[/bold yellow]: [bold green]F2[/bold green] - Toggle typing | [bold green]F3[/bold green] - Start recording | [bold green]F4[/bold green] - Stop & Transcribe"
)
panel = Panel(panel_content, title="Information", border_style="green")
console.print(panel)

# If paste_enabled is True at start, ensure it's enabled
if paste_enabled:
    console.print("[italic green]Typing[/italic green] [italic]is enabled on start.[/italic]")

# Keep the script running
try:
    keyboard.wait()
except KeyboardInterrupt:
    console.print("\n[italic red]Exiting...[/italic red]")
finally:
    if stream is not None:
        stream.stop_stream()
        stream.close()
    audio.terminate()
