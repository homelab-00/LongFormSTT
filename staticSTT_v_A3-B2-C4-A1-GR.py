import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import keyboard
import pyaudio
import wave
import tempfile
import os
import threading
import pyperclip
from rich import print
from rich.console import Console

# Initialize Rich console
console = Console()

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
    chunk_length_s=None
)

# Adjust max_new_tokens to ensure total length <= 448
max_new_tokens = 445

# Transcription paste toggle
paste_enabled = False
temp_file = None
recording = False
transcribing = False
lock = threading.Lock()

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
    console.print(f"[bold yellow]Paste enabled:[/bold yellow] {paste_enabled}")

def start_recording():
    global recording, temp_file, stream, frames
    if recording:
        return
    recording = True
    console.print("[bold green]Recording started...[/bold green]")
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    frames = []
    try:
        stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    except Exception as e:
        console.print(f"[bold red]Failed to start recording: {e}[/bold red]")
        recording = False
        return
    threading.Thread(target=record_audio, args=(temp_file.name,), daemon=True).start()

def record_audio(filename):
    global recording, frames
    with lock:
        try:
            while recording:
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
        except Exception as e:
            console.print(f"[bold red]Recording error: {e}[/bold red]")
        finally:
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
    global recording, transcribing, temp_file
    if not recording:
        return
    recording = False
    console.print("[bold blue]Stopping recording and transcribing...[/bold blue]")
    transcribing = True
    threading.Thread(target=transcribe, args=(temp_file.name,), daemon=True).start()

def transcribe(filename):
    global transcribing, temp_file
    try:
        result = pipe(
            filename,
            return_timestamps=True,
            generate_kwargs={
                "num_beams": 5,
                "temperature": [0.0],
                "max_new_tokens": max_new_tokens,
                "condition_on_prev_tokens": False,
                "compression_ratio_threshold": 2.4,
                "logprob_threshold": -1.0,
                "no_speech_threshold": 0.6
            }
        )
        text = result["text"]
        console.print(f"[bold magenta]Transcription:[/bold magenta] {text}")
        if paste_enabled:
            pyperclip.copy(text)
            console.print("[bold green]Transcription copied to clipboard[/bold green]")
        os.remove(filename)
    except Exception as e:
        console.print(f"[bold red]Transcription failed: {e}[/bold red]")
    finally:
        transcribing = False
        temp_file = None

def reset():
    global recording, transcribing, temp_file, frames
    with lock:
        if recording:
            recording = False
            if stream is not None:
                stream.stop_stream()
                stream.close()
            console.print("[bold yellow]Recording stopped.[/bold yellow]")
        if transcribing and temp_file:
            try:
                os.remove(temp_file.name)
                console.print("[bold yellow]Temp audio file deleted.[/bold yellow]")
            except Exception as e:
                console.print(f"[bold red]Failed to delete temp file: {e}[/bold red]")
            transcribing = False
            temp_file = None
        frames = []
        console.print("[bold yellow]Reset completed[/bold yellow]")

# Set up global hotkeys
keyboard.add_hotkey('F2', toggle_paste, suppress=True)
keyboard.add_hotkey('F3', start_recording, suppress=True)
keyboard.add_hotkey('F4', stop_recording_and_transcribe, suppress=True)
keyboard.add_hotkey('F5', reset, suppress=True)

console.print("[bold cyan]Hotkeys:[/bold cyan] F2: Toggle paste | F3: Start recording | F4: Stop & Transcribe | F5: Reset")

# Keep the script running
keyboard.wait()