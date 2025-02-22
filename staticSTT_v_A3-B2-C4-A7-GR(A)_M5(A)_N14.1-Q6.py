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
import struct
import re
import glob

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
USE_SYSTEM_AUDIO = False  # Set to True to capture system audio via stereomix
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

recording = False
current_session = None

# PyAudio parameters
audio = pyaudio.PyAudio()
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024

# --------------------------------------------------------------------------------------
# GROK modifications
# --------------------------------------------------------------------------------------
class Session:
    def __init__(self):
        import datetime
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(os.path.abspath(os.path.curdir), f"session_{now}")
        os.mkdir(self.session_dir)
        self.current_chunk_index = 1
        self.active_filename = None
        self.active_wave_file = None
        self.partial_transcripts = []
        self.transcription_threads = []
        self.stream = None
        self.recording_thread = None
        self.buffer = []
        self.silent_duration = 0.0
        self.record_start_time = 0
        self.next_split_time = 0
        self.chunk_split_requested = False
        self.chunks_per_second = RATE // CHUNK

    def start_recording(self):
        self.open_stream()
        self.open_first_chunk()
        self.start_recording_loop()

    def open_stream(self):
        self.stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
            input_device_index=INPUT_DEVICE_INDEX if USE_SYSTEM_AUDIO else None
        )

    def open_first_chunk(self):
        self.active_filename = os.path.join(self.session_dir, f"temp_audio_file{self.current_chunk_index}.wav")
        self.active_wave_file = wave.open(self.active_filename, 'wb')
        self.active_wave_file.setnchannels(CHANNELS)
        self.active_wave_file.setsampwidth(audio.get_sample_size(FORMAT))
        self.active_wave_file.setframerate(RATE)
        self.current_chunk_index += 1

    def start_recording_loop(self):
        self.recording_thread = threading.Thread(target=self.record_audio, daemon=True)
        self.recording_thread.start()

    def record_audio(self):
        global recording
        chunk_count = 0
        while recording:
            data = self.stream.read(CHUNK, exception_on_overflow=False)
            samples = struct.unpack('<{}h'.format(len(data)//2), data)
            peak = max(abs(sample) for sample in samples)
            chunk_time = float(CHUNK) / RATE
            now = time.time()
            elapsed = now - self.record_start_time
            if (not self.chunk_split_requested) and (elapsed >= (self.next_split_time - self.record_start_time)):
                console.print(f"[yellow]Reached {int(elapsed)} seconds. Will split on next silence.[/yellow]")
                self.chunk_split_requested = True
            if peak < THRESHOLD:
                self.silent_duration += chunk_time
                if self.silent_duration <= SILENCE_LIMIT_SEC:
                    self.buffer.append(data)
                    chunk_count += 1
                if self.chunk_split_requested and (self.silent_duration >= 0.1):
                    console.print("[bold green]Splitting now at silence...[/bold green]")
                    self.split_current_chunk()
                    self.next_split_time += CHUNK_SPLIT_INTERVAL
                    self.chunk_split_requested = False
            else:
                self.silent_duration = 0.0
                self.buffer.append(data)
                chunk_count += 1
            if chunk_count >= self.chunks_per_second:
                self.active_wave_file.writeframes(b''.join(self.buffer))
                self.buffer = []
                chunk_count = 0
        if self.buffer:
            self.active_wave_file.writeframes(b''.join(self.buffer))
            self.buffer = []
        if self.active_wave_file:
            self.active_wave_file.close()
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        finalize_thread = threading.Thread(target=self.finalize, daemon=True)
        finalize_thread.start()

    def split_current_chunk(self):
        if self.active_wave_file:
            self.active_wave_file.close()
            chunk_path = self.active_filename
            t = threading.Thread(target=self.partial_transcribe, args=(chunk_path,), daemon=True)
            t.start()
            self.transcription_threads.append(t)
        new_filename = os.path.join(self.session_dir, f"temp_audio_file{self.current_chunk_index}.wav")
        self.current_chunk_index += 1
        self.active_filename = new_filename
        self.active_wave_file = wave.open(new_filename, 'wb')
        self.active_wave_file.setnchannels(CHANNELS)
        self.active_wave_file.setsampwidth(audio.get_sample_size(FORMAT))
        self.active_wave_file.setframerate(RATE)
        console.print(f"[green]Opened new chunk file: {os.path.basename(new_filename)}[/green]")

    def partial_transcribe(self, chunk_path):
        segments, info = model.transcribe(chunk_path, language=language, task=task)
        text = "".join(s.text for s in segments)
        for pattern in HALLUCINATIONS_REGEX:
            text = pattern.sub("", text)
        console.print(f"[cyan]Partial transcription of {os.path.basename(chunk_path)}[/cyan]")
        console.print(f"[bold magenta]{text}[/bold magenta]\n")
        self.partial_transcripts.append(text)

    def finalize(self):
        for t in self.transcription_threads:
            t.join()
        full_text = "".join(self.partial_transcripts)
        panel = Panel(
            f"[bold magenta]Final Combined Transcription for {os.path.basename(self.session_dir)}:[/bold magenta] {full_text}",
            title="Transcription",
            border_style="yellow"
        )
        console.print(panel)
        if paste_enabled:
            pyperclip.copy(full_text)
            keyboard.send('ctrl+v')
        import shutil
        shutil.rmtree(self.session_dir, ignore_errors=True)
        console.print("[italic green]Session directory deleted.[/italic green]")

# --------------------------------------------------------------------------------------
# Hotkey Handlers
# --------------------------------------------------------------------------------------
def toggle_paste():
    global paste_enabled
    paste_enabled = not paste_enabled
    status = "enabled" if paste_enabled else "disabled"
    console.print(f"[italic green]Paste is now {status}.[/italic]")


def start_recording():
    global recording, current_session
    if recording:
        console.print("[bold yellow]Already recording![/bold yellow]")
        return
    console.print("[bold green]Starting a new recording session[/bold green]")
    current_session = Session()
    recording = True
    current_session.start_recording()


def stop_recording_and_transcribe():
    global recording, current_session
    if not recording:
        console.print("[italic bold yellow]Recording[/italic bold yellow] [italic]not in progress[/italic]")
        return
    console.print("[bold blue]Stopping recording and transcribing...[/bold blue]")
    recording = False
    current_session.recording_thread.join()
    console.print("[green]Recording stopped.[/green]")
    current_session = None


# --------------------------------------------------------------------------------------
# Hotkeys
# --------------------------------------------------------------------------------------
def setup_hotkeys():
    keyboard.add_hotkey('F2', toggle_paste, suppress=True)
    keyboard.add_hotkey('F3', start_recording, suppress=True)
    keyboard.add_hotkey('F4', stop_recording_and_transcribe, suppress=True)


# --------------------------------------------------------------------------------------
# Startup
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

    if paste_enabled:
        console.print("[italic green]Typing is enabled on start.[/italic green]")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    startup()
    keyboard.wait()
