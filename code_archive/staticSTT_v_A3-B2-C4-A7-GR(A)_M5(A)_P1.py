import time
import torch
import keyboard
import pyaudio
import wave
import os
import threading
import pyperclip
import audioop
import re
import glob
import logging
import numpy as np
from rich.console import Console
from rich.panel import Panel
from faster_whisper import WhisperModel
from scipy.io import wavfile
import noisereduce as nr

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
THRESHOLD = 500
SILENCE_LIMIT_SEC = 1.5
CHUNK_SPLIT_INTERVAL = 60

# Language-specific settings
INITIAL_PROMPT = (
    "Γεια σας, αυτή είναι μια ομιλία στα Ελληνικά. "
    "Παρακαλώ χρησιμοποιήστε σωστά τονισμούς και σημεία στίξης. "
    "Διευκρινίζεται πως πρόκειται για ανεπίσημη συζήτηση. "
    "Ομιλητής: Άνδρας, Ελληνική προφορά."
)
CUSTOM_WORDS_FILE = os.path.join(os.path.dirname(__file__), "custom_words.txt")

HALLUCINATIONS_REGEX = [
    re.compile(r"\bΥπότιτλοι\s+AUTHORWAVE\b[^\w]*", re.IGNORECASE),
    re.compile(r"\bΣας\s+ευχαριστώ\b[^\w]*", re.IGNORECASE),
    re.compile(r"[^\u0370-\u03FF\u1F00-\u1FFF\u0300-\u0301\u0308\u0342\s.,!?\-]", re.UNICODE),
]

# --------------------------------------------------------------------------------------
# Globals
# --------------------------------------------------------------------------------------
console = Console()
script_dir = os.path.dirname(os.path.abspath(__file__))

# Configure logging
logging.basicConfig()
logging.getLogger("faster_whisper").setLevel(logging.WARNING)
logging.getLogger("numba").setLevel(logging.WARNING)

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
partial_transcripts = []
transcription_threads = []
current_chunk_index = 1
record_start_time = 0
next_split_time = 0

# Audio parameters
audio = pyaudio.PyAudio()
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
chunks_per_second = RATE // CHUNK
buffer = []

# --------------------------------------------------------------------------------------
# Audio Processing
# --------------------------------------------------------------------------------------
def preprocess_audio(filepath):
    """Enhance audio quality before transcription"""
    try:
        # Read audio file
        rate, data = wavfile.read(filepath)
        
        # Convert to float32 for processing
        if data.dtype != np.float32:
            data = data.astype(np.float32) / np.iinfo(data.dtype).max

        # Reduce noise even in quiet environments
        reduced_noise = nr.reduce_noise(
            y=data,
            sr=rate,
            stationary=True,
            n_fft=1024,
            use_tensorflow=True
        )

        # Normalize volume
        max_val = np.max(np.abs(reduced_noise))
        if max_val > 0:
            reduced_noise = (reduced_noise / max_val) * 0.9

        # Convert back to int16 and save
        int_data = (reduced_noise * 32767).astype(np.int16)
        
        with wave.open(filepath, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(RATE)
            wf.writeframes(int_data.tobytes())
            
    except Exception as e:
        console.print(f"[yellow]Audio preprocessing failed: {e}[/yellow]")

# --------------------------------------------------------------------------------------
# Text Processing
# --------------------------------------------------------------------------------------
def enhance_greek_text(text):
    """Post-process transcribed text for Greek language"""
    replacements = {
        r"\bσπυ\b": "σου",
        r"\bπρπει\b": "πρέπει",
        r"\bθάτoυ\b": "θάτου",
        r"\bμ(?:ου|ο)π(?:ο|ω)ρ(?:ό|ω)\b": "μπορώ",
        r"\bετ(?:σ|ζ)ι\b": "έτσι",
        r"\bνα\s+το\b": "να το",
        r"\bσ\s+αυτό\b": "σ' αυτό"
    }
    
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # Improve punctuation spacing
    text = re.sub(r"(?<=[.,!?])(?=\S)", " ", text)
    text = re.sub(r"\s+([.,!?])", r"\1", text)
    
    return text.strip()

# --------------------------------------------------------------------------------------
# Transcription Core
# --------------------------------------------------------------------------------------
def partial_transcribe(chunk_path):
    """Enhanced transcription with preprocessing and custom words"""
    global partial_transcripts
    
    try:
        # 1. Audio preprocessing
        preprocess_audio(chunk_path)
        
        # 2. Load custom vocabulary
        custom_words = []
        if os.path.exists(CUSTOM_WORDS_FILE):
            with open(CUSTOM_WORDS_FILE, "r", encoding="utf-8") as f:
                custom_words = [line.strip() for line in f if line.strip()]
        
        # 3. Transcribe with enhanced parameters
        segments, info = model.transcribe(
            chunk_path,
            language=language,
            task=task,
            beam_size=10,
            best_of=10,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            initial_prompt=INITIAL_PROMPT,
            vad_filter=True,
            suppress_blank=False,
            word_timestamps=True,
            repetition_penalty=1.2,
            word_boost=custom_words,
            boost_num=2.0
        )
        
        text = "".join(s.text for s in segments)
        
        # 4. Post-processing
        for pattern in HALLUCINATIONS_REGEX:
            text = pattern.sub("", text)
            
        text = enhance_greek_text(text)
        
        console.print(f"[cyan]Partial transcription of {os.path.basename(chunk_path)}[/cyan]")
        console.print(f"[bold magenta]{text}[/bold magenta]\n")

        partial_transcripts.append(text)
        
    except Exception as e:
        console.print(f"[bold red]Transcription failed for {chunk_path}: {e}[/bold red]")

# --------------------------------------------------------------------------------------
# Internal Reset
# --------------------------------------------------------------------------------------
def internal_reset():
    """
    Reset all relevant variables so user can press F3 again immediately.
    Called after F4 has finished transcription & printing.
    """
    global recording, recording_thread, stream
    global active_wave_file, active_filename
    global partial_transcripts, transcription_threads
    global current_chunk_index, record_start_time, next_split_time
    global buffer

    recording = False
    recording_thread = None
    stream = None
    if active_wave_file:
        active_wave_file.close()
        active_wave_file = None
    active_filename = None

    partial_transcripts.clear()
    transcription_threads.clear()

    current_chunk_index = 1
    record_start_time = 0
    next_split_time = 0

    buffer = []

# --------------------------------------------------------------------------------------
# Utility Functions
# --------------------------------------------------------------------------------------
def cleanup_before_recording():
    """
    Whenever F3 is pressed, we delete all temp_audio_file*.wav (including the plain one).
    This ensures a clean start each session. If the script had crashed previously,
    leftover files won't cause confusion with chunk numbering.
    """
    temp_files = glob.glob(os.path.join(script_dir, "temp_audio_file*.wav"))
    for f in temp_files:
        try:
            os.remove(f)
            console.print(f"[yellow]Deleted file: {os.path.basename(f)}[/yellow]")
        except Exception as e:
            console.print(f"[red]Failed to delete {os.path.basename(f)}: {e}[/red]")

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
    Press F3 to start a brand new recording session.
      1) Cleanup leftover temp_audio_file*.wav
      2) Reset chunk indexing to 1
      3) Open temp_audio_file1.wav for writing
      4) Launch record_audio() in a thread
    """
    global recording, recording_thread
    global partial_transcripts, transcription_threads
    global buffer, current_chunk_index
    global record_start_time, next_split_time
    global active_filename, active_wave_file, stream

    if recording:
        console.print("[bold yellow]Already recording![/bold yellow]")
        return

    console.print("[bold green]Starting a new recording session[/bold green]")

    # 1) Cleanup old files from any previous session
    cleanup_before_recording()

    # 2) Reset all internal state
    partial_transcripts.clear()
    transcription_threads.clear()
    buffer = []
    current_chunk_index = 1

    # 3) Timing
    record_start_time = time.time()
    next_split_time = record_start_time + CHUNK_SPLIT_INTERVAL

    # 4) Set recording True, open "temp_audio_file1.wav"
    recording = True
    first_file = os.path.join(script_dir, f"temp_audio_file{current_chunk_index}.wav")
    active_filename = first_file

    # Open mic stream
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

    # Open wave file
    try:
        active_wave_file = wave.open(first_file, 'wb')
        active_wave_file.setnchannels(CHANNELS)
        active_wave_file.setsampwidth(audio.get_sample_size(FORMAT))
        active_wave_file.setframerate(RATE)
    except Exception as e:
        console.print(f"[bold red]Failed to open wave file: {e}[/bold red]")
        recording = False
        return

    recording_thread = threading.Thread(target=record_audio, daemon=True)
    recording_thread.start()


def record_audio():
    """
    Main recording loop:
      - Reads from mic
      - Trims silence
      - Splits into new files every 1 minute
    """
    global recording, active_wave_file, active_filename, buffer
    global record_start_time, next_split_time, current_chunk_index
    global recording_thread

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

            # Write to wave once per second
            if chunk_count >= chunks_per_second:
                active_wave_file.writeframes(b''.join(buffer))
                buffer = []
                chunk_count = 0

            # Check if it's time to split
            now = time.time()
            if now >= next_split_time:
                split_current_chunk()
                current_chunk_index += 1
                next_split_time += CHUNK_SPLIT_INTERVAL

        # End of while => user pressed F4
        # Write leftover buffer
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

        recording = False
        recording_thread = None
        console.print("[green]Recording stopped.[/green]")


def split_current_chunk():
    """
    Closes the current chunk file, spawns partial transcription,
    opens a new chunk file for the next minute.
    No renaming. The chunk is already 'temp_audio_fileN.wav'.
    We'll proceed with 'temp_audio_file(N+1).wav' next.
    """
    global active_wave_file, active_filename, current_chunk_index
    global transcription_threads

    # 1) Close the current wave file
    if active_wave_file:
        active_wave_file.close()

    # 2) Spawn partial transcribe for the chunk we just closed
    chunk_path = active_filename  # e.g. .../temp_audio_file1.wav, etc.
    t = threading.Thread(target=partial_transcribe, args=(chunk_path,))
    t.start()
    transcription_threads.append(t)

    # 3) Now open the next chunk file => "temp_audio_file{current_chunk_index+1}.wav"
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
        segments, info = model.transcribe(chunk_path, language=language, task=task)
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

    # 1) Stop the recording loop
    console.print("[bold blue]Stopping recording and transcribing...[/bold blue]")
    recording = False

    if recording_thread:
        recording_thread.join()

    # 2) If the last chunk has data, transcribe it
    if active_filename and os.path.exists(active_filename):
        if os.path.getsize(active_filename) > 44:
            final_chunk = active_filename
            # Transcribe final chunk
            t = threading.Thread(target=partial_transcribe, args=(final_chunk,))
            t.start()
            transcription_threads.append(t)

    # 3) Wait for partial transcripts
    console.print("[blue]Waiting for partial transcriptions...[/blue]")
    for t in transcription_threads:
        t.join()

    # 4) Combine partial transcripts
    full_text = "".join(partial_transcripts)

    # 5) Print final text & paste
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


# --------------------------------------------------------------------------------------
# Startup
# --------------------------------------------------------------------------------------
def startup():
    # We do NOT do cleanup here anymore. We'll do it on F3 each time.
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
