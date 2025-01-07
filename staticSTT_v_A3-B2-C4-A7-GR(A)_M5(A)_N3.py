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

# -------------- CONFIGURATIONS -------------- #
# Silence detection
THRESHOLD = 500
SILENCE_LIMIT_SEC = 1.5

# Chunk splitting
CHUNK_SPLIT_INTERVAL = 120  # seconds (2 minutes)

# Regex-based hallucination filtering
HALLUCINATIONS_REGEX = [
    re.compile(r"\bΥπότιτλοι\s+AUTHORWAVE\b", re.IGNORECASE),
    # Add more patterns if needed
]

# -------------- GLOBALS -------------- #
console = Console()
script_dir = os.path.dirname(os.path.abspath(__file__))

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "Systran/faster-whisper-large-v3"
model = WhisperModel(model_id, device=device, compute_type="float16" if device == "cuda" else "float32")

language = "el"
task = "transcribe"

paste_enabled = True
temp_file = None
recording = False
transcribing = False
lock = threading.Lock()

recording_thread = None
transcription_thread = None

wave_file = None
audio = pyaudio.PyAudio()
stream = None

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024

buffer = []
chunks_per_second = RATE // CHUNK

# For auto-splitting logic
record_start_time = 0.0           # Wall-clock time at F3 press
next_split_time = 0.0            # Time (in seconds) when next 2-minute chunk ends
current_chunk_index = 0          # Which chunk are we on?
partial_transcripts = []         # Holds text for each chunk
transcription_threads = []       # Keep track of in-flight transcription threads

# -------------- HOTKEY HANDLERS -------------- #
def toggle_paste():
    global paste_enabled
    paste_enabled = not paste_enabled
    status = "enabled" if paste_enabled else "disabled"
    console.print(f"[italic green]Paste[/italic green] [italic]{status}.[/italic]")

def start_recording():
    """
    Press F3 to start recording. Immediately open the first chunk file,
    set up timers, etc.
    """
    global recording, temp_file, stream, wave_file
    global record_start_time, next_split_time, current_chunk_index
    if recording:
        console.print("[italic bold yellow]Recording already in progress.[/italic bold yellow]")
        return
    recording = True
    console.print("[italic bold green]Starting recording[/italic bold green]")
    record_start_time = time.time()
    next_split_time = record_start_time + CHUNK_SPLIT_INTERVAL
    current_chunk_index = 0
    partial_transcripts.clear()
    transcription_threads.clear()

    # The "active" chunk is always "temp_audio_file.wav"
    temp_file = os.path.join(script_dir, "temp_audio_file.wav")
    try:
        stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                            input=True, frames_per_buffer=CHUNK)
        wave_file = wave.open(temp_file, 'wb')
        wave_file.setnchannels(CHANNELS)
        wave_file.setsampwidth(audio.get_sample_size(FORMAT))
        wave_file.setframerate(RATE)
    except Exception as e:
        console.print(f"[bold red]Failed to start recording: {e}[/bold red]")
        recording = False
        return

    global recording_thread
    recording_thread = threading.Thread(target=record_audio, daemon=True)
    recording_thread.start()

def record_audio():
    """
    Continuously record microphone audio into temp_audio_file.wav,
    trimming silence. Every CHUNK_SPLIT_INTERVAL seconds of wall-clock time,
    we 'split' the current chunk and start a new one.
    """
    global recording, wave_file, buffer
    global record_start_time, next_split_time, current_chunk_index

    chunk_count = 0
    silence_duration = 0.0

    try:
        while recording:
            data = stream.read(CHUNK, exception_on_overflow=False)
            peak = audioop.max(data, 2)
            chunk_time = float(CHUNK) / RATE

            # Silence trimming
            if peak < THRESHOLD:
                silence_duration += chunk_time
                if silence_duration <= SILENCE_LIMIT_SEC:
                    buffer.append(data)
                    chunk_count += 1
            else:
                silence_duration = 0.0
                buffer.append(data)
                chunk_count += 1

            # Write to file once per second of audio
            if chunk_count >= chunks_per_second:
                wave_file.writeframes(b''.join(buffer))
                buffer = []
                chunk_count = 0

            # Check for chunk-split time
            now = time.time()
            if now >= next_split_time:
                # We have reached 2 minutes of real time. Let's split.
                split_recording_chunk()
                next_split_time += CHUNK_SPLIT_INTERVAL
                current_chunk_index += 1

        # If we exit the while (F4 pressed), just handle leftover buffer
        if buffer:
            wave_file.writeframes(b''.join(buffer))
            buffer = []

    except Exception as e:
        console.print(f"[bold red]Recording error: {e}[/bold red]")
    finally:
        if wave_file:
            wave_file.close()
        if stream:
            stream.stop_stream()
            stream.close()
        console.print("[italic green]Recording finished[/italic green] (temp file closed)")
        recording = False

def split_recording_chunk():
    """
    Called every time we've hit 2 minutes of real time.
    Closes the current wave_file, spawns a transcription thread,
    then re-opens wave_file for the next chunk.
    """
    global wave_file, temp_file

    # 1) Close the active wave file
    wave_file.close()

    # 2) Rename that chunk to an indexed file, so we can still transcribe it
    #    without overwriting
    chunk_path = temp_file.replace(".wav", f"_{int(time.time())}.wav")
    os.rename(temp_file, chunk_path)

    # 3) Spawn a separate thread to transcribe that chunk
    t = threading.Thread(target=partial_transcribe, args=(chunk_path,))
    t.start()
    transcription_threads.append(t)

    # 4) Open a NEW wave file for the next chunk
    wave_file = wave.open(temp_file, 'wb')
    wave_file.setnchannels(CHANNELS)
    wave_file.setsampwidth(audio.get_sample_size(FORMAT))
    wave_file.setframerate(RATE)

def partial_transcribe(chunk_path):
    """
    Transcribe a single chunk, store the text in partial_transcripts,
    and print it to the console (but do NOT paste).
    Also shows a progress bar for that chunk.
    """
    global partial_transcripts

    # Actually run whisper
    try:
        segments, info = model.transcribe(chunk_path, language=language, task=task)

        # We'll build the text incrementally and also track progress
        total_duration = info.duration
        accumulated = 0.0
        chunk_text = ""

        console.print(f"\n[cyan]Started partial transcription[/cyan] of [bold]{chunk_path}[/bold]")

        for i, seg in enumerate(segments):
            # Remove hallucinations
            seg_text = seg.text
            for pattern in HALLUCINATIONS_REGEX:
                seg_text = pattern.sub("", seg_text)

            chunk_text += seg_text

            # Progress bar update
            segment_length = seg.end - seg.start
            accumulated += segment_length
            progress_pct = min(accumulated / total_duration, 1.0) * 100.0
            # Reprint in-place
            console.print(
                f"\r[cyan]Transcribing {chunk_path}: {progress_pct:.1f}%[/cyan]",
                end="",
                flush=True
            )

        # Final newline after finishing
        console.print()  
        console.print(f"[green]Partial transcription finished[/green] for [bold]{chunk_path}[/bold].")

        # Print partial text
        console.print(f"[bold magenta]Partial text[/bold magenta]: {chunk_text}\n")

        # Save partial text. (We just append in order of finishing.)
        # If you want to preserve chunk ordering strictly, store (timestamp, text) pairs.
        partial_transcripts.append(chunk_text)

    except Exception as e:
        console.print(f"[bold red]Partial transcription failed for {chunk_path}: {e}[/bold red]")

def stop_recording_and_transcribe():
    """
    When F4 is pressed:
      1) Stop the recording loop
      2) Force one last chunk-split (whatever's recorded so far)
      3) Wait for all partial-chunk transcriptions to finish
      4) Combine partials + final chunk => final text
      5) Print final text & paste it (if paste_enabled)
    """
    global recording, temp_file, recording_thread, transcription_thread
    if not recording:
        console.print("[italic bold yellow]Recording not in progress[/italic bold yellow]")
        return

    console.print("[italic bold blue]Stopping recording...[/italic bold blue]")
    recording = False

    # Wait for record_audio() to exit
    if recording_thread is not None:
        recording_thread.join()

    # If there's anything in temp_audio_file.wav not yet chunked, that means
    # we had a partial chunk at the end. Let's forcibly split it for transcription.
    # But first, make sure we haven't already closed it
    if os.path.exists(temp_file):
        # 'wave_file' in record_audio() is closed after the loop,
        # so let's confirm it is closed before rename:
        # We spawn partial transcription for the final chunk.
        if os.path.getsize(temp_file) > 44:  # WAV header is ~44 bytes, so > 44 => has audio
            final_chunk = temp_file.replace(".wav", f"_{int(time.time())}.wav")
            try:
                os.rename(temp_file, final_chunk)
                # Transcribe final chunk
                t = threading.Thread(target=partial_transcribe, args=(final_chunk,))
                t.start()
                transcription_threads.append(t)
            except:
                pass

    console.print("[italic blue]Waiting for all partial transcription threads to complete...[/italic blue]")
    for t in transcription_threads:
        t.join()  # Wait until all chunk transcripts are done

    console.print("[italic green]All chunk transcripts are ready.[/italic green]")

    # Combine all partial transcripts
    full_text = "".join(partial_transcripts)

    # Print final text
    final_panel = Panel(
        f"[bold magenta]Final Combined Transcription:[/bold magenta] {full_text}",
        title="Transcription",
        border_style="yellow"
    )
    console.print(final_panel)

    # Now do the single paste, if enabled
    if paste_enabled:
        pyperclip.copy(full_text)
        keyboard.send('ctrl+v')
    console.print("[italic green]Done.[/italic green]")


# -------------- HOTKEY REGISTRATION -------------- #
keyboard.add_hotkey('F2', toggle_paste, suppress=True)
keyboard.add_hotkey('F3', start_recording, suppress=True)
keyboard.add_hotkey('F4', stop_recording_and_transcribe, suppress=True)

# -------------- PANEL & STARTUP -------------- #
panel_content = (
    f"[bold yellow]Model[/bold yellow]  : {model_id}\n"
    "[bold yellow]Hotkeys[/bold yellow]: [bold green]F2[/bold green] - Toggle typing "
    "| [bold green]F3[/bold green] - Start recording "
    "| [bold green]F4[/bold green] - Stop & Transcribe\n"
    "[bold yellow]Silence detection[/bold yellow]: RMS/peak value-based trimming\n"
    "[bold yellow]Chunks[/bold yellow]: Auto-split & transcribe every 2 min\n"
    "[bold yellow]Transcripts[/bold yellow]: Partial shown (no paste), final combined with paste"
)
panel = Panel(panel_content, title="Information", border_style="green")
console.print(panel)

if paste_enabled:
    console.print("[italic green]Typing is enabled on start.[/italic green]")

# Keep the script running
keyboard.wait()
