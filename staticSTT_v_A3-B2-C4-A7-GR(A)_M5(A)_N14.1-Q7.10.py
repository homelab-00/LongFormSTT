import time
import torch
import sys
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
import socket
import subprocess
import psutil

# [MOD] For beep sounds on Windows
if sys.platform.startswith("win"):
    import winsound

# [MOD] For file dialog (transcribe_static)
import tkinter
from tkinter import filedialog

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
USE_SYSTEM_AUDIO = False  
INPUT_DEVICE_INDEX = 2

THRESHOLD = 500
SILENCE_LIMIT_SEC = 1.5
CHUNK_SPLIT_INTERVAL = 60

HALLUCINATIONS_REGEX = [
    re.compile(r"\bΥπότιτλοι\s+AUTHORWAVE\b[^\w]*", re.IGNORECASE),
    re.compile(r"\bΣας\s+ευχαριστώ\b[^\w]*", re.IGNORECASE),
]

console = Console()
script_dir = os.path.dirname(os.path.abspath(__file__))

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "Systran/faster-whisper-large-v3"
model = WhisperModel(model_id, device=device, compute_type="float16" if device == "cuda" else "float32")

language = "el"
task = "transcribe"

# Recording state
recording = False
recording_thread = None
keep_running = True
stream = None
active_wave_file = None
active_filename = None

partial_transcripts_dict = {}
transcription_threads = []

current_chunk_index = 1
record_start_time = 0
next_split_time = 0
chunk_split_requested = False

audio = pyaudio.PyAudio()
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
chunks_per_second = RATE // CHUNK
buffer = []

# AutoHotkey script
hotkey_script = "Hotkeys-AHK_A1.ahk"
ahk_pid = None

send_enter = True

# --------------------------------------------------------------------------------------
# Existing Functions (unchanged except for beep additions)
# --------------------------------------------------------------------------------------

def toggle_language():
    global language, model
    old_lang = language
    
    # Toggle the language code
    if language == "el":
        language = "en"
    else:
        language = "el"
    
    console.print(f"[yellow]Language toggled from {old_lang} to {language}[/yellow]")

def cleanup_before_recording():
    temp_files = glob.glob(os.path.join(script_dir, "temp_audio_file*.wav"))
    for f in temp_files:
        try:
            os.remove(f)
            console.print(f"[yellow]Deleted file: {os.path.basename(f)}[/yellow]")
        except Exception as e:
            console.print(f"[red]Failed to delete {os.path.basename(f)}: {e}[/red]")

def start_recording():
    global recording, recording_thread
    global transcription_threads
    global buffer, current_chunk_index
    global partial_transcripts_dict
    global record_start_time, next_split_time, chunk_split_requested
    global active_filename, active_wave_file, stream

    if recording:
        console.print("[bold yellow]Already recording![/bold yellow]")
        return

    console.print("[bold green]Starting a new recording session[/bold green]")
    cleanup_before_recording()

    partial_transcripts_dict.clear()
    transcription_threads.clear()
    buffer = []
    current_chunk_index = 1

    record_start_time = time.time()
    next_split_time = record_start_time + CHUNK_SPLIT_INTERVAL
    chunk_split_requested = False

    recording = True
    first_file = os.path.join(script_dir, f"temp_audio_file{current_chunk_index}.wav")
    active_filename = first_file

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
        recording = False
        return

    try:
        active_wave_file = wave.open(first_file, 'wb')
        active_wave_file.setnchannels(CHANNELS)
        active_wave_file.setsampwidth(audio.get_sample_size(FORMAT))
        active_wave_file.setframerate(RATE)
    except Exception as e:
        console.print(f"[bold red]Failed to open wave file: {e}[/bold red]")
        recording = False
        return

    # [MOD] Play a short beep to indicate recording has started (Windows only)
    if sys.platform.startswith("win"):
        try:
            console.print("[dim green]Playing start-recording beep...[/dim green]")
            winsound.Beep(440, 200)  # frequency=440Hz, duration=200ms
        except Exception as beep_err:
            console.print(f"[red]Beep error: {beep_err}[/red]")

    recording_thread = threading.Thread(target=record_audio, daemon=True)
    recording_thread.start()

def record_audio():
    global recording, active_wave_file, active_filename, buffer
    global record_start_time, next_split_time, current_chunk_index
    global chunk_split_requested, recording_thread

    chunk_count = 0
    silence_duration = 0.0

    try:
        while recording:
            data = stream.read(CHUNK, exception_on_overflow=False)
            samples = struct.unpack('<{}h'.format(len(data)//2), data)
            peak = max(abs(sample) for sample in samples)
            chunk_time = float(CHUNK) / RATE
            now = time.time()
            elapsed = now - record_start_time

            if (not chunk_split_requested) and (elapsed >= (next_split_time - record_start_time)):
                console.print(f"[yellow]Reached {int(elapsed)} seconds. Will split on next silence.[/yellow]")
                chunk_split_requested = True

            if peak < THRESHOLD:
                silence_duration += chunk_time
                if silence_duration <= SILENCE_LIMIT_SEC:
                    buffer.append(data)
                    chunk_count += 1

                if chunk_split_requested and (silence_duration >= 0.1):
                    console.print("[bold green]Splitting now at silence...[/bold green]")
                    split_current_chunk()

                    next_split_time += CHUNK_SPLIT_INTERVAL
                    chunk_split_requested = False
            else:
                silence_duration = 0.0
                buffer.append(data)
                chunk_count += 1

            if chunk_count >= chunks_per_second:
                active_wave_file.writeframes(b''.join(buffer))
                buffer = []
                chunk_count = 0

        if buffer:
            active_wave_file.writeframes(b''.join(buffer))
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
        recording_thread = None
        console.print("[green]Recording stopped.[/green]")

def split_current_chunk():
    global active_wave_file, active_filename, current_chunk_index
    global transcription_threads

    if active_wave_file:
        active_wave_file.close()

    chunk_path = active_filename

    # --- NEW DEBUG PRINT ---
    if os.path.exists(chunk_path):
        console.print(f"[yellow]split_current_chunk() -> chunk_path: {chunk_path}, size={os.path.getsize(chunk_path)} bytes[/yellow]")
    else:
        console.print(f"[red]split_current_chunk() -> chunk_path: {chunk_path} does NOT exist[/red]")

    t = threading.Thread(target=partial_transcribe, args=(chunk_path, current_chunk_index))
    t.start()
    transcription_threads.append(t)

    current_chunk_index += 1
    new_filename = os.path.join(script_dir, f"temp_audio_file{current_chunk_index}.wav")
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

def partial_transcribe(chunk_path, chunk_idx):
    global partial_transcripts_dict
    try:
        segments, info = model.transcribe(chunk_path, language=language, task=task)
        text = "".join(s.text for s in segments)
        for pattern in HALLUCINATIONS_REGEX:
            text = pattern.sub("", text)

        console.print(f"[cyan]Partial transcription of {os.path.basename(chunk_path)}[/cyan]")
        console.print(f"[bold magenta]{text}[/bold magenta]\n")

        partial_transcripts_dict[chunk_idx] = text
    except Exception as e:
        console.print(f"[bold red]Partial transcription failed for {chunk_path}: {e}[/bold red]")
        partial_transcripts_dict[chunk_idx] = ""

def stop_recording_and_transcribe():
    global recording, recording_thread, active_filename
    global transcription_threads
    global current_chunk_index

    if not recording:
        console.print("[italic bold yellow]Recording not in progress[/italic bold yellow]")
        return

    console.print("[bold blue]Stopping recording and transcribing...[/bold blue]")
    recording = False

    if recording_thread:
        recording_thread.join()

    # --- NEW DEBUG PRINT ---
    if active_filename and os.path.exists(active_filename):
        console.print(f"[yellow]stop_recording_and_transcribe() -> final chunk path: {active_filename}, size={os.path.getsize(active_filename)} bytes[/yellow]")
    else:
        console.print(f"[red]stop_recording_and_transcribe() -> active_filename does NOT exist or is None[/red]")

    # [MOD] Play a short beep to indicate recording has stopped (Windows only)
    if sys.platform.startswith("win"):
        try:
            console.print("[dim green]Playing stop-recording beep...[/dim green]")
            winsound.Beep(600, 300)  # frequency=600Hz, duration=300ms
        except Exception as beep_err:
            console.print(f"[red]Beep error: {beep_err}[/red]")

    if active_filename and os.path.exists(active_filename):
        if os.path.getsize(active_filename) > 44:
            final_chunk = active_filename

            # Use the current chunk index or whatever index you want for the final chunk
            final_chunk_idx = current_chunk_index

            t = threading.Thread(target=partial_transcribe, args=(final_chunk, final_chunk_idx))
            t.start()
            transcription_threads.append(t)
            current_chunk_index += 1

    console.print("[blue]Waiting for partial transcriptions...[/blue]")

    for t in transcription_threads:
        t.join()

    ordered_texts = []
    for idx in sorted(partial_transcripts_dict.keys()):
        ordered_texts.append(partial_transcripts_dict[idx])
    full_text = "".join(ordered_texts)
    if len(full_text) > 0:
        # Remove leading character if it is always a space (only if you want this logic)
        if full_text[0].isspace():
            full_text = full_text[1:]

    panel = Panel(
        f"[bold magenta]Final Combined Transcription:[/bold magenta] {full_text}",
        title="Transcription",
        border_style="yellow"
    )
    console.print(panel)

    pyperclip.copy(full_text)
    keyboard.send('ctrl+v')
    
    if send_enter:
        keyboard.send('enter')
        console.print("[yellow]Sent an ENTER keystroke after transcription.[/yellow]")

    console.print("[italic green]Done.[/italic green]")

# --------------------------------------------------------------------------------------
# [MOD] New helper function: Convert arbitrary audio to WAV (16k, mono) if needed
# --------------------------------------------------------------------------------------
def ensure_wav_format(input_path):
    """
    Checks if the file is a WAV with the correct sample rate/channels.
    If not, uses ffmpeg to convert to a temp WAV file.
    Returns the path of the WAV file to be used for transcription.
    """
    try:
        with wave.open(input_path, 'rb') as wf:
            # Check for the correct format: channels=1, rate=16000
            if wf.getnchannels() == 1 and wf.getframerate() == 16000:
                # Already matches. Let's just return input_path.
                console.print("[blue]No conversion needed (already 16k mono WAV).[/blue]")
                return input_path
    except wave.Error:
        # This means it's not even a valid WAV in the normal sense, we must convert
        pass

    # If we get here, we convert
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    temp_wav = os.path.join(script_dir, f"static_temp_file.wav")

    console.print(f"[cyan]Converting file '{input_path}' -> '{temp_wav}'[/cyan]")
    try:
        subprocess.run([
            "ffmpeg",
            "-y",
            "-i", input_path,
            "-ac", "1",
            "-ar", "16000",
            temp_wav
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        console.print("[cyan]Conversion successful.[/cyan]")
        return temp_wav
    except Exception as e:
        console.print(f"[bold red]FFmpeg conversion error: {e}[/bold red]")
        return None

# --------------------------------------------------------------------------------------
# [MOD] New function: Transcribe a static file from file dialog
# --------------------------------------------------------------------------------------
def transcribe_static_file():
    """
    Opens a file dialog for the user to pick any audio file.
    Converts to the correct format if needed, then transcribes via model.
    """
    console.print("[bold yellow]TRANSCRIBE_STATIC command received[/bold yellow]")
    # Minimal tkinter usage
    root = tkinter.Tk()
    root.withdraw()  # hide the main window

    file_path = filedialog.askopenfilename(
        title="Select an audio file to transcribe",
        filetypes=[("Audio Files", "*.*")]
    )
    root.destroy()

    if not file_path:
        console.print("[red]No file selected. Aborting static transcription.[/red]")
        return

    console.print(f"[green]Selected file: {file_path}[/green]")

    # Convert if necessary
    wav_path = ensure_wav_format(file_path)
    if not wav_path or not os.path.exists(wav_path):
        console.print("[bold red]Could not obtain a valid WAV file. Aborting.[/bold red]")
        return

    # Now we do a direct, normal transcription using the model
    # (We do NOT do partial chunk splitting here since it's a static file.)
    try:
        console.print("[blue]Beginning static transcription...[/blue]")
        segments, info = model.transcribe(wav_path, language=language, task=task)
        final_text = "".join(s.text for s in segments)
        for pattern in HALLUCINATIONS_REGEX:
            final_text = pattern.sub("", final_text)

        # Possibly remove leading whitespace
        if len(final_text) > 0 and final_text[0].isspace():
            final_text = final_text[1:]

        panel = Panel(
            f"[bold magenta]Static File Transcription:[/bold magenta] {final_text}",
            title="Static Transcription",
            border_style="yellow"
        )
        console.print(panel)

        # Copy to clipboard + optionally paste + enter
        pyperclip.copy(final_text)
        keyboard.send('ctrl+v')
        if send_enter:
            keyboard.send('enter')
            console.print("[yellow]Sent an ENTER keystroke after static transcription.[/yellow]")

    except Exception as e:
        console.print(f"[bold red]Static transcription failed: {e}[/bold red]")

# --------------------------------------------------------------------------------------
# TCP Server for AHK -> Python
# --------------------------------------------------------------------------------------
def run_socket_server():
    host = '127.0.0.1'
    port = 34909  # arbitrary free port
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((host, port))
    sock.listen(5)

    # Allow accept() to time out periodically so we can check if keep_running is still True
    sock.settimeout(1)

    console.print(f"[bold yellow]TCP server listening on {host}:{port}[/bold yellow]")

    while True:
        if not keep_running:
            break

        try:
            conn, addr = sock.accept()
        except socket.timeout:
            # Just loop again if no connection is received within the timeout
            continue

        data = conn.recv(1024).decode('utf-8').strip()
        console.print(f"[italic cyan]Received command: '{data}'[/italic cyan]")

        if data == "TOGGLE_LANGUAGE":
            toggle_language()
        elif data == "START_RECORDING":
            start_recording()
        elif data == "STOP_AND_TRANSCRIBE":
            stop_recording_and_transcribe()
        elif data == "TOGGLE_ENTER":
            global send_enter
            send_enter = not send_enter
            console.print(f"[cyan]Toggled send_enter to: {send_enter}[/cyan]")
        elif data == "TRANSCRIBE_STATIC":  # [MOD] New command
            transcribe_static_file()
        elif data == "QUIT":
            console.print("[bold red]Received QUIT command[/bold red]")
            graceful_exit(sock)

        conn.close()

def graceful_exit(sock):
    global keep_running, ahk_pid
    console.print("[bold red]Shutting down gracefully...[/bold red]")
    keep_running = False
    sock.close()  # close the server socket

    if ahk_pid is not None:
        console.print(f"[bold red]Killing AHK script by stored PID={ahk_pid}[/bold red]")
        try:
            psutil.Process(ahk_pid).kill()
        except Exception as e:
            console.print(f"[red]Failed to kill AHK process with PID {ahk_pid}: {e}[/red]")
    else:
        console.print("[yellow]No stored AHK PID. Skipping kill.[/yellow]")

    sys.exit(0)

# --------------------------------------------------------------------------------------
# Startup (modded)
# --------------------------------------------------------------------------------------
def startup():
    global ahk_pid

    panel_content = (
        f"[bold yellow]Model[/bold yellow]: {model_id}\n"
        f"[bold yellow]Hotkeys[/bold yellow]: Controlled by AutoHotKey script '{hotkey_script}'\n"
        " F2 -> toggle language\n"
        " F3 -> start recording\n"
        " F4 -> stop & transcribe\n"
        " F5 -> toggle enter\n"
        " F6 -> quit\n"
        " F10 -> (AHK side) triggers TRANSCRIBE_STATIC (example)\n"
        f"[bold yellow]Language[/bold yellow]: {language}"
    )
    panel = Panel(panel_content, title="Information", border_style="green")
    console.print(panel)

    # Pre-scan: gather all AutoHotkeyU64.exe PIDs before launching
    pre_pids = set()
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['name'] == 'AutoHotkeyU64.exe':
                pre_pids.add(proc.info['pid'])
        except:
            pass

    # Launch .ahk script with shell=True (the Windows default association)
    ahk_path = os.path.join(script_dir, hotkey_script)  # e.g. "Hotkeys-AHK_A1.ahk"
    console.print("[green]Launching AHK script...[/green]")
    subprocess.Popen(
        [ahk_path],
        creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
        shell=True
    )

    # Post-scan: gather all AutoHotkeyU64.exe PIDs after launching
    time.sleep(1.0)  # short pause to let the new process appear
    post_pids = set()
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['name'] == 'AutoHotkeyU64.exe':
                post_pids.add(proc.info['pid'])
        except:
            pass

    # The new PID(s) is the difference
    new_pids = post_pids - pre_pids
    if len(new_pids) == 1:
        ahk_pid = new_pids.pop()
        console.print(f"[green]Detected new AHK script PID: {ahk_pid}[/green]")
    else:
        console.print("[red]Could not detect a single new AHK script PID. No PID stored.[/red]")
        ahk_pid = None


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    startup()

    # 1) Start the TCP server on a background thread
    server_thread = threading.Thread(target=run_socket_server, daemon=True)
    server_thread.start()

    # 2) Keep the main thread alive forever
    while keep_running:
        time.sleep(1)
