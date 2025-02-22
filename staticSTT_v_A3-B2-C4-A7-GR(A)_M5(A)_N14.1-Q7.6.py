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

# --------------------------------------------------------------------------------------
# Existing Functions (unchanged?)
# --------------------------------------------------------------------------------------

def is_ahk_script_running(script_name):
    """
    Returns True if there's an AutoHotkey.exe process running
    with 'script_name' in its command line.
    """
    for proc in psutil.process_iter(['name', 'cmdline']):
        try:
            # 1) Make sure cmdline is not None or empty
            cmdline = proc.info['cmdline']
            if not cmdline:  # could be None or an empty list
                continue
            
            # 2) Check process name
            if proc.info['name'] in ('AutoHotkeyU64.exe'):
                # 3) Join the cmdline into a single string to search
                cmdline_str = " ".join(cmdline)
                if script_name in cmdline_str:
                    return True
        
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    return False

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
    global partial_transcripts, transcription_threads
    global buffer, current_chunk_index
    global record_start_time, next_split_time, chunk_split_requested
    global active_filename, active_wave_file, stream

    if recording:
        console.print("[bold yellow]Already recording![/bold yellow]")
        return

    console.print("[bold green]Starting a new recording session[/bold green]")
    cleanup_before_recording()

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
    global partial_transcripts, transcription_threads
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

    if active_filename and os.path.exists(active_filename):
        if os.path.getsize(active_filename) > 44:
            final_chunk = active_filename

            # Use the current chunk index or whatever index you want for the final chunk
            final_chunk_idx = current_chunk_index

            # Run partial_transcribe with TWO arguments now
            t = threading.Thread(target=partial_transcribe, args=(final_chunk, final_chunk_idx))
            t.start()
            transcription_threads.append(t)

            # Optionally increment current_chunk_index to reserve the next number
            current_chunk_index += 1

    console.print("[blue]Waiting for partial transcriptions...[/blue]")

    for t in transcription_threads:
        t.join()

    ordered_texts = []
    for idx in sorted(partial_transcripts_dict.keys()):
        ordered_texts.append(partial_transcripts_dict[idx])
    full_text = "".join(ordered_texts)
    if len(full_text) > 0:
        full_text = full_text[1:]  # Remove the leading character (always a space)

    panel = Panel(
        f"[bold magenta]Final Combined Transcription:[/bold magenta] {full_text}",
        title="Transcription",
        border_style="yellow"
    )
    console.print(panel)

    pyperclip.copy(full_text)
    keyboard.send('ctrl+v')

    console.print("[italic green]Done.[/italic green]")

# --------------------------------------------------------------------------------------
# New: TCP Server for AHK -> Python
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

        if data == "F2":
            toggle_language()
        elif data == "F3":
            start_recording()
        elif data == "F4":
            stop_recording_and_transcribe()
        elif data == "QUIT":
            console.print("[bold red]Received QUIT command[/bold red]")
            graceful_exit(sock)

        conn.close()

def graceful_exit(sock):
    global keep_running
    console.print("[bold red]Shutting down gracefully...[/bold red]")
    keep_running = False
    sock.close()  # close the server socket

    # Attempt to close the AHK script if it's running
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        # On most systems, the AHK executable is named AutoHotkeyU64.exe
        if proc.info['name'] == 'AutoHotkeyU64.exe':
            # Join cmdline pieces into one string to search for the script name
            cmdline_str = " ".join(proc.info['cmdline'])
            if hotkey_script in cmdline_str:
                console.print(f"[bold red]Killing AHK script ({hotkey_script})...[/bold red]")
                proc.kill()

    sys.exit(0)   # immediately exit Python process


# --------------------------------------------------------------------------------------
# Startup (modded)
# --------------------------------------------------------------------------------------
def startup():
    panel_content = (
        f"[bold yellow]Model[/bold yellow]: {model_id}\n"
        "[bold yellow]Hotkeys[/bold yellow]: Controlled by AutoHotkey now.\n"
        "F2 -> toggle language | F3 -> start recording | F4 -> stop & transcribe\n"
        f"[bold yellow]Language[/bold yellow]: {language}"
    )
    panel = Panel(panel_content, title="Information", border_style="green")
    console.print(panel)

    # Launch the AHK script automatically and check if it's already running
    ahk_path = os.path.join(script_dir, hotkey_script)
    if not is_ahk_script_running("Hotkeys-AHK_A1"):
        console.print("[green]Launching AHK detached...[/green]")
        subprocess.Popen([ahk_path], creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP, shell=True)
    else:
        console.print("[yellow]AHK script already running. Skipping launch.[/yellow]")

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
