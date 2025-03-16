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

# We'll use pystray to create an icon in the system tray.
#   pip install pystray Pillow
try:
    import pystray
    from PIL import Image, ImageDraw
    TRAY_AVAILABLE = True
except ImportError:
    TRAY_AVAILABLE = False

# For file dialog (transcribe_static)
import tkinter
from tkinter import filedialog

# YAMNet-based filtering
#   pip install tensorflow tensorflow_hub soundfile
try:
    import tensorflow as tf
    import tensorflow_hub as hub
    import soundfile as sf
    YAMNET_AVAILABLE = True
except ImportError:
    YAMNET_AVAILABLE = False

import shutil
import math

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

# Recording-related
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

# AHK script
hotkey_script = "Hotkeys-AHK_A1.ahk"
ahk_pid = None
send_enter = True

# --------------------------------------------------------------------------------------
# System Tray Icon Logic
# --------------------------------------------------------------------------------------
tray_icon = None
GRAY_ICON = None
RED_ICON = None
BLUE_ICON = None
YELLOW_ICON = None

def create_circle_icon(size, fill_color, outline_color=(0,0,0)):
    img = Image.new("RGBA", (size, size), (0,0,0,0))
    draw = ImageDraw.Draw(img)
    margin = 2
    draw.ellipse(
        [(margin, margin), (size-margin, size-margin)],
        fill=fill_color,
        outline=outline_color,
        width=2
    )
    return img

def init_tray_icons():
    global tray_icon, GRAY_ICON, RED_ICON, BLUE_ICON, YELLOW_ICON
    if not TRAY_AVAILABLE:
        console.print("[red]pystray or Pillow not available. Tray icons won't be used.[/red]")
        return

    size = 24
    GRAY_ICON = create_circle_icon(size, (128,128,128,255), (0,0,0,255))
    RED_ICON = create_circle_icon(size, (255,0,0,255), (0,0,0,255))
    BLUE_ICON = create_circle_icon(size, (0,128,255,255), (0,0,0,255))
    YELLOW_ICON = create_circle_icon(size, (255,255,0,255), (0,0,0,255))

    tray_icon = pystray.Icon(
        "TranscriptionSTT",
        GRAY_ICON,
        "STT Script"
    )
    tray_icon.run_detached()

def set_tray_icon_color(color_name):
    if not TRAY_AVAILABLE or tray_icon is None:
        return
    if color_name == 'red':
        tray_icon.icon = RED_ICON
    elif color_name == 'blue':
        tray_icon.icon = BLUE_ICON
    elif color_name == 'yellow':
        tray_icon.icon = YELLOW_ICON
    else:
        tray_icon.icon = GRAY_ICON

def stop_tray_icon():
    global tray_icon
    if tray_icon is not None:
        tray_icon.stop()
        tray_icon = None

# --------------------------------------------------------------------------------------
# Basic STT / Recording
# --------------------------------------------------------------------------------------
def toggle_language():
    global language
    old_lang = language
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
    global buffer, current_chunk_index
    global partial_transcripts_dict, transcription_threads
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
            "format": FORMAT,
            "channels": CHANNELS,
            "rate": RATE,
            "input": True,
            "frames_per_buffer": CHUNK,
            "input_device_index": INPUT_DEVICE_INDEX if USE_SYSTEM_AUDIO else None,
        }
        stream = audio.open(**stream_params)
    except Exception as e:
        console.print(f"[bold red]Failed to open audio stream: {e}[/bold red]")
        recording = False
        return

    try:
        active_wave_file = wave.open(first_file, "wb")
        active_wave_file.setnchannels(CHANNELS)
        active_wave_file.setsampwidth(audio.get_sample_size(FORMAT))
        active_wave_file.setframerate(RATE)
    except Exception as e:
        console.print(f"[bold red]Failed to open wave file: {e}[/bold red]")
        recording = False
        return

    set_tray_icon_color("red")

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
            samples = struct.unpack("<{}h".format(len(data)//2), data)
            peak = max(abs(s) for s in samples)
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

            # flush once we have ~1s of audio
            if chunk_count >= chunks_per_second:
                active_wave_file.writeframes(b"".join(buffer))
                buffer = []
                chunk_count = 0

        # final flush
        if buffer:
            active_wave_file.writeframes(b"".join(buffer))
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
        wf = wave.open(new_filename, "wb")
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
    global transcription_threads, current_chunk_index

    if not recording:
        console.print("[italic bold yellow]Recording not in progress[/italic bold yellow]")
        return

    console.print("[bold blue]Stopping recording and transcribing...[/bold blue]")
    recording = False

    if recording_thread:
        recording_thread.join()

    if active_filename and os.path.exists(active_filename):
        console.print(f"[yellow]stop_recording_and_transcribe() -> final chunk path: {active_filename}, size={os.path.getsize(active_filename)} bytes[/yellow]")
    else:
        console.print(f"[red]stop_recording_and_transcribe() -> active_filename does NOT exist or is None[/red]")

    set_tray_icon_color("blue")

    if active_filename and os.path.exists(active_filename):
        if os.path.getsize(active_filename) > 44:
            final_chunk = active_filename
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
    if len(full_text) > 0 and full_text[0].isspace():
        full_text = full_text[1:]

    panel = Panel(
        f"[bold magenta]Final Combined Transcription:[/bold magenta] {full_text}",
        title="Transcription",
        border_style="yellow"
    )
    console.print(panel)

    pyperclip.copy(full_text)
    keyboard.send("ctrl+v")
    if send_enter:
        keyboard.send("enter")
        console.print("[yellow]Sent an ENTER keystroke after transcription.[/yellow]")

    console.print("[italic green]Done.[/italic green]")
    set_tray_icon_color("gray")

# --------------------------------------------------------------------------------------
# Cleanup
# --------------------------------------------------------------------------------------
def cleanup_static_temp():
    """
    Remove leftover static temp files
    """
    candidates = [
        os.path.join(script_dir, "temp_static_file.wav"),
        os.path.join(script_dir, "temp_static_voice_only.wav"),
    ]
    for f in candidates:
        if os.path.exists(f):
            try:
                os.remove(f)
                console.print(f"[yellow]Deleted previous temp file: {os.path.basename(f)}[/yellow]")
            except Exception as e:
                console.print(f"[red]Failed to delete {os.path.basename(f)}: {e}[/red]")

# --------------------------------------------------------------------------------------
# Convert or copy to 16k, mono
# --------------------------------------------------------------------------------------
def ensure_wav_format(input_path):
    temp_wav = os.path.join(script_dir, "temp_static_file.wav")
    try:
        with wave.open(input_path, "rb") as wf:
            channels = wf.getnchannels()
            rate = wf.getframerate()
            if channels == 1 and rate == 16000:
                console.print("[blue]No conversion needed, but copying to temp_static_file.wav.[/blue]")
                shutil.copy(input_path, temp_wav)
                return temp_wav
    except wave.Error:
        pass

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
        console.print("[cyan]Conversion successful to temp_static_file.wav.[/cyan]")
        return temp_wav
    except Exception as e:
        console.print(f"[bold red]FFmpeg conversion error: {e}[/bold red]")
        return None

# --------------------------------------------------------------------------------------
# YAMNet-based music removal
# --------------------------------------------------------------------------------------
def apply_yamnet_filter(in_wav_path, speech_threshold=0.3):
    """
    Load YAMNet from TF Hub, classify each ~0.96s frame.
    Keep frames where 'Speech' confidence >= speech_threshold.
    Return path to 'temp_static_voice_only.wav'.
    """
    if not YAMNET_AVAILABLE:
        console.print("[red]YAMNet not installed. Skipping music removal.[/red]")
        return in_wav_path

    out_wav = os.path.join(script_dir, "temp_static_voice_only.wav")

    try:
        # 1) Load audio as 1D float32
        data, sr = sf.read(in_wav_path)
        if sr != 16000:
            console.print("[red]YAMNet requires 16k sample rate. Skipping music removal.[/red]")
            return in_wav_path

        # 2) Load YAMNet
        console.print("[yellow]Loading YAMNet from TF Hub...[/yellow]")
        yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

        # 3) YAMNet expects shape=(samples,), float32
        waveform = tf.constant(data, tf.float32)  # shape => (samples,)

        # 4) Inference
        #   returns: scores: (num_frames, 521), embeddings, spectrogram
        scores, embeddings, spectrogram = yamnet_model(waveform)

        scores = scores.numpy()  # shape => (num_frames, 521)

        # 'Speech' is typically index 0 in the YAMNet label map
        speech_index = 0
        keep_mask = scores[:, speech_index] >= speech_threshold  # boolean (num_frames,)

        if not any(keep_mask):
            console.print("[red]No speech frames found by YAMNet. Returning original.[/red]")
            return in_wav_path

        # Each YAMNet frame is ~0.96 sec for 16k input
        frame_size = int(0.96 * sr)
        total_samples = len(data)
        n_frames = scores.shape[0]

        result_chunks = []
        for i in range(n_frames):
            start_i = i * frame_size
            end_i = min((i+1)*frame_size, total_samples)
            if start_i >= total_samples:
                break
            if keep_mask[i]:
                chunk = data[start_i:end_i]
                result_chunks.append(chunk)

        if not result_chunks:
            console.print("[red]All frames filtered out. Returning original.[/red]")
            return in_wav_path

        voice_data = tf.concat(
            [tf.constant(ch, dtype=tf.float32) for ch in result_chunks], axis=0
        ).numpy()

        sf.write(out_wav, voice_data, sr)
        console.print("[yellow]YAMNet filtering done => saved temp_static_voice_only.wav[/yellow]")
        return out_wav

    except Exception as e:
        console.print(f"[red]YAMNet filtering error: {e}[/red]")
        return in_wav_path

# --------------------------------------------------------------------------------------
# Static transcription
# --------------------------------------------------------------------------------------
def transcribe_static_file():
    console.print("[bold yellow]TRANSCRIBE_STATIC command received[/bold yellow]")
    set_tray_icon_color("yellow")

    cleanup_static_temp()
    root = tkinter.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select an audio file to transcribe",
        filetypes=[("Audio Files", "*.*")]
    )
    root.destroy()

    if not file_path:
        console.print("[red]No file selected. Aborting static transcription.[/red]")
        set_tray_icon_color("gray")
        return

    console.print(f"[green]Selected file: {file_path}[/green]")

    wav_path = ensure_wav_format(file_path)
    if not wav_path or not os.path.exists(wav_path):
        console.print("[bold red]Could not obtain a valid temp_static_file.wav. Aborting.[/bold red]")
        set_tray_icon_color("gray")
        return

    voice_wav = apply_yamnet_filter(wav_path, speech_threshold=0.3)

    try:
        console.print("[blue]Beginning static transcription with YAMNet speech-only data...[/blue]")
        segments, info = model.transcribe(voice_wav, language=language, task=task)
        final_text = "".join(s.text for s in segments)
        for pattern in HALLUCINATIONS_REGEX:
            final_text = pattern.sub("", final_text)
        if len(final_text) > 0 and final_text[0].isspace():
            final_text = final_text[1:]

        panel = Panel(
            f"[bold magenta]Static File Transcription:[/bold magenta] {final_text}",
            title="Static Transcription",
            border_style="yellow"
        )
        console.print(panel)

        # Save .txt next to original
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        dir_name = os.path.dirname(file_path)
        out_txt_path = os.path.join(dir_name, base_name + ".txt")
        with open(out_txt_path, "w", encoding="utf-8") as f:
            f.write(final_text)

        console.print(f"[green]Saved transcription to: {out_txt_path}[/green]")

    except Exception as e:
        console.print(f"[bold red]Static transcription failed: {e}[/bold red]")
    finally:
        set_tray_icon_color("gray")

# --------------------------------------------------------------------------------------
# TCP / AHK integration
# --------------------------------------------------------------------------------------
def run_socket_server():
    host = '127.0.0.1'
    port = 34909
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((host, port))
    sock.listen(5)
    sock.settimeout(1)

    console.print(f"[bold yellow]TCP server listening on {host}:{port}[/bold yellow]")

    while True:
        if not keep_running:
            break
        try:
            conn, addr = sock.accept()
        except socket.timeout:
            continue

        data = conn.recv(1024).decode("utf-8").strip()
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
        elif data == "TRANSCRIBE_STATIC":
            transcribe_static_file()
        elif data == "QUIT":
            console.print("[bold red]Received QUIT command[/bold red]")
            graceful_exit(sock)

        conn.close()

def graceful_exit(sock):
    global keep_running, ahk_pid
    console.print("[bold red]Shutting down gracefully...[/bold red]")
    keep_running = False
    sock.close()

    if ahk_pid is not None:
        console.print(f"[bold red]Killing AHK script by stored PID={ahk_pid}")
        try:
            psutil.Process(ahk_pid).kill()
        except Exception as e:
            console.print(f"[red]Failed to kill AHK process with PID {ahk_pid}: {e}[/red]")
    else:
        console.print("[yellow]No stored AHK PID. Skipping kill.[/yellow]")

    set_tray_icon_color("gray")
    stop_tray_icon()
    sys.exit(0)

def kill_leftover_ahk():
    for proc in psutil.process_iter(['pid','name','cmdline']):
        try:
            if proc.info['name'] == 'AutoHotkeyU64.exe' and hotkey_script in ' '.join(proc.info['cmdline'] or []):
                console.print(f"[yellow]Killing leftover AHK process with PID={proc.pid}[/yellow]")
                psutil.Process(proc.pid).kill()
        except:
            pass

def startup():
    global ahk_pid

    panel_content = (
        f"[bold yellow]Model[/bold yellow]: {model_id}\n"
        f"[bold yellow]Hotkeys[/bold yellow]: Controlled by AutoHotKey script '{hotkey_script}'\n"
        " F2  -> toggle language\n"
        " F3  -> start recording\n"
        " F4  -> stop & transcribe\n"
        " F5  -> toggle enter\n"
        " F6  -> quit\n"
        " F10 -> static file transcription\n"
        f"[bold yellow]Language[/bold yellow]: {language}"
    )
    panel = Panel(panel_content, title="Information", border_style="green")
    console.print(panel)

    kill_leftover_ahk()

    init_tray_icons()
    set_tray_icon_color("gray")

    pre_pids = set()
    for proc in psutil.process_iter(['pid','name','cmdline']):
        try:
            if proc.info['name'] == 'AutoHotkeyU64.exe':
                pre_pids.add(proc.info['pid'])
        except:
            pass

    ahk_path = os.path.join(script_dir, hotkey_script)
    console.print("[green]Launching AHK script...[/green]")
    subprocess.Popen(
        [ahk_path],
        creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
        shell=True
    )

    time.sleep(1.0)
    post_pids = set()
    for proc in psutil.process_iter(['pid','name','cmdline']):
        try:
            if proc.info['name'] == 'AutoHotkeyU64.exe':
                post_pids.add(proc.info['pid'])
        except:
            pass

    new_pids = post_pids - pre_pids
    if len(new_pids) == 1:
        ahk_pid = new_pids.pop()
        console.print(f"[green]Detected new AHK script PID: {ahk_pid}[/green]")
    else:
        console.print("[red]Could not detect a single new AHK script PID. No PID stored.[/red]")
        ahk_pid = None

if __name__ == "__main__":
    startup()

    server_thread = threading.Thread(target=run_socket_server, daemon=True)
    server_thread.start()

    while keep_running:
        time.sleep(1)
