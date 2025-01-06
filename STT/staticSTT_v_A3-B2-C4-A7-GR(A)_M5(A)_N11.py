import win32con
import win32gui
import win32api
import torch
import time
import sys
import threading
import os
import wave
import audioop
import re
import glob
import pyperclip
import pyaudio
from rich.console import Console
from rich.panel import Panel
from faster_whisper import WhisperModel

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
THRESHOLD = 500              # Amplitude threshold for silence vs. voice
SILENCE_LIMIT_SEC = 1.5      # Keep up to 1.5 seconds of silence
CHUNK_SPLIT_INTERVAL = 60    # 1 minute in seconds

# Hallucination filtering with regex (optional)
HALLUCINATIONS_REGEX = [
    re.compile(r"\bΥπότιτλοι\s+AUTHORWAVE\b", re.IGNORECASE),
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

# Recording state
recording = False
recording_thread = None
stream = None
active_wave_file = None
active_filename = None

# Tracks partial transcriptions and partial transcription threads
partial_transcripts = []
transcription_threads = []

# For chunk logic
current_chunk_index = 1
record_start_time = 0
next_split_time = 0

# PyAudio parameters
audio = pyaudio.PyAudio()
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
chunks_per_second = RATE // CHUNK

# Audio buffer used in record loop
buffer = []

# Define hotkey IDs
HOTKEY_START = 1  # F3
HOTKEY_STOP = 2   # F4

# Initialize hwnd as a global variable
hwnd = None

# --------------------------------------------------------------------------------------
# Utility Functions
# --------------------------------------------------------------------------------------
def cleanup_before_recording():
    """
    Whenever recording starts, delete all temp_audio_file*.wav to ensure a clean session.
    """
    temp_files = glob.glob(os.path.join(script_dir, "temp_audio_file*.wav"))
    for f in temp_files:
        try:
            os.remove(f)
            console.print(f"[yellow][DEBUG] Deleted file: {os.path.basename(f)}[/yellow]")
        except Exception as e:
            console.print(f"[red][DEBUG] Failed to delete {os.path.basename(f)}: {e}[/red]")

def get_unique_filename(base_filepath):
    """
    Returns a unique filename by appending '_2', '_3', etc. if base_filepath already exists.
    """
    if not os.path.exists(base_filepath):
        return base_filepath

    base_dir = os.path.dirname(base_filepath)
    base_name = os.path.basename(base_filepath)
    name, ext = os.path.splitext(base_name)

    counter = 2
    while True:
        new_name = f"{name}_{counter}{ext}"
        new_path = os.path.join(base_dir, new_name)
        if not os.path.exists(new_path):
            return new_path
        counter += 1

# --------------------------------------------------------------------------------------
# Hotkey Handlers
# --------------------------------------------------------------------------------------
def toggle_paste():
    global paste_enabled
    paste_enabled = not paste_enabled
    status = "enabled" if paste_enabled else "disabled"
    console.print(f"[DEBUG] toggle_paste() called. Paste is now {status}.")

def start_recording():
    global recording, recording_thread
    global partial_transcripts, transcription_threads
    global buffer, current_chunk_index
    global record_start_time, next_split_time
    global active_filename, active_wave_file, stream

    if recording:
        console.print("[bold yellow][DEBUG] Already recording! Ignoring F3.[/bold yellow]")
        return

    console.print("[bold green]Starting a new recording session[/bold green]")

    # 1) Cleanup old files
    cleanup_before_recording()

    # 2) Reset local tracking
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

    console.print(f"[DEBUG] Opening mic stream. Target file: {first_file}")

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
        console.print(f"[bold red][DEBUG] Failed to open audio stream: {e}[/bold red]")
        recording = False
        return

    # Open wave file
    try:
        active_wave_file = wave.open(first_file, 'wb')
        active_wave_file.setnchannels(CHANNELS)
        active_wave_file.setsampwidth(audio.get_sample_size(FORMAT))
        active_wave_file.setframerate(RATE)
    except Exception as e:
        console.print(f"[bold red][DEBUG] Failed to open wave file: {e}[/bold red]")
        recording = False
        return

    console.print(f"[DEBUG] start_recording() -> Spawning record_audio thread.")
    recording_thread = threading.Thread(target=record_audio, daemon=True)
    recording_thread.start()

def record_audio():
    global recording, active_wave_file, active_filename, buffer
    global record_start_time, next_split_time, current_chunk_index
    global recording_thread

    console.print("[DEBUG] record_audio() thread started.")
    console.print(f"[DEBUG] record_audio() initial state: recording={recording}, thread={recording_thread}")

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
                if active_wave_file:
                    active_wave_file.writeframes(b''.join(buffer))
                buffer = []
                chunk_count = 0

            # Check if it's time to split
            now = time.time()
            if now >= next_split_time:
                console.print("[DEBUG] record_audio() -> It's time to split_current_chunk()")
                split_current_chunk()
                current_chunk_index += 1
                next_split_time += CHUNK_SPLIT_INTERVAL

        # End of while => user pressed F4 or something set recording=False
        console.print("[DEBUG] record_audio() -> Exiting while loop. recording=False")

        # Write leftover buffer
        if buffer and active_wave_file:
            active_wave_file.writeframes(b''.join(buffer))
            buffer = []

    except Exception as e:
        console.print(f"[bold red][DEBUG] Recording error: {e}[/bold red]")
    finally:
        # Close wave file, close stream
        console.print("[DEBUG] record_audio() -> finally block. Closing wave_file and stream.")
        if active_wave_file:
            active_wave_file.close()
        if stream:
            stream.stop_stream()
            stream.close()

        recording = False
        recording_thread = None
        console.print(f"[DEBUG] record_audio() finished. recording={recording}, thread={recording_thread}")

def split_current_chunk():
    global active_wave_file, active_filename, current_chunk_index
    global transcription_threads

    console.print(f"[DEBUG] split_current_chunk() called. Closing {active_filename}")

    # 1) Close the current wave file
    if active_wave_file:
        active_wave_file.close()

    chunk_path = active_filename
    console.print(f"[DEBUG] split_current_chunk() -> partial_transcribe on {chunk_path}")
    t = threading.Thread(target=partial_transcribe, args=(chunk_path,))
    t.start()
    transcription_threads.append(t)

    # 2) New chunk filename
    new_filename = os.path.join(
        script_dir, f"temp_audio_file{current_chunk_index + 1}.wav"
    )
    active_filename = new_filename

    console.print(f"[DEBUG] split_current_chunk() -> Opening new chunk {new_filename}")
    try:
        wf = wave.open(new_filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        console.print(f"[DEBUG] Successfully opened {new_filename}")
    except Exception as e:
        console.print(f"[bold red][DEBUG] Failed to open new chunk file {new_filename}: {e}[/bold red]")
        return

    active_wave_file = wf

def partial_transcribe(chunk_path):
    global partial_transcripts
    console.print(f"[DEBUG] partial_transcribe() called for {chunk_path}")
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
        console.print(f"[bold red][DEBUG] Partial transcription failed for {chunk_path}: {e}[/bold red]")

def stop_recording_and_transcribe():
    global recording, recording_thread, active_wave_file, active_filename
    global partial_transcripts, transcription_threads
    global hwnd  # Ensure hwnd is accessible

    console.print(f"[DEBUG] stop_recording_and_transcribe() called. recording={recording}, thread={recording_thread}")

    # If not recording, do nothing
    if not recording:
        console.print("[DEBUG] No recording in progress. Ignoring F4.")
        return

    console.print("[bold blue][DEBUG] Stopping recording and transcribing...[/bold blue]")
    recording = False

    if recording_thread:
        console.print("[DEBUG] Joining recording_thread...")
        recording_thread.join()
        console.print("[DEBUG] recording_thread joined. Setting recording_thread=None.")
        recording_thread = None

    # 2) If the last chunk has data, transcribe it
    if active_filename and os.path.exists(active_filename):
        size = os.path.getsize(active_filename)
        console.print(f"[DEBUG] Last chunk file {os.path.basename(active_filename)} size={size} bytes.")
        if size > 44:
            final_chunk = active_filename
            console.print(f"[DEBUG] Spawning partial_transcribe for final chunk {final_chunk}")
            t = threading.Thread(target=partial_transcribe, args=(final_chunk,))
            t.start()
            transcription_threads.append(t)
        else:
            console.print("[DEBUG] Final chunk is empty (<=44 bytes). Skipping partial_transcribe.")
    else:
        console.print("[DEBUG] active_filename does not exist or is None. No final chunk to transcribe.")

    console.print("[DEBUG] Waiting for all partial transcription threads to finish...")
    for t in transcription_threads:
        t.join()
    console.print("[DEBUG] All partial transcription threads joined.")

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
        console.print("[DEBUG] Pasting final text...")
        pyperclip.copy(full_text)
        # Simulate Ctrl+V using win32api
        win32api.keybd_event(win32con.VK_CONTROL, 0, 0, 0)
        win32api.keybd_event(win32con.VK_V, 0, 0, 0)
        win32api.keybd_event(win32con.VK_V, 0, win32con.KEYEVENTF_KEYUP, 0)
        win32api.keybd_event(win32con.VK_CONTROL, 0, win32con.KEYEVENTF_KEYUP, 0)

    console.print("[italic green]Done.[/italic green]")

    # 6) Hacky Solution: Send F4 to reset the state
    console.print("[DEBUG] Sending F4 to reset the state.")
    send_f4()

def send_f4():
    """
    Programmatically send the F4 key press to trigger stop_recording_and_transcribe again.
    """
    global hwnd
    if hwnd:
        # Simulate the F4 key press and release
        win32api.PostMessage(hwnd, win32con.WM_KEYDOWN, win32con.VK_F4, 0)
        win32api.PostMessage(hwnd, win32con.WM_KEYUP, win32con.VK_F4, 0)
    else:
        console.print("[red][DEBUG] hwnd is not defined. Cannot send F4.")

# --------------------------------------------------------------------------------------
# Window Procedure
# --------------------------------------------------------------------------------------
def wnd_proc(hwnd_window, msg, wparam, lparam):
    if msg == win32con.WM_HOTKEY:
        if wparam == HOTKEY_START:
            console.print("[DEBUG] F3 hotkey pressed.")
            start_recording()
        elif wparam == HOTKEY_STOP:
            console.print("[DEBUG] F4 hotkey pressed.")
            stop_recording_and_transcribe()
    elif msg == win32con.WM_DESTROY:
        win32gui.PostQuitMessage(0)
    else:
        return win32gui.DefWindowProc(hwnd_window, msg, wparam, lparam)
    return 0

# --------------------------------------------------------------------------------------
# Register Hotkeys and Create Window
# --------------------------------------------------------------------------------------
def register_hotkeys():
    # Register F3 as HOTKEY_START
    if not win32api.RegisterHotKey(None, HOTKEY_START, 0, win32con.VK_F3):
        console.print("[bold red][DEBUG] Failed to register F3 hotkey.[/bold red]")
    else:
        console.print("[DEBUG] Registered F3 hotkey.")

    # Register F4 as HOTKEY_STOP
    if not win32api.RegisterHotKey(None, HOTKEY_STOP, 0, win32con.VK_F4):
        console.print("[bold red][DEBUG] Failed to register F4 hotkey.[/bold red]")
    else:
        console.print("[DEBUG] Registered F4 hotkey.")

def unregister_hotkeys():
    win32api.UnregisterHotKey(None, HOTKEY_START)
    win32api.UnregisterHotKey(None, HOTKEY_STOP)
    console.print("[DEBUG] Unregistered all hotkeys.")

# --------------------------------------------------------------------------------------
# Create Message-Only Window
# --------------------------------------------------------------------------------------
def create_message_only_window():
    wc = win32gui.WNDCLASS()
    wc.lpfnWndProc = wnd_proc
    wc.lpszClassName = "MyHotkeyWindow"
    class_atom = win32gui.RegisterClass(wc)
    hwnd_window = win32gui.CreateWindowEx(
        0,
        class_atom,
        "Hotkey Window",
        0,
        0,
        0,
        0,
        0,
        win32con.HWND_MESSAGE,
        0,
        0,
        None
    )
    return hwnd_window

# --------------------------------------------------------------------------------------
# Cleanup on Exit
# --------------------------------------------------------------------------------------
def on_exit():
    global hwnd
    unregister_hotkeys()
    if hwnd:
        win32gui.DestroyWindow(hwnd)
    win32gui.UnregisterClass("MyHotkeyWindow", None)
    console.print("[DEBUG] Exiting script.")
    sys.exit()

# --------------------------------------------------------------------------------------
# Startup Function
# --------------------------------------------------------------------------------------
def startup():
    global hwnd
    register_hotkeys()
    hwnd = create_message_only_window()

    panel_content = (
        f"[bold yellow]Model[/bold yellow]: {model_id}\n"
        "[bold yellow]Hotkeys[/bold yellow]: "
        "[bold green]F2[/bold green] - Toggle typing | "
        "[bold green]F3[/bold green] - Start recording | "
        "[bold green]F4[/bold green] - Stop & Transcribe\n"
        "[bold yellow]Chunks[/bold yellow]: Every 1 min => temp_audio_file1.wav, temp_audio_file2.wav, etc.\n"
        "[bold yellow]Partial[/bold yellow]: Partial text for each chunk is printed\n"
        "[bold yellow]Final[/bold yellow]: Combined text is pasted after last chunk\n"
        "[bold yellow]Note[/bold yellow]: Cleanup is done each time you press F3, not mid-run"
    )
    panel = Panel(panel_content, title="Information", border_style="green")
    console.print(panel)

    if paste_enabled:
        console.print("[italic green]Typing is enabled on start.[/italic green]")

    console.print("[DEBUG] Startup complete. Waiting for hotkeys...")

    # Message Loop in a separate thread to keep the main thread free
    msg_thread = threading.Thread(target=message_loop, args=(hwnd,), daemon=True)
    msg_thread.start()

def message_loop(hwnd_window):
    while True:
        msg = win32gui.GetMessage(hwnd_window, 0, 0)
        win32gui.TranslateMessage(msg)
        win32gui.DispatchMessage(msg)

# --------------------------------------------------------------------------------------
# Main Execution
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        startup()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        on_exit()
    except Exception as e:
        console.print(f"[bold red][DEBUG] Unexpected error: {e}[/bold red]")
        on_exit()
