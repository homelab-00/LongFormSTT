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
import glob
import webrtcvad

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
USE_SYSTEM_AUDIO = True  # Set to True to capture system audio via stereomix
INPUT_DEVICE_INDEX = 3    # Device index for stereomix (usually 2 on Windows 10)

MIN_CHUNK_LENGTH_SEC = 60   # Minimum length before we allow a chunk split
SILENCE_FRAMES_REQUIRED = 12  # Keep the chunk-splitting logic as before

# New parameters for trimming large silence blocks:
MIN_SILENCE_TRIM_SEC = 4.0   # If a silence block is >= 4s, we trim it down
KEEP_SILENCE_PAD_SEC = 2.0   # Keep 2 seconds at start/end of a large silence block

# Hallucination filtering (optional)
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

console.print(f"[bright_blue][DEBUG] Using device: {device} with model: {model_id}[/bright_blue]")

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

# Keep track of partial transcriptions
partial_transcripts = []
transcription_threads = []

current_chunk_index = 1
chunk_start_time = 0.0  # We'll set this when we open each new chunk

# PyAudio parameters
audio = pyaudio.PyAudio()
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024  # We'll read ~64ms chunks
chunks_per_second = RATE // CHUNK  # ~16 reads per second

# This buffer holds all audio data (including silence) for the current chunk
buffer = []

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

###############################################################################
# generate_trimmed_audio:
# - Reads the chunk WAV file
# - Runs webrtcvad to identify speech vs. silence blocks
# - If a silence block is >= 4s, keep only 2s at the start & 2s at the end
#   if both a preceding and following speech block exist.
# - If no preceding speech block, discard the front silence portion. If no
#   next speech block, discard the trailing silence portion.
# - If the entire chunk is silence, we skip it entirely (no transcription).
###############################################################################
def generate_trimmed_audio(input_wav: str, output_wav: str, vad_mode=1):
    """
    Reads the entire wave file, runs webrtcvad, segments into speech/silence,
    and trims large silence blocks according to your logic.
    Writes the trimmed audio to 'output_wav'.
    Returns True if we created a non-empty trimmed file, False otherwise.
    """
    console.print(f"[blue][DEBUG] generate_trimmed_audio() called with input: {input_wav}, output: {output_wav}[/blue]")

    # Open input wave
    try:
        wf_in = wave.open(input_wav, 'rb')
    except Exception as e:
        console.print(f"[bold red][ERROR] Failed to open {input_wav} for trimming: {e}[/bold red]")
        return False

    n_channels = wf_in.getnchannels()
    sampwidth = wf_in.getsampwidth()
    framerate = wf_in.getframerate()
    n_frames = wf_in.getnframes()

    console.print(f"[blue][DEBUG] Input WAV info: Channels={n_channels}, SampleWidth={sampwidth}, Rate={framerate}, TotalFrames={n_frames}[/blue]")

    audio_data = wf_in.readframes(n_frames)
    wf_in.close()

    # If not 16k/mono/16-bit, VAD can be off, but we'll proceed anyway
    if n_channels != 1 or framerate != 16000 or sampwidth != 2:
        console.print("[yellow]Warning: input wave is not 16-bit/16kHz mono. VAD results may be suboptimal.[/yellow]")

    vad = webrtcvad.Vad(vad_mode)

    frame_length = 640  # 20ms @ 16kHz, 16-bit mono => 640 bytes
    total_len = len(audio_data)
    offset = 0

    blocks = []  # will store (is_speech, data)
    current_block_type = None
    current_block_data = []

    # Chunk the raw data into 20ms frames
    while offset + frame_length <= total_len:
        frame = audio_data[offset: offset + frame_length]
        offset += frame_length

        is_speech = vad.is_speech(frame, 16000)
        if current_block_type is None:
            # First frame sets the block type
            current_block_type = is_speech
            current_block_data.append(frame)
        else:
            if is_speech == current_block_type:
                current_block_data.append(frame)
            else:
                blocks.append((current_block_type, b"".join(current_block_data)))
                current_block_type = is_speech
                current_block_data = [frame]

    # Handle remainder frames
    remainder = audio_data[offset:]
    if remainder:
        if current_block_type is None:
            pass  # means empty audio
        else:
            current_block_data.append(remainder)

    if current_block_data:
        blocks.append((current_block_type, b"".join(current_block_data)))

    console.print(f"[blue][DEBUG] Number of blocks identified: {len(blocks)}[/blue]")

    # Count how many speech blocks exist
    speech_blocks_count = sum(1 for (isp, dat) in blocks if isp and len(dat) > 0)
    console.print(f"[blue][DEBUG] Speech blocks count: {speech_blocks_count}[/blue]")
    if speech_blocks_count == 0:
        # Entire chunk is silence => skip
        console.print(f"[cyan]No speech in {os.path.basename(input_wav)}, skipping entire chunk[/cyan]")
        return False

    # We'll do a new list of blocks with trimmed silence
    def block_duration_sec(data_block: bytes) -> float:
        # 16-bit mono @16k => # of samples = len(data_block)//2
        # 16000 samples = 1 sec
        return (len(data_block) / 2.0) / 16000.0

    new_blocks = []

    for i, (is_sp, dat) in enumerate(blocks):
        dur = block_duration_sec(dat)
        console.print(f"[blue][DEBUG] Block {i}: is_speech={is_sp}, duration={dur:.2f}s[/blue]")

        if is_sp:
            # Keep speech block fully
            new_blocks.append((True, dat))
        else:
            # Silence block
            if dur < MIN_SILENCE_TRIM_SEC:
                # Keep entire short silence
                console.print(f"[green][DEBUG] Keeping short silence (duration {dur:.2f}s)[/green]")
                new_blocks.append((False, dat))
            else:
                # Silence >= 4s => keep 2s from front/back if there's preceding & next speech
                console.print(f"[yellow][DEBUG] Trimming large silence block (duration {dur:.2f}s)[/yellow]")
                has_prev_speech = False
                has_next_speech = False

                if len(new_blocks) > 0 and new_blocks[-1][0]:
                    has_prev_speech = True

                # look ahead for next speech
                for j in range(i + 1, len(blocks)):
                    if blocks[j][0]:
                        has_next_speech = True
                        break

                two_sec_bytes = int(16000 * 2 * 2)  # 2 seconds => 64000 bytes

                to_keep_front = b""
                to_keep_back = b""

                if has_prev_speech:
                    if len(dat) >= two_sec_bytes:
                        to_keep_front = dat[:two_sec_bytes]
                    else:
                        # if the block is just a bit over 4s, keep it entirely in front?
                        to_keep_front = dat

                if has_next_speech:
                    if len(dat) >= two_sec_bytes:
                        to_keep_back = dat[-two_sec_bytes:]
                    else:
                        to_keep_back = dat

                final_silence = to_keep_front
                if has_prev_speech and has_next_speech:
                    final_silence += to_keep_back
                elif has_next_speech and not has_prev_speech:
                    # We only keep the back portion
                    if not final_silence:
                        final_silence = to_keep_back

                if final_silence:
                    new_blocks.append((False, final_silence))

    # Check if we kept anything
    total_kept_bytes = sum(len(d) for (_, d) in new_blocks)
    if total_kept_bytes == 0:
        console.print(f"[cyan]All data trimmed from {os.path.basename(input_wav)}, skipping[/cyan]")
        return False

    # Write the new blocks to an output wave
    try:
        wf_out = wave.open(output_wav, 'wb')
        wf_out.setnchannels(n_channels)
        wf_out.setsampwidth(sampwidth)
        wf_out.setframerate(framerate)
        for (isp, block_data) in new_blocks:
            wf_out.writeframes(block_data)
        wf_out.close()
    except Exception as e:
        console.print(f"[bold red][ERROR] Failed to write trimmed audio to {output_wav}: {e}[/bold red]")
        return False

    console.print(
        f"[green]Trimmed audio written to {os.path.basename(output_wav)} "
        f"(kept {total_kept_bytes} bytes).[/green]"
    )
    return True


def cleanup_before_recording():
    """
    Whenever F3 is pressed, delete all temp_audio_file*.wav (including the plain one).
    Ensures a clean start each session.
    """
    console.print("[blue][DEBUG] cleanup_before_recording() called[/blue]")
    temp_files = glob.glob(os.path.join(script_dir, "temp_audio_file*.wav"))
    for f in temp_files:
        try:
            os.remove(f)
            console.print(f"[yellow]Deleted file: {os.path.basename(f)}[/yellow]")
        except Exception as e:
            console.print(f"[red]Failed to delete {os.path.basename(f)}: {e}[/red]")


def toggle_paste():
    global paste_enabled
    paste_enabled = not paste_enabled
    status = "enabled" if paste_enabled else "disabled"
    console.print(f"[italic green]Paste is now {status}.[/italic green]")


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
    global chunk_start_time
    global active_filename, active_wave_file, stream

    if recording:
        console.print("[bold yellow]Already recording![/bold yellow]")
        return

    console.print("[bold green]Starting a new recording session[/bold green]")
    console.print("[blue][DEBUG] start_recording() initiated[/blue]")

    cleanup_before_recording()

    # Reset state
    partial_transcripts.clear()
    transcription_threads.clear()
    buffer = []
    current_chunk_index = 1

    # Open mic stream
    try:
        stream_params = {
            'format': FORMAT,
            'channels': CHANNELS,
            'rate': RATE,
            'input': True,
            'frames_per_buffer': CHUNK,
            'input_device_index': INPUT_DEVICE_INDEX if USE_SYSTEM_AUDIO else None
        }
        console.print(f"[blue][DEBUG] Opening PyAudio stream with params: {stream_params}[/blue]")
        stream = audio.open(**stream_params)
    except Exception as e:
        console.print(f"[bold red][ERROR] Failed to open audio stream: {e}[/bold red]")
        return

    # Open first chunk file
    active_filename = os.path.join(
        script_dir, f"temp_audio_file{current_chunk_index}.wav"
    )
    try:
        active_wave_file = wave.open(active_filename, 'wb')
        active_wave_file.setnchannels(CHANNELS)
        active_wave_file.setsampwidth(audio.get_sample_size(FORMAT))
        active_wave_file.setframerate(RATE)
        console.print(f"[blue][DEBUG] Opened first chunk file: {active_filename}[/blue]")
    except Exception as e:
        console.print(f"[bold red][ERROR] Failed to open wave file: {e}[/bold red]")
        return

    # Mark the time we started this chunk
    chunk_start_time = time.time()
    console.print("[blue][DEBUG] chunk_start_time set[/blue]")

    recording = True
    recording_thread = threading.Thread(target=record_audio, daemon=True)
    recording_thread.start()


def record_audio():
    """
    Main recording loop with VAD:
      - Reads from mic
      - Writes all audio (including silence) to the current chunk file
      - Uses webrtcvad to detect speech frames only to see if we should split
        after MIN_CHUNK_LENGTH_SEC has passed.
    """
    global recording, active_wave_file, active_filename
    global partial_transcripts, transcription_threads
    global buffer, current_chunk_index
    global chunk_start_time
    global stream

    console.print("[blue][DEBUG] record_audio() thread started[/blue]")
    vad = webrtcvad.Vad(1)  # Low aggressiveness (0-3)
    leftover = b""
    chunk_count = 0
    silence_frame_count = 0

    try:
        while recording:
            data = stream.read(CHUNK, exception_on_overflow=False)
            buffer.append(data)

            combined = leftover + data
            offset = 0
            speech_detected_in_this_read = False

            # Break the read into 20ms frames for VAD
            while len(combined) - offset >= 640:  # 20ms
                frame = combined[offset:offset+640]
                offset += 640
                if vad.is_speech(frame, RATE):
                    speech_detected_in_this_read = True

            leftover = combined[offset:]

            chunk_count += 1
            if chunk_count >= chunks_per_second:
                if buffer:
                    active_wave_file.writeframes(b"".join(buffer))
                    buffer = []
                chunk_count = 0

            # --- CHUNK SPLIT LOGIC ---
            now = time.time()
            elapsed_in_chunk = now - chunk_start_time

            if elapsed_in_chunk >= MIN_CHUNK_LENGTH_SEC:
                if not speech_detected_in_this_read:
                    silence_frame_count += 1
                else:
                    silence_frame_count = 0

                console.print(f"[blue][DEBUG] Elapsed={elapsed_in_chunk:.2f}s, "
                              f"silentFrames={silence_frame_count}, "
                              f"speechDetected={speech_detected_in_this_read}[/blue]")

                if silence_frame_count >= SILENCE_FRAMES_REQUIRED:
                    console.print("[yellow][DEBUG] Splitting chunk due to silence[/yellow]")
                    split_current_chunk()
                    chunk_start_time = time.time()
                    current_chunk_index += 1
                    silence_frame_count = 0

        # Done => user pressed F4
        if buffer:
            active_wave_file.writeframes(b"".join(buffer))
            buffer = []
    except Exception as e:
        console.print(f"[bold red][ERROR] Recording error: {e}[/bold red]")
    finally:
        if active_wave_file:
            active_wave_file.close()
        if stream:
            stream.stop_stream()
            stream.close()

        recording = False
        console.print("[green]Recording stopped.[/green]")


def split_current_chunk():
    """
    Closes the current chunk file, transcribes it in a background thread,
    and opens a new file for the next chunk.
    """
    global active_wave_file, active_filename
    global transcription_threads, current_chunk_index

    console.print(f"[blue][DEBUG] split_current_chunk() called for chunk {current_chunk_index}[/blue]")
    if active_wave_file:
        active_wave_file.close()

    chunk_path = active_filename
    t = threading.Thread(target=partial_transcribe, args=(chunk_path,))
    t.start()
    transcription_threads.append(t)

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
        console.print(f"[blue][DEBUG] new filename: {new_filename}[/blue]")
    except Exception as e:
        console.print(f"[bold red][ERROR] Failed to open new chunk file {new_filename}: {e}[/bold red]")
        return

    active_wave_file = wf


def partial_transcribe(chunk_path):
    """
    Transcribe the given chunk after trimming large silence blocks, remove
    hallucinations, and store partial text. Print partial (but do NOT paste).
    """
    global partial_transcripts

    console.print(f"[blue][DEBUG] partial_transcribe() called for {chunk_path}[/blue]")

    # We'll first create a trimmed version of the chunk
    trimmed_path = chunk_path.replace(".wav", "_trimmed.wav")

    success = generate_trimmed_audio(chunk_path, trimmed_path, vad_mode=1)
    if not success:
        console.print(f"[blue][DEBUG] No audio to transcribe from {chunk_path} (possibly silent or error).[/blue]")
        return

    # Now transcribe the trimmed audio
    try:
        console.print(f"[blue][DEBUG] Transcribing trimmed audio {trimmed_path}[/blue]")
        segments, info = model.transcribe(
            trimmed_path,
            language=language,
            task=task,
            beam_size=10,
            best_of=10,
            temperature=0.0
        )
        text = "".join(s.text for s in segments)

        # Remove hallucinations
        for pattern in HALLUCINATIONS_REGEX:
            text = pattern.sub("", text)

        console.print(f"[cyan]Partial transcription of {os.path.basename(chunk_path)}[/cyan]")
        console.print(f"[bold magenta]{text}[/bold magenta]\n")
        partial_transcripts.append(text)

    except Exception as e:
        console.print(f"[bold red][ERROR] Partial transcription failed for {chunk_path}: {e}[/bold red]")


def stop_recording_and_transcribe():
    """
    Press F4 to stop recording, finalize last chunk, wait for partials,
    combine them, and print & paste the final text.
    """
    global recording, recording_thread, active_wave_file, active_filename
    global partial_transcripts, transcription_threads

    if not recording:
        console.print("[italic bold yellow]Recording[/italic bold yellow] [italic]not in progress[/italic]")
        return

    console.print("[bold blue]Stopping recording and transcribing...[/bold blue]")
    console.print("[blue][DEBUG] stop_recording_and_transcribe() initiated[/blue]")
    recording = False

    if recording_thread:
        recording_thread.join()

    # Transcribe the last chunk if it has data
    if active_filename and os.path.exists(active_filename):
        if os.path.getsize(active_filename) > 44:  # bigger than just WAV header
            final_chunk = active_filename
            console.print(f"[blue][DEBUG] Final chunk to transcribe: {final_chunk}[/blue]")
            t = threading.Thread(target=partial_transcribe, args=(final_chunk,))
            t.start()
            transcription_threads.append(t)
        else:
            console.print(f"[yellow][DEBUG] Last chunk file {active_filename} is too small to process[/yellow]")

    console.print("[blue]Waiting for partial transcriptions...[/blue]")
    for t in transcription_threads:
        t.join()

    # Combine all partial transcripts
    full_text = "".join(partial_transcripts)

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
    console.print("[blue][DEBUG] setup_hotkeys() called[/blue]")
    keyboard.add_hotkey('F2', toggle_paste, suppress=True)
    keyboard.add_hotkey('F3', start_recording, suppress=True)
    keyboard.add_hotkey('F4', stop_recording_and_transcribe, suppress=True)


def startup():
    console.print("[blue][DEBUG] startup() called[/blue]")
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


if __name__ == "__main__":
    startup()
    keyboard.wait()
