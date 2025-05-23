# Speech to Text with Faster Whisper
# - Handles both real-time transcription and static file transcription
# - Supports language toggling and voice activity detection
# - Includes music detection and removal for static file transcription

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
import shutil
import tkinter
from tkinter import filedialog
from dataclasses import dataclass
from typing import Optional, List, Dict, Set, Union, Tuple

# Optional dependencies with graceful fallbacks
try:
    import pystray
    from PIL import Image, ImageDraw
    TRAY_AVAILABLE = True
except ImportError:
    TRAY_AVAILABLE = False

try:
    import webrtcvad
    WEBRTC_VAD_AVAILABLE = True
except ImportError:
    WEBRTC_VAD_AVAILABLE = False

# Check for music detection libraries
try:
    import librosa
    import numpy as np
    from scipy.io import wavfile
    from scipy.ndimage import binary_dilation
    MUSIC_DETECTION_AVAILABLE = True
except ImportError:
    MUSIC_DETECTION_AVAILABLE = False

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
@dataclass
class Config:
    # Audio settings
    use_system_audio: bool = False
    input_device_index: int = 2
    format: int = pyaudio.paInt16
    channels: int = 1
    rate: int = 16000
    chunk: int = 1024
    
    # Detection settings
    threshold: int = 500
    silence_limit_sec: float = 1.5
    chunk_split_interval: int = 60
    
    # Transcription settings
    language: str = "el"
    task: str = "transcribe"
    send_enter: bool = False  # Changed from True to False as requested
    
    # System settings
    hotkey_script: str = "Hotkeys-AHK_A1.ahk"
    
    # Derived properties
    @property
    def chunks_per_second(self) -> int:
        return self.rate // self.chunk
    
    # Hallucination filters
    @property
    def hallucinations_regex(self) -> List[re.Pattern]:
        return [
            re.compile(r"\bΥπότιτλοι\s+AUTHORWAVE\b[^\w]*", re.IGNORECASE),
            re.compile(r"\bΣας\s+ευχαριστώ\b[^\w]*", re.IGNORECASE),
        ]

# --------------------------------------------------------------------------------------
# System Tray Icon Manager
# --------------------------------------------------------------------------------------
class TrayManager:
    def __init__(self, console: Console):
        self.console = console
        self.tray_icon = None
        self.icons = {}        # Normal icons (black outline)
        self.icons_green = {}  # Icons with green outline when send_enter is True
        
        if TRAY_AVAILABLE:
            self._init_icons()
        else:
            self.console.print("[red]pystray or Pillow not available. Tray icons won't be used.[/red]")
    
    def _create_circle_icon(self, size: int, fill_color: Tuple[int, int, int, int], 
                           outline_color: Tuple[int, int, int, int] = (0, 0, 0, 255)) -> Image.Image:
        """Create a circular icon with the specified colors."""
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        margin = 2
        draw.ellipse(
            [(margin, margin), (size-margin, size-margin)],
            fill=fill_color,
            outline=outline_color,
            width=2
        )
        return img
    
    def _init_icons(self) -> None:
        """Initialize all tray icons with both normal and green outlines."""
        size = 24
        black_outline = (0, 0, 0, 255)
        green_outline = (0, 255, 0, 255)
        
        # Create icons with black outline (send_enter=False)
        self.icons = {
            'gray': self._create_circle_icon(size, (128, 128, 128, 255), black_outline),
            'red': self._create_circle_icon(size, (255, 0, 0, 255), black_outline),
            'blue': self._create_circle_icon(size, (0, 128, 255, 255), black_outline),
            'yellow': self._create_circle_icon(size, (255, 255, 0, 255), black_outline)
        }
        
        # Create icons with green outline (send_enter=True)
        self.icons_green = {
            'gray': self._create_circle_icon(size, (128, 128, 128, 255), green_outline),
            'red': self._create_circle_icon(size, (255, 0, 0, 255), green_outline),
            'blue': self._create_circle_icon(size, (0, 128, 255, 255), green_outline),
            'yellow': self._create_circle_icon(size, (255, 255, 0, 255), green_outline)
        }
        
        self.tray_icon = pystray.Icon(
            "TranscriptionSTT",
            self.icons['gray'],
            "STT Script",
        )
        self.tray_icon.run_detached()
    
    def set_color(self, color_name: str, send_enter: bool = False) -> None:
        """Set the tray icon color and outline based on send_enter state."""
        if not TRAY_AVAILABLE or self.tray_icon is None:
            return
        
        if color_name in self.icons:
            icon_set = self.icons_green if send_enter else self.icons
            self.tray_icon.icon = icon_set[color_name]
    
    def stop(self) -> None:
        """Stop the tray icon."""
        if self.tray_icon is not None:
            self.tray_icon.stop()
            self.tray_icon = None

# --------------------------------------------------------------------------------------
# Transcription Engine
# --------------------------------------------------------------------------------------
class Transcriber:
    def __init__(self, config: Config, console: Console):
        self.config = config
        self.console = console
        
        # Initialize the model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = "Systran/faster-whisper-large-v3"
        self.model = WhisperModel(
            self.model_id, 
            device=self.device, 
            compute_type="float16" if self.device == "cuda" else "float32"
        )
    
    def toggle_language(self) -> None:
        """Toggle between Greek and English."""
        old_lang = self.config.language
        self.config.language = "en" if self.config.language == "el" else "el"
        self.console.print(f"[yellow]Language toggled from {old_lang} to {self.config.language}[/yellow]")
    
    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file and clean up the result."""
        try:
            segments, info = self.model.transcribe(
                audio_path, 
                language=self.config.language, 
                task=self.config.task
            )
            
            # Combine segments and clean up
            text = "".join(s.text for s in segments)
            
            # Remove known hallucinations
            for pattern in self.config.hallucinations_regex:
                text = pattern.sub("", text)
            
            # Clean up leading whitespace
            if text and text[0].isspace():
                text = text[1:]
                
            return text
        except Exception as e:
            self.console.print(f"[bold red]Transcription failed for {audio_path}: {e}[/bold red]")
            return ""

# --------------------------------------------------------------------------------------
# Audio Recorder
# --------------------------------------------------------------------------------------
class AudioRecorder:
    def __init__(self, config: Config, console: Console, transcriber: Transcriber, tray: TrayManager):
        self.config = config
        self.console = console
        self.transcriber = transcriber
        self.tray = tray
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        
        # Recording state
        self.recording = False
        self.recording_thread = None
        self.stream = None
        self.active_wave_file = None
        self.active_filename = None
        
        # Chunking state
        self.current_chunk_index = 1
        self.record_start_time = 0
        self.next_split_time = 0
        self.chunk_split_requested = False
        
        # Buffers and results
        self.buffer = []
        self.partial_transcripts = {}
        self.transcription_threads = []
    
    def _cleanup_temp_files(self) -> None:
        """Remove any temporary audio files from previous recordings."""
        temp_files = glob.glob(os.path.join(self.script_dir, "temp_audio_file*.wav"))
        for f in temp_files:
            try:
                os.remove(f)
                self.console.print(f"[yellow]Deleted file: {os.path.basename(f)}[/yellow]")
            except Exception as e:
                self.console.print(f"[red]Failed to delete {os.path.basename(f)}: {e}[/red]")
    
    def start(self) -> None:
        """Start recording audio from the microphone."""
        if self.recording:
            self.console.print("[bold yellow]Already recording![/bold yellow]")
            return

        self.console.print("[bold green]Starting a new recording session[/bold green]")
        self._cleanup_temp_files()

        # Reset state
        self.partial_transcripts.clear()
        self.transcription_threads.clear()
        self.buffer.clear()
        self.current_chunk_index = 1

        # Initialize timing for chunking
        self.record_start_time = time.time()
        self.next_split_time = self.record_start_time + self.config.chunk_split_interval
        self.chunk_split_requested = False

        # Set up recording
        self.recording = True
        first_file = os.path.join(self.script_dir, f"temp_audio_file{self.current_chunk_index}.wav")
        self.active_filename = first_file

        try:
            # Open audio stream
            stream_params = {
                'format': self.config.format,
                'channels': self.config.channels,
                'rate': self.config.rate,
                'input': True,
                'frames_per_buffer': self.config.chunk,
                'input_device_index': self.config.input_device_index if self.config.use_system_audio else None
            }
            self.stream = self.audio.open(**stream_params)
            
            # Open wave file for saving
            self.active_wave_file = wave.open(first_file, 'wb')
            self.active_wave_file.setnchannels(self.config.channels)
            self.active_wave_file.setsampwidth(self.audio.get_sample_size(self.config.format))
            self.active_wave_file.setframerate(self.config.rate)
            
            # Update tray icon and start recording thread
            self.tray.set_color('red', self.config.send_enter)
            self.recording_thread = threading.Thread(target=self._record_loop, daemon=True)
            self.recording_thread.start()
            
        except Exception as e:
            self.console.print(f"[bold red]Failed to start recording: {e}[/bold red]")
            self.recording = False
            self._cleanup_resources()
    
    def _record_loop(self) -> None:
        """Main recording loop that captures audio and handles chunking."""
        chunk_count = 0
        silence_duration = 0.0

        try:
            while self.recording:
                # Read audio data
                data = self.stream.read(self.config.chunk, exception_on_overflow=False)
                samples = struct.unpack(f'<{len(data)//2}h', data)
                peak = max(abs(sample) for sample in samples)
                
                # Calculate timing
                chunk_time = float(self.config.chunk) / self.config.rate
                now = time.time()
                elapsed = now - self.record_start_time

                # Check if we need to request a chunk split
                if (not self.chunk_split_requested) and (elapsed >= (self.next_split_time - self.record_start_time)):
                    self.console.print(f"[yellow]Reached {int(elapsed)} seconds. Will split on next silence.[/yellow]")
                    self.chunk_split_requested = True

                # Handle silence detection
                if peak < self.config.threshold:
                    silence_duration += chunk_time
                    
                    # Still add data during brief silences
                    if silence_duration <= self.config.silence_limit_sec:
                        self.buffer.append(data)
                        chunk_count += 1

                    # If splitting was requested and we have some silence, do the split
                    if self.chunk_split_requested and (silence_duration >= 0.1):
                        self.console.print("[bold green]Splitting now at silence...[/bold green]")
                        self._split_chunk()
                        self.next_split_time += self.config.chunk_split_interval
                        self.chunk_split_requested = False
                else:
                    # Reset silence counter when we detect sound
                    silence_duration = 0.0
                    self.buffer.append(data)
                    chunk_count += 1

                # Periodically flush buffer to file
                if chunk_count >= self.config.chunks_per_second:
                    self._flush_buffer()
                    chunk_count = 0

            # Final buffer flush when stopping
            self._flush_buffer()
            
        except Exception as e:
            self.console.print(f"[bold red]Recording error: {e}[/bold red]")
        finally:
            self._cleanup_resources()
            self.recording = False
            self.recording_thread = None
            self.console.print("[green]Recording stopped.[/green]")
    
    def _flush_buffer(self) -> None:
        """Write buffered audio data to the active wave file."""
        if self.buffer and self.active_wave_file:
            self.active_wave_file.writeframes(b''.join(self.buffer))
            self.buffer.clear()
    
    def _cleanup_resources(self) -> None:
        """Clean up audio resources."""
        if hasattr(self, 'active_wave_file') and self.active_wave_file:
            self.active_wave_file.close()
            self.active_wave_file = None
            
        if hasattr(self, 'stream') and self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
    
    def _split_chunk(self) -> None:
        """Split the current audio chunk and start transcribing it."""
        if self.active_wave_file:
            self.active_wave_file.close()
            self.active_wave_file = None

        chunk_path = self.active_filename
        
        # Log file info for debugging
        if os.path.exists(chunk_path):
            self.console.print(f"[yellow]split_chunk() -> file: {chunk_path}, size={os.path.getsize(chunk_path)} bytes[/yellow]")
        else:
            self.console.print(f"[red]split_chunk() -> file: {chunk_path} does NOT exist[/red]")

        # Start transcription in a separate thread
        t = threading.Thread(
            target=self._transcribe_chunk, 
            args=(chunk_path, self.current_chunk_index)
        )
        t.start()
        self.transcription_threads.append(t)

        # Prepare for the next chunk
        self.current_chunk_index += 1
        new_filename = os.path.join(self.script_dir, f"temp_audio_file{self.current_chunk_index}.wav")
        self.active_filename = new_filename

        try:
            # Create a new wave file for the next chunk
            self.active_wave_file = wave.open(new_filename, 'wb')
            self.active_wave_file.setnchannels(self.config.channels)
            self.active_wave_file.setsampwidth(self.audio.get_sample_size(self.config.format))
            self.active_wave_file.setframerate(self.config.rate)
            self.console.print(f"[green]Opened new chunk file: {os.path.basename(new_filename)}[/green]")
        except Exception as e:
            self.console.print(f"[bold red]Failed to open new chunk file {new_filename}: {e}[/bold red]")
    
    def _transcribe_chunk(self, chunk_path: str, chunk_idx: int) -> None:
        """Transcribe a single audio chunk."""
        text = self.transcriber.transcribe(chunk_path)
        
        self.console.print(f"[cyan]Partial transcription of {os.path.basename(chunk_path)}[/cyan]")
        self.console.print(f"[bold magenta]{text}[/bold magenta]\n")

        self.partial_transcripts[chunk_idx] = text
    
    def stop_and_transcribe(self) -> None:
        """Stop recording and transcribe all chunks."""
        if not self.recording:
            self.console.print("[italic bold yellow]Recording not in progress[/italic bold yellow]")
            return

        self.console.print("[bold blue]Stopping recording and transcribing...[/bold blue]")
        self.recording = False

        # Wait for recording thread to finish
        if self.recording_thread:
            self.recording_thread.join()

        # Process the final chunk if it exists and has content
        if self.active_filename and os.path.exists(self.active_filename):
            self.console.print(f"[yellow]Final chunk: {self.active_filename}, size={os.path.getsize(self.active_filename)} bytes[/yellow]")
            
            # WAV header is 44 bytes, so check if there's actual audio data
            if os.path.getsize(self.active_filename) > 44:
                final_chunk_idx = self.current_chunk_index
                t = threading.Thread(
                    target=self._transcribe_chunk, 
                    args=(self.active_filename, final_chunk_idx)
                )
                t.start()
                self.transcription_threads.append(t)
                self.current_chunk_index += 1

        # Update tray icon
        self.tray.set_color('blue', self.config.send_enter)
        
        # Wait for all transcription threads to complete
        self.console.print("[blue]Waiting for partial transcriptions...[/blue]")
        for t in self.transcription_threads:
            t.join()

        # Combine all transcriptions in order
        ordered_texts = []
        for idx in sorted(self.partial_transcripts.keys()):
            ordered_texts.append(self.partial_transcripts[idx])
        
        full_text = "".join(ordered_texts)
        if full_text and full_text[0].isspace():
            full_text = full_text[1:]

        # Display the result
        panel = Panel(
            f"[bold magenta]Final Combined Transcription:[/bold magenta] {full_text}",
            title="Transcription",
            border_style="yellow"
        )
        self.console.print(panel)

        # Copy to clipboard and paste
        pyperclip.copy(full_text)
        keyboard.send('ctrl+v')
        if self.config.send_enter:
            keyboard.send('enter')
            self.console.print("[yellow]Sent an ENTER keystroke after transcription.[/yellow]")

        self.console.print("[italic green]Done.[/italic green]")
        self.tray.set_color('gray', self.config.send_enter)

# --------------------------------------------------------------------------------------
# Static File Processor
# --------------------------------------------------------------------------------------
class StaticFileProcessor:
    def __init__(self, config: Config, console: Console, transcriber: Transcriber, tray: TrayManager):
        self.config = config
        self.console = console
        self.transcriber = transcriber
        self.tray = tray
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
    
    def _cleanup_temp_files(self) -> None:
        """Remove temporary files used for static transcription."""
        temp_files = [
            os.path.join(self.script_dir, "temp_static_file.wav"),
            os.path.join(self.script_dir, "temp_static_silence_removed.wav"),
            os.path.join(self.script_dir, "temp_static_music_removed.wav")  # Added new temp file
        ]
        
        for f in temp_files:
            if os.path.exists(f):
                try:
                    os.remove(f)
                    self.console.print(f"[yellow]Deleted temp file: {os.path.basename(f)}[/yellow]")
                except Exception as e:
                    self.console.print(f"[red]Failed to delete {os.path.basename(f)}: {e}[/red]")
    
    def _ensure_wav_format(self, input_path: str) -> Optional[str]:
        """Convert input file to 16kHz mono WAV if needed."""
        temp_wav = os.path.join(self.script_dir, "temp_static_file.wav")
        
        # Check if the file is already in the correct format
        try:
            with wave.open(input_path, 'rb') as wf:
                channels = wf.getnchannels()
                rate = wf.getframerate()
                if channels == 1 and rate == 16000:
                    self.console.print("[blue]No conversion needed, copying to temp file.[/blue]")
                    shutil.copy(input_path, temp_wav)
                    return temp_wav
        except wave.Error:
            pass  # Not a valid WAV file, needs conversion

        # Convert using FFmpeg
        self.console.print(f"[cyan]Converting file '{input_path}' to 16kHz mono WAV[/cyan]")
        try:
            subprocess.run([
                "ffmpeg",
                "-y",  # Overwrite output file if it exists
                "-i", input_path,
                "-ac", "1",  # Mono
                "-ar", "16000",  # 16kHz
                temp_wav
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.console.print("[cyan]Conversion successful.[/cyan]")
            return temp_wav
        except Exception as e:
            self.console.print(f"[bold red]FFmpeg conversion error: {e}[/bold red]")
            return None
    
    def _remove_music_segments(self, in_wav_path: str) -> str:
        """Process an audio file to detect and remove music segments using spectral analysis."""
        out_wav = os.path.join(self.script_dir, "temp_static_music_removed.wav")
        
        if not MUSIC_DETECTION_AVAILABLE:
            self.console.print("[yellow]Music detection requires librosa, numpy, and scipy. Skipping...[/yellow]")
            return in_wav_path
        
        try:
            self.console.print("[cyan]Analyzing audio to detect and remove music segments...[/cyan]")
            
            # Load audio file
            y, sr = librosa.load(in_wav_path, sr=None)
            
            # Process in windows (e.g., 1-second segments)
            window_size = sr  # 1 second
            hop_length = sr // 2  # 50% overlap
            
            # Features we'll track
            is_music = np.zeros_like(y, dtype=bool)
            
            # Process each window
            for i in range(0, len(y) - window_size, hop_length):
                window = y[i:i+window_size]
                
                # Extract key features that differentiate music from speech
                spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=window, sr=sr)[0])
                spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=window, sr=sr)[0])
                spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=window)[0])
                zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(window)[0])
                
                # Rhythm regularity (more regular in music)
                onset_env = librosa.onset.onset_strength(y=window, sr=sr)
                pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sr)
                pulse_clarity = np.mean(pulse)
                
                # Calculate music probability
                music_probability = (
                    0.3 * (spectral_flatness > 0.05) +           # Music tends to have higher flatness
                    0.3 * (pulse_clarity > 0.1) +                # Music has clearer pulse
                    0.2 * (spectral_centroid > 2000) +           # Music often has higher centroids
                    0.2 * (zero_crossing_rate < 0.1)             # Speech has more zero crossings
                )
                
                # Mark segment as music if probability exceeds threshold
                if music_probability > 0.6:
                    is_music[i:i+window_size] = True
            
            # Create mask (non-music segments)
            speech_mask = ~is_music
            
            # Apply smoothing to avoid choppy transitions
            speech_mask = binary_dilation(speech_mask, iterations=sr//100)
            
            # Keep only speech segments
            filtered_audio = y.copy()
            filtered_audio[~speech_mask] = 0
            
            # Count how much was removed
            music_duration = np.sum(is_music) / sr
            total_duration = len(y) / sr
            music_percentage = (music_duration / total_duration) * 100
            
            self.console.print(f"[green]Removed music segments: {music_percentage:.1f}% of audio[/green]")
            
            # Write to output file
            wavfile.write(out_wav, sr, (filtered_audio * 32767).astype(np.int16))
            return out_wav
            
        except Exception as e:
            self.console.print(f"[red]Music removal error: {e}[/red]")
            return in_wav_path
    
    def _apply_vad(self, in_wav_path: str, aggressiveness: int = 2) -> str:
        """Apply Voice Activity Detection to keep only speech frames."""
        if not WEBRTC_VAD_AVAILABLE:
            self.console.print("[red]webrtcvad not installed. Skipping VAD.[/red]")
            return in_wav_path

        out_wav = os.path.join(self.script_dir, "temp_static_silence_removed.wav")

        try:
            # Open and read input file
            wf_in = wave.open(in_wav_path, 'rb')
            channels = wf_in.getnchannels()
            rate = wf_in.getframerate()
            
            # Check if file format is compatible with VAD
            if channels != 1:
                self.console.print("[red]VAD requires mono audio. Skipping VAD.[/red]")
                wf_in.close()
                return in_wav_path
                
            if rate not in [8000, 16000, 32000, 48000]:
                self.console.print("[red]VAD requires specific sample rates. Skipping VAD.[/red]")
                wf_in.close()
                return in_wav_path

            # Read all audio data
            audio_data = wf_in.readframes(wf_in.getnframes())
            wf_in.close()

            # Initialize VAD
            vad = webrtcvad.Vad(aggressiveness)

            # Process audio in 30ms frames
            frame_ms = 30
            frame_bytes = int(rate * 2 * (frame_ms/1000.0))  # 16-bit samples = 2 bytes each
            
            voiced_bytes = bytearray()
            idx = 0

            # Process each frame
            while idx + frame_bytes <= len(audio_data):
                frame = audio_data[idx:idx+frame_bytes]
                is_speech = vad.is_speech(frame, rate)
                if is_speech:
                    voiced_bytes.extend(frame)
                idx += frame_bytes

            # Check if we found any speech
            if len(voiced_bytes) == 0:
                self.console.print("[red]VAD found no voice frames. Using original audio.[/red]")
                return in_wav_path

            # Write out the speech-only audio
            wf_out = wave.open(out_wav, 'wb')
            wf_out.setnchannels(1)
            wf_out.setsampwidth(2)  # 16-bit
            wf_out.setframerate(rate)
            wf_out.writeframes(voiced_bytes)
            wf_out.close()

            self.console.print("[yellow]VAD processing complete: Only voice frames retained.[/yellow]")
            return out_wav
            
        except Exception as e:
            self.console.print(f"[red]VAD processing error: {e}[/red]")
            return in_wav_path
    
    def transcribe_file(self) -> None:
        """Transcribe a static audio file selected by the user."""
        self.console.print("[bold yellow]TRANSCRIBE_STATIC command received[/bold yellow]")
        self.tray.set_color('yellow', self.config.send_enter)

        self._cleanup_temp_files()
        
        # Open file dialog
        root = tkinter.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="Select an audio file to transcribe",
            filetypes=[("Audio Files", "*.*")]
        )
        root.destroy()

        if not file_path:
            self.console.print("[red]No file selected. Aborting static transcription.[/red]")
            self.tray.set_color('gray', self.config.send_enter)
            return

        self.console.print(f"[green]Selected file: {file_path}[/green]")

        try:
            # Step 1: Convert to WAV format if needed
            wav_path = self._ensure_wav_format(file_path)
            if not wav_path or not os.path.exists(wav_path):
                self.console.print("[bold red]Failed to convert audio file. Aborting.[/bold red]")
                self.tray.set_color('gray', self.config.send_enter)
                return

            # Step 2: Remove music segments (NEW STEP!)
            music_free_path = self._remove_music_segments(wav_path)

            # Step 3: Apply VAD to remove non-speech sections
            voice_wav = self._apply_vad(music_free_path, aggressiveness=2)

            # Step 4: Transcribe the processed audio
            self.console.print("[blue]Beginning transcription with voice-only data...[/blue]")
            final_text = self.transcriber.transcribe(voice_wav)
            
            # Display results
            panel = Panel(
                f"[bold magenta]Static File Transcription:[/bold magenta] {final_text}",
                title="Static Transcription",
                border_style="yellow"
            )
            self.console.print(panel)

            # Save .txt alongside the original file
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            dir_name = os.path.dirname(file_path)
            out_txt_path = os.path.join(dir_name, base_name + ".txt")

            with open(out_txt_path, "w", encoding="utf-8") as f:
                f.write(final_text)

            self.console.print(f"[green]Saved transcription to: {out_txt_path}[/green]")
            
        except Exception as e:
            self.console.print(f"[bold red]Static transcription failed: {e}[/bold red]")
            
        finally:
            self.tray.set_color('gray', self.config.send_enter)

# --------------------------------------------------------------------------------------
# Command Server
# --------------------------------------------------------------------------------------
class CommandServer:
    def __init__(self, app: 'STTApp'):
        self.app = app
        self.console = app.console
        self.keep_running = True
        self.server_thread = None
    
    def start(self) -> None:
        """Start the TCP server in a separate thread."""
        self.server_thread = threading.Thread(target=self._run_server, daemon=True)
        self.server_thread.start()
    
    def _run_server(self) -> None:
        """Run the TCP server to receive commands from AHK."""
        host = '127.0.0.1'
        port = 34909
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind((host, port))
            sock.listen(5)
            sock.settimeout(1)  # Allow checking keep_running flag every second

            self.console.print(f"[bold yellow]TCP server listening on {host}:{port}[/bold yellow]")

            while self.keep_running:
                try:
                    conn, addr = sock.accept()
                    data = conn.recv(1024).decode('utf-8').strip()
                    self.console.print(f"[italic cyan]Received command: '{data}'[/italic cyan]")

                    # Process command
                    self._handle_command(data)
                    conn.close()
                    
                except socket.timeout:
                    continue  # Just a timeout, check keep_running and continue
                except Exception as e:
                    self.console.print(f"[red]Socket error: {e}[/red]")
                    
        except Exception as e:
            self.console.print(f"[bold red]Server error: {e}[/bold red]")
            
        finally:
            if 'sock' in locals():
                sock.close()
    
    def _handle_command(self, command: str) -> None:
        """Process commands received from AHK."""
        if command == "TOGGLE_LANGUAGE":
            self.app.toggle_language()
        elif command == "START_RECORDING":
            self.app.start_recording()
        elif command == "STOP_AND_TRANSCRIBE":
            self.app.stop_and_transcribe()
        elif command == "TOGGLE_ENTER":
            self.app.toggle_enter()
        elif command == "TRANSCRIBE_STATIC":
            self.app.transcribe_static()
        elif command == "QUIT":
            self.console.print("[bold red]Received QUIT command[/bold red]")
            self.stop()
    
    def stop(self) -> None:
        """Stop the server and application."""
        self.keep_running = False
        self.app.shutdown()

# --------------------------------------------------------------------------------------
# Main Application
# --------------------------------------------------------------------------------------
class STTApp:
    def __init__(self):
        self.console = Console()
        self.config = Config()
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.ahk_pid = None
        
        # Initialize components
        self.tray = TrayManager(self.console)
        self.transcriber = Transcriber(self.config, self.console)
        self.recorder = AudioRecorder(self.config, self.console, self.transcriber, self.tray)
        self.static_processor = StaticFileProcessor(self.config, self.console, self.transcriber, self.tray)
        self.server = CommandServer(self)
    
    def _display_info(self) -> None:
        """Display startup information."""
        panel_content = (
            f"[bold yellow]Model[/bold yellow]: {self.transcriber.model_id}\n"
            f"[bold yellow]Hotkeys[/bold yellow]: Controlled by AutoHotKey script '{self.config.hotkey_script}'\n"
            " F2  -> toggle language\n"
            " F3  -> start recording\n"
            " F4  -> stop & transcribe\n"
            " F5  -> toggle enter\n"
            " F6  -> quit\n"
            " F10 -> static file transcription\n"
            f"[bold yellow]Language[/bold yellow]: {self.config.language}\n"
            f"[bold yellow]Send Enter[/bold yellow]: {self.config.send_enter}"
        )
        panel = Panel(panel_content, title="Information", border_style="green")
        self.console.print(panel)
    
    def _kill_leftover_ahk(self) -> None:
        """Kill any existing AHK processes using our script."""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if (
                    proc.info['name'] == 'AutoHotkeyU64.exe'
                    and proc.info['cmdline'] is not None
                    and self.config.hotkey_script in ' '.join(proc.info['cmdline'])
                ):
                    self.console.print(f"[yellow]Killing leftover AHK process with PID={proc.pid}[/yellow]")
                    psutil.Process(proc.pid).kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
    
    def _start_ahk_script(self) -> None:
        """Launch the AHK script and track its PID."""
        # Record existing AHK PIDs before launching
        pre_pids = set()
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if proc.info['name'] == 'AutoHotkeyU64.exe':
                    pre_pids.add(proc.info['pid'])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        # Launch the AHK script
        ahk_path = os.path.join(self.script_dir, self.config.hotkey_script)
        self.console.print("[green]Launching AHK script...[/green]")
        subprocess.Popen(
            [ahk_path],
            creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
            shell=True
        )

        # Give it a moment to start
        time.sleep(1.0)
        
        # Find the new AHK process
        post_pids = set()
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if proc.info['name'] == 'AutoHotkeyU64.exe':
                    post_pids.add(proc.info['pid'])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        # Store the PID of the new process
        new_pids = post_pids - pre_pids
        if len(new_pids) == 1:
            self.ahk_pid = new_pids.pop()
            self.console.print(f"[green]Detected new AHK script PID: {self.ahk_pid}[/green]")
        else:
            self.console.print("[red]Could not detect a single new AHK script PID. No PID stored.[/red]")
            self.ahk_pid = None
    
    def start(self) -> None:
        """Start the application."""
        self._display_info()
        self._kill_leftover_ahk()
        self._start_ahk_script()
        self.tray.set_color('gray', self.config.send_enter)
        self.server.start()
        
        # Keep the main thread alive
        try:
            while self.server.keep_running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.shutdown()
    
    # Command handler methods
    def toggle_language(self) -> None:
        """Toggle between Greek and English languages."""
        self.transcriber.toggle_language()
    
    def start_recording(self) -> None:
        """Start recording audio from the microphone."""
        self.recorder.start()
    
    def stop_and_transcribe(self) -> None:
        """Stop recording and transcribe the audio."""
        self.recorder.stop_and_transcribe()
    
    def toggle_enter(self) -> None:
        """Toggle whether to send Enter after pasting transcription."""
        self.config.send_enter = not self.config.send_enter
        self.console.print(f"[cyan]Toggled send_enter to: {self.config.send_enter}[/cyan]")
        
        # Update the tray icon to reflect the new send_enter state
        current_color = 'gray'  # Default color if we can't determine current state
        if hasattr(self.tray, 'tray_icon') and self.tray.tray_icon is not None:
            # Try to determine current color by comparing with known icons
            icon_sets = [self.tray.icons, self.tray.icons_green]
            for icon_set in icon_sets:
                for color, icon in icon_set.items():
                    if self.tray.tray_icon.icon == icon:
                        current_color = color
                        break
        
        # Update the icon with the current color but new outline
        self.tray.set_color(current_color, self.config.send_enter)
    
    def transcribe_static(self) -> None:
        """Transcribe a static audio file."""
        self.static_processor.transcribe_file()
    
    def shutdown(self) -> None:
        """Shut down the application gracefully."""
        self.console.print("[bold red]Shutting down gracefully...[/bold red]")
        
        # Kill AHK script if we know its PID
        if self.ahk_pid is not None:
            self.console.print(f"[bold red]Killing AHK script with PID={self.ahk_pid}[/bold red]")
            try:
                psutil.Process(self.ahk_pid).kill()
            except Exception as e:
                self.console.print(f"[red]Failed to kill AHK process: {e}[/red]")
        
        # Clean up resources
        self.tray.set_color('gray', self.config.send_enter)
        self.tray.stop()
        
        sys.exit(0)

# --------------------------------------------------------------------------------------
# Main Entry Point
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    app = STTApp()
    app.start()