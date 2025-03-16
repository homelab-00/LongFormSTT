import time
import threading
import os
import re
import wave
import glob
import audioop
import torch
import keyboard
import pyperclip
import pyaudio
from dataclasses import dataclass
from typing import List, Optional, Pattern
from rich.console import Console
from rich.panel import Panel
from faster_whisper import WhisperModel

# --------------------------------------------------------------------------------------
# Configuration Classes
# --------------------------------------------------------------------------------------
@dataclass
class AudioConfig:
    format: int = pyaudio.paInt16
    channels: int = 1
    rate: int = 16000
    chunk: int = 1024
    device_index: Optional[int] = None
    threshold: int = 500
    silence_limit: float = 1.5
    chunk_interval: int = 60

@dataclass
class ModelConfig:
    model_id: str = "Systran/faster-whisper-large-v3"
    language: str = "el"
    task: str = "transcribe"
    compute_type: str = "float16" if torch.cuda.is_available() else "float32"

@dataclass
class AppConfig:
    use_system_audio: bool = True
    input_device_index: int = 2
    paste_enabled: bool = True
    hallucinations: List[Pattern] = None

# --------------------------------------------------------------------------------------
# Main Application Class
# --------------------------------------------------------------------------------------
class AudioTranscriber:
    def __init__(self, audio_cfg: AudioConfig, model_cfg: ModelConfig, app_cfg: AppConfig):
        self.audio_cfg = audio_cfg
        self.model_cfg = model_cfg
        self.app_cfg = app_cfg
        self.console = Console()
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Audio initialization
        self.pyaudio = pyaudio.PyAudio()
        self.stream = None
        self.buffer = []
        
        # Model initialization
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = WhisperModel(
            self.model_cfg.model_id,
            device=self.device,
            compute_type=self.model_cfg.compute_type
        )
        
        # State management
        self.recording = False
        self.current_chunk = 1
        self.active_wavefile = None
        self.partial_transcripts = []
        self.transcription_threads = []
        self.record_start_time = 0
        self.next_split_time = 0
        self.chunk_split_requested = False

    # ----------------------------------------------------------------------------------
    # Core Functionality
    # ----------------------------------------------------------------------------------
    def start_recording(self):
        """Start a new recording session with clean state"""
        if self.recording:
            self.console.print("[yellow]Already recording![/yellow]")
            return

        self._cleanup_previous_files()
        self._initialize_recording_state()
        
        try:
            self._open_audio_stream()
            self._create_wave_file()
            threading.Thread(target=self._recording_loop, daemon=True).start()
            self.console.print("[green]Recording started[/green]")
        except Exception as e:
            self.console.print(f"[red]Failed to start recording: {e}[/red]")
            self._cleanup_resources()

    def stop_recording(self):
        """Stop recording and process collected audio"""
        if not self.recording:
            return

        self.recording = False
        self._finalize_recording()
        self._process_remaining_chunk()
        self._wait_for_transcriptions()
        
        final_text = "".join(self.partial_transcripts)
        self._output_results(final_text)
        self._cleanup_resources()

    # ----------------------------------------------------------------------------------
    # Recording Implementation
    # ----------------------------------------------------------------------------------
    def _recording_loop(self):
        """Main audio capture and processing loop"""
        silence_duration = 0.0
        chunk_counter = 0
        chunks_per_second = self.audio_cfg.rate // self.audio_cfg.chunk
        
        try:
            while self.recording:
                data = self.stream.read(self.audio_cfg.chunk, exception_on_overflow=False)
                peak = audioop.max(data, 2)
                chunk_time = self.audio_cfg.chunk / self.audio_cfg.rate
                
                self._handle_timing()
                self._handle_silence_detection(peak, chunk_time, silence_duration)
                
                self.buffer.append(data)
                chunk_counter += 1
                
                # Write every second of audio
                if chunk_counter >= chunks_per_second:
                    self._write_buffer()
                    chunk_counter = 0
                    
        except Exception as e:
            self.console.print(f"[red]Recording error: {e}[/red]")
        finally:
            self._write_buffer()

    def _handle_timing(self):
        """Check if we need to request a chunk split"""
        elapsed = time.time() - self.record_start_time
        if not self.chunk_split_requested and elapsed >= self.audio_cfg.chunk_interval:
            self.console.print(f"[yellow]Reached {int(elapsed)}s - will split at next silence[/yellow]")
            self.chunk_split_requested = True

    def _write_buffer(self):
        """Write accumulated audio data to disk"""
        if self.buffer:
            try:
                self.active_wavefile.writeframes(b''.join(self.buffer))
                self.buffer.clear()
            except Exception as e:
                self.console.print(f"[red]Error writing buffer: {e}[/red]")

    def _close_current_wavefile(self):
        """Properly close the current wave file"""
        if self.active_wavefile:
            try:
                self.active_wavefile.close()
                self.console.print(f"[yellow]Closed chunk {self.current_chunk}[/yellow]")
            except Exception as e:
                self.console.print(f"[red]Error closing wave file: {e}[/red]")
            finally:
                self.active_wavefile = None
    
    def _create_wave_file(self):
        """Create new wave file for current chunk"""
        filename = os.path.join(self.script_dir, f"temp_audio_file{self.current_chunk}.wav")
        try:
            self.active_wavefile = wave.open(filename, "wb")
            self.active_wavefile.setnchannels(self.audio_cfg.channels)
            self.active_wavefile.setsampwidth(self.pyaudio.get_sample_size(self.audio_cfg.format))
            self.active_wavefile.setframerate(self.audio_cfg.rate)
            self.console.print(f"[green]Created new chunk: {filename}[/green]")
        except Exception as e:
            raise RuntimeError(f"Failed to create wave file: {e}")

    # ----------------------------------------------------------------------------------
    # Silence Detection (existing code below)
    # ----------------------------------------------------------------------------------
    def _handle_silence_detection(self, peak: int, chunk_time: float, silence_duration: float):
        """Manage silence detection and chunk splitting"""
        if peak < self.audio_cfg.threshold:
            silence_duration += chunk_time
            if silence_duration > self.audio_cfg.silence_limit:
                self.buffer = []
            if self.chunk_split_requested and silence_duration >= 0.1:
                self._split_chunk()
        else:
            silence_duration = 0.0
            if self.chunk_split_requested:
                self.chunk_split_requested = False

    # ----------------------------------------------------------------------------------
    # Chunk Management
    # ----------------------------------------------------------------------------------
    def _split_chunk(self):
        """Handle chunk splitting logic"""
        self.console.print("[yellow]Splitting chunk...[/yellow]")
        self._close_current_wavefile()
        self._transcribe_chunk(os.path.join(self.script_dir, f"temp_audio_file{self.current_chunk}. wav"))
        self.current_chunk += 1
        self._create_wave_file()
        self.next_split_time += self.audio_cfg.chunk_interval
        self.chunk_split_requested = False

    def _transcribe_chunk(self, filename: str):
        """Process a single audio chunk"""
        def transcribe():
            try:
                segments, _ = self.model.transcribe(
                    filename,
                    language=self.model_cfg.language,
                    task=self.model_cfg.task
                )
                text = "".join(s.text for s in segments)
                text = self._filter_hallucinations(text)
                self.partial_transcripts.append(text)
                self.console.print(f"[cyan]Chunk {self.current_chunk} transcribed[/cyan]")
            except Exception as e:
                self.console.print(f"[red]Transcription failed: {e}[/red]")

        thread = threading.Thread(target=transcribe)
        thread.start()
        self.transcription_threads.append(thread)

    # ----------------------------------------------------------------------------------
    # Utility Methods
    # ----------------------------------------------------------------------------------
    def _filter_hallucinations(self, text: str) -> str:
        """Apply regex filters to remove unwanted artifacts"""
        for pattern in self.app_cfg.hallucinations:
            text = pattern.sub("", text)
        return text.strip()

    def _create_wave_file(self):
        """Create new wave file for current chunk"""
        filename = os.path.join(self.script_dir, f"temp_audio_file{self.current_chunk}.wav")
        try:
            self.active_wavefile = wave.open(filename, "wb")
            self.active_wavefile.setnchannels(self.audio_cfg.channels)
            self.active_wavefile.setsampwidth(self.pyaudio.get_sample_size(self.audio_cfg.format))
            self.active_wavefile.setframerate(self.audio_cfg.rate)
        except Exception as e:
            raise RuntimeError(f"Failed to create wave file: {e}")

    def _open_audio_stream(self):
        """Initialize audio input stream"""
        stream_params = {
            'format': self.audio_cfg.format,
            'channels': self.audio_cfg.channels,
            'rate': self.audio_cfg.rate,
            'input': True,
            'frames_per_buffer': self.audio_cfg.chunk,
            'input_device_index': self.app_cfg.input_device_index
        }
        self.stream = self.pyaudio.open(**stream_params)

    # ----------------------------------------------------------------------------------
    # Cleanup & Initialization
    # ----------------------------------------------------------------------------------
    def _cleanup_previous_files(self):
        """Remove temporary files from previous sessions"""
        for f in glob.glob(os.path.join(self.script_dir, "temp_audio_file*.wav")):
            try:
                os.remove(f)
            except Exception as e:
                self.console.print(f"[red]Error removing file {f}: {e}[/red]")

    def _initialize_recording_state(self):
        """Reset all state variables for new recording"""
        self.recording = True
        self.current_chunk = 1
        self.partial_transcripts.clear()
        self.transcription_threads.clear()
        self.buffer.clear()
        self.record_start_time = time.time()
        self.next_split_time = self.record_start_time + self.audio_cfg.chunk_interval
        self.chunk_split_requested = False

    def _cleanup_resources(self):
        """Release all audio resources"""
        if self.active_wavefile:
            self.active_wavefile.close()
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.pyaudio.terminate()

    # ----------------------------------------------------------------------------------
    # Finalization Methods
    # ----------------------------------------------------------------------------------
    def _finalize_recording(self):
        """Final steps when recording stops"""
        self._close_current_wavefile()
        self.console.print("[green]Recording stopped[/green]")

    def _process_remaining_chunk(self):
        """Process the final incomplete chunk"""
        if os.path.exists(self.active_wavefile.name) and os.path.getsize(self.active_wavefile.name) > 44:
            self._transcribe_chunk(self.active_wavefile.name)

    def _wait_for_transcriptions(self):
        """Wait for all transcription threads to complete"""
        for thread in self.transcription_threads:
            thread.join()

    def _output_results(self, text: str):
        """Display and handle final transcription output"""
        panel = Panel(
            f"[bold green]Transcription Result:[/bold green]\n{text}",
            title="Complete Transcription",
            border_style="blue"
        )
        self.console.print(panel)
        
        if self.app_cfg.paste_enabled:
            pyperclip.copy(text)
            keyboard.send("ctrl+v")

# --------------------------------------------------------------------------------------
# Initialization & Execution
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Configuration setup
    hallucinations = [
        re.compile(r"\bΥπότιτλοι\s+AUTHORWAVE\b[^\w]*", re.IGNORECASE),
        re.compile(r"\bΣας\s+ευχαριστώ\b[^\w]*", re.IGNORECASE),
    ]
    
    app_config = AppConfig(
        input_device_index=2,
        hallucinations=hallucinations
    )
    
    audio_config = AudioConfig()
    model_config = ModelConfig()
    
    # Create transcriber instance
    transcriber = AudioTranscriber(audio_config, model_config, app_config)
    
    # Setup hotkeys
    keyboard.add_hotkey("F2", lambda: setattr(
        transcriber.app_cfg, "paste_enabled", not transcriber.app_cfg.paste_enabled
    ), suppress=True)
    
    keyboard.add_hotkey("F3", transcriber.start_recording, suppress=True)
    keyboard.add_hotkey("F4", transcriber.stop_recording, suppress=True)
    
    # Display startup information
    console = Console()
    info_panel = Panel(
        f"[bold]Audio Transcription Tool[/bold]\n\n"
        f"Model: {model_config.model_id}\n"
        f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}\n"
        f"Hotkeys: F2 (Toggle Paste), F3 (Start), F4 (Stop)",
        title="System Info",
        border_style="green"
    )
    console.print(info_panel)
    
    # Start main loop
    keyboard.wait()