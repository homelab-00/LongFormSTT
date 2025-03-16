# realtime_transcription_handler.py
#
# Provides real-time speech-to-text transcription with immediate feedback
#
# This module:
# - Captures and processes audio in real-time from microphone or system audio
# - Detects speech segments using Voice Activity Detection (WebRTC VAD)
# - Performs immediate transcription of detected speech
# - Displays transcription results as they become available
# - Manages real-time transcription models with lazy loading
# - Handles translation differently based on model capabilities:
#   * Turbo models (faster) can only transcribe but not translate
#   * When using turbo models, translation requests are delegated to the long-form model
#   * Non-turbo models handle both transcription and translation themselves
# - Provides graceful error handling with fallback to main model
#
# The real-time mode offers lower latency at the cost of potentially
# reduced accuracy compared to the long-form transcription

import collections
import numpy as np
import time
import threading
import torch
import pyaudio
from rich.panel import Panel
from faster_whisper import WhisperModel

# Optional dependencies
try:
    import webrtcvad
    WEBRTC_VAD_AVAILABLE = True
except ImportError:
    WEBRTC_VAD_AVAILABLE = False

class RealtimeTranscriptionHandler:
    def __init__(self, config, console, transcriber, tray, model_name=None):
        self.config = config
        self.console = console
        self.transcriber = transcriber
        self.tray = tray
        
        # Real-time transcription state
        self.is_running = False
        self.thread = None
        self.stop_event = threading.Event()
        self.beam_size_realtime = 3  # NEW: attribute to avoid "no attribute" errors
        self.audio_buffer_lock = threading.Lock()
        
        # Audio buffers
        self.audio_buffer = collections.deque(maxlen=50)
        self.frames = []
        
        # Silence detection
        self.webrtc_vad_model = None if not WEBRTC_VAD_AVAILABLE else webrtcvad.Vad(3)  # Aggression level 3 (least sensitive)
        self.silence_threshold_ms = 500  # Silent period to consider speech finished (milliseconds)
        self.is_speech_active = False
        self.last_speech_time = 0
        
        # Real-time model
        self.realtime_model = None
        self.realtime_model_loaded = False
        self.realtime_model_name = model_name if model_name else "deepdml/faster-whisper-large-v3-turbo-ct2"
        
        # Audio input stream
        self.audio = None
        self.stream = None
    
    def _is_turbo_model(self):
        """Check if the current real-time model is a turbo model."""
        return "turbo" in self.realtime_model_name.lower()
    
    def _load_realtime_model(self):
        """Lazy-load the real-time transcription model."""
        if not self.realtime_model_loaded:
            self.console.print(f"[bold green]Loading real-time transcription model: {self.realtime_model_name}[/bold green]")
            
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                compute_type = "float16" if device == "cuda" else "float32"

                # Normalize the model name if it's in the wrong format
                model_name = self.realtime_model_name
                if model_name.startswith("models--"):
                    parts = model_name.split("--")
                    if len(parts) >= 3:
                        model_name = f"{parts[1]}/{parts[2]}"
                        self.console.print(f"[yellow]Normalized model name from {self.realtime_model_name} to: {model_name}[/yellow]")
                else:
                    model_name = self.realtime_model_name
                
                self.realtime_model = WhisperModel(
                    model_name,
                    device=device,
                    compute_type=compute_type
                )
                self.realtime_model_loaded = True
                self.console.print("[bold green]Real-time model successfully loaded![/bold green]")
            except Exception as e:
                self.console.print(f"[bold red]Failed to load real-time model: {e}[/bold red]")
                # Fall back to using the main model if available
                self.console.print("[yellow]Falling back to main transcription model[/yellow]")
                return False
                
        return True
    
    def _process_text(self, text):
        """Display real-time transcription results."""
        panel = Panel(
            f"[bold magenta]Live Transcription:[/bold magenta] {text}",
            title="Real-time",
            border_style="cyan"
        )
        self.console.print(panel)
    
    def _initialize_audio(self):
        """Initialize the audio input stream."""
        self.audio = pyaudio.PyAudio()

        # Set up audio parameters
        stream_params = {
            'format': self.config.format,
            'channels': self.config.channels,
            'rate': self.config.rate,
            'input': True,
            'frames_per_buffer': self.config.chunk,
            'input_device_index': self.config.input_device_index if self.config.realtime_use_system_audio else None
        }

        self.stream = self.audio.open(**stream_params)
        return True
    
    def _cleanup_audio(self):
        """Clean up audio resources."""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
            
        if self.audio:
            self.audio.terminate()
            self.audio = None
    
    def _is_speech(self, audio_chunk):
        """Detect if audio chunk contains speech using WebRTC VAD."""
        if not WEBRTC_VAD_AVAILABLE or not self.webrtc_vad_model:
            return True  # If VAD not available, treat all audio as speech
        
        # WebRTC VAD expects specific frame durations (10, 20, or 30ms)
        # We need to ensure our chunk size is compatible
        frame_duration_ms = 30  # 30ms frames work well for WebRTC VAD
        
        # Calculate required sample size for given duration
        required_samples = int(self.config.rate * frame_duration_ms / 1000)
        
        # Skip if chunk is too small
        if len(audio_chunk) < required_samples * 2:  # *2 because 16-bit = 2 bytes per sample
            return False
        
        # Process in 30ms frames
        frames_to_check = []
        for i in range(0, len(audio_chunk) - required_samples * 2, required_samples * 2):
            frames_to_check.append(audio_chunk[i:i + required_samples * 2])
            
        # Consider chunk to contain speech if at least 25% of frames have speech
        speech_frames = 0
        for frame in frames_to_check:
            if len(frame) == required_samples * 2:  # Ensure correct frame size
                try:
                    if self.webrtc_vad_model.is_speech(frame, self.config.rate):
                        speech_frames += 1
                except Exception:
                    continue
        
        speech_ratio = speech_frames / len(frames_to_check) if frames_to_check else 0
        return speech_ratio > 0.25  # At least 25% of frames contain speech
        
    def start(self):
        """Start real-time transcription in a separate thread."""
        if self.is_running:
            self.console.print("[bold yellow]Real-time transcription already running![/bold yellow]")
            return
        
        self.console.print("[bold green]Starting real-time transcription...[/bold green]")
        
        # Load the real-time model if not already loaded
        if not self._load_realtime_model():
            self.console.print("[yellow]Will use main model for real-time transcription[/yellow]")
        
        # Update tray icon
        self.tray.set_color('blue', self.config.send_enter)
        
        # Initialize audio
        if not self._initialize_audio():
            self.console.print("[bold red]Failed to initialize audio input.[/bold red]")
            return
        
        # Reset state
        self.is_speech_active = False
        self.last_speech_time = 0
        
        # Start the transcription thread
        self.is_running = True
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._transcription_loop, daemon=True)
        self.thread.start()
        
        self.console.print("[bold green]Real-time transcription started![/bold green]")
    
    def stop(self):
        """Stop real-time transcription."""
        if not self.is_running:
            self.console.print("[bold yellow]Real-time transcription not running![/bold yellow]")
            return
        
        self.console.print("[bold yellow]Stopping real-time transcription...[/bold yellow]")
        
        # Signal the thread to stop
        self.is_running = False
        self.stop_event.set()
        
        # Clean up audio resources
        self._cleanup_audio()
        
        # Wait for the thread to finish
        if self.thread:
            self.thread.join(timeout=2)
            self.thread = None
        
        # Update tray icon
        self.tray.set_color('gray', self.config.send_enter)
        
        self.console.print("[bold green]Real-time transcription stopped.[/bold green]")
    
    def toggle(self):
        """Toggle real-time transcription on/off."""
        if self.is_running:
            self.stop()
        else:
            self.start()
    
    def toggle_audio_source(self):
        """Toggle between system audio and microphone for real-time transcription."""
        # Simply toggle the setting without starting/stopping transcription
        self.config.realtime_use_system_audio = not self.config.realtime_use_system_audio
        source_str = "system audio" if self.config.realtime_use_system_audio else "microphone"
        self.console.print(f"[cyan]Real-time audio source set to: {source_str}[/cyan]")

        # Update the tray icon to reflect the change
        self.tray.flash_white('gray', self.config.send_enter)
    
    def _transcription_loop(self):
        """Main loop for real-time transcription."""
        try:
            accumulated_speech = []
            current_segment = []
            silence_start_time = 0
            
            while self.is_running and not self.stop_event.is_set():
                # Read audio data
                try:
                    data = self.stream.read(self.config.chunk, exception_on_overflow=False)
                    
                    # Check if this chunk contains speech
                    contains_speech = self._is_speech(data)
                    current_time = time.time()
                    
                    if contains_speech:
                        # Speech detected
                        if not self.is_speech_active:
                            self.console.print("[cyan]Speech detected[/cyan]")
                            self.is_speech_active = True
                            silence_start_time = 0
                        with self.audio_buffer_lock:
                            current_segment.append(data)
                    else:
                        # No speech in this chunk
                        if self.is_speech_active:
                            if silence_start_time == 0:
                                silence_start_time = current_time
                            with self.audio_buffer_lock:
                                current_segment.append(data)
                            
                            if (current_time - silence_start_time) * 1000 >= self.silence_threshold_ms:
                                self.console.print("[cyan]End of speech segment detected[/cyan]")
                                self.is_speech_active = False
                                
                                if current_segment:
                                    with self.audio_buffer_lock:
                                        accumulated_speech.extend(current_segment)
                                        current_segment = []
                                    
                                    # Convert audio data to float32 format
                                    audio_data = np.frombuffer(b''.join(accumulated_speech), dtype=np.int16)
                                    audio_float = audio_data.astype(np.float32) / 32768.0
                                    
                                    # Determine the transcription task based on language
                                    transcription_task = "transcribe"
                                    if self.config.realtime_language not in ["en", "el"]:
                                        transcription_task = "translate"

                                    if self._is_turbo_model():
                                        # Turbo models can't translate, so only use them for transcription
                                        if transcription_task == "transcribe" and self.realtime_model_loaded:
                                            try:
                                                segments, info = self.realtime_model.transcribe(
                                                    audio_float,
                                                    language=self.config.realtime_language,
                                                    beam_size=self.beam_size_realtime,
                                                    task="transcribe"  # Explicitly set for transcription
                                                )
                                                text = "".join(segment.text for segment in segments)
                                            except Exception as e:
                                                self.console.print(f"[red]Real-time model transcription error: {e}[/red]")
                                                # Fallback to long-form model on error
                                                text = self.transcriber.transcribe_audio_data(audio_float)
                                        else:
                                            # Use long-form model for translation since turbo doesn't support it
                                            text = self.transcriber.transcribe_audio_data(audio_float)
                                    else:
                                        # Non-turbo models can handle both transcription and translation
                                        if self.realtime_model_loaded:
                                            try:
                                                segments, info = self.realtime_model.transcribe(
                                                    audio_float,
                                                    language=self.config.realtime_language,
                                                    beam_size=self.beam_size_realtime,
                                                    task=transcription_task
                                                )
                                                text = "".join(segment.text for segment in segments)
                                            except Exception as e:
                                                self.console.print(f"[red]Real-time model transcription error: {e}[/red]")
                                                # Fallback to main transcriber on error
                                                text = self.transcriber.transcribe_audio_data(audio_float)
                                        else:
                                            # Use main transcriber as fallback
                                            text = self.transcriber.transcribe_audio_data(audio_float)
                                    
                                    if text:
                                        self._process_text(text)
                                    
                                    # Keep some frames for context
                                    keep_frames = min(20, len(accumulated_speech))
                                    with self.audio_buffer_lock:
                                        accumulated_speech = accumulated_speech[-keep_frames:] if keep_frames > 0 else []
                    
                    time.sleep(0.01)
                    
                except Exception as e:
                    self.console.print(f"[bold red]Error reading audio: {e}[/bold red]")
                    time.sleep(0.1)
                
        except Exception as e:
            self.console.print(f"[bold red]Error in real-time transcription: {e}[/bold red]")
        finally:
            self._cleanup_audio()