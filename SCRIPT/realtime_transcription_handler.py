# realtime_transcription_handler.py
#
# Provides real-time speech-to-text transcription with immediate feedback
#
# This module:
# - Captures and processes audio in real-time from microphone or system audio
# - Detects speech segments using Voice Activity Detection
# - Performs immediate transcription of detected speech
# - Displays transcription results as they become available
# - Leverages the RealtimeSTT library for efficient, real-time transcription
# - Manages model loading and configuration
# - Provides graceful error handling with recovery
#
# The real-time mode offers lower latency at the cost of potentially
# reduced accuracy compared to the long-form transcription

import threading
import time
from rich.panel import Panel
from RealtimeSTT import AudioToTextRecorderClient

class RealtimeTranscriptionHandler:
    def __init__(self, config, console, transcriber, tray, model_name=None):
        self.config = config
        self.console = console
        self.transcriber = transcriber
        self.tray = tray
        
        # Real-time transcription state
        self.is_running = False
        self.realtime_recorder = None
        self.stop_event = threading.Event()
        
        # Real-time model
        self.realtime_model_name = model_name if model_name else "deepdml/faster-whisper-large-v3-turbo-ct2"
    
    def _process_text(self, text):
        """Display real-time transcription results."""
        panel = Panel(
            f"[bold magenta]Live Transcription:[/bold magenta] {text}",
            title="Real-time",
            border_style="cyan"
        )
        self.console.print(panel)
        
    def _setup_realtime_recorder(self):
        """Initialize the AudioToTextRecorderClient for real-time transcription."""
        try:
            # Only create a new instance if one doesn't already exist or if it's not running
            if self.realtime_recorder is None:
                self.console.print(f"[bold green]Initializing real-time transcription with model: {self.realtime_model_name}[/bold green]")
                
                # Create AudioToTextRecorderClient instance with appropriate parameters
                self.realtime_recorder = AudioToTextRecorderClient(
                    model=self.realtime_model_name,
                    language=self.config.realtime_language,
                    input_device_index=self.config.input_device_index if self.config.realtime_use_system_audio else None,
                    enable_realtime_transcription=True,
                    on_realtime_transcription_update=self._on_realtime_update,
                    on_recording_start=self._on_recording_start,
                    on_recording_stop=self._on_recording_stop,
                    spinner=False,  # Don't use the library's spinner since we use our own tray icon
                    silero_sensitivity=0.5,
                    webrtc_sensitivity=3,
                    post_speech_silence_duration=0.6,
                    min_length_of_recording=0.5,
                    pre_recording_buffer_duration=0.3,
                    debug_mode=False
                )
                return True
                
            return True
                
        except Exception as e:
            self.console.print(f"[bold red]Failed to initialize real-time recorder: {e}[/bold red]")
            return False
    
    def _on_realtime_update(self, text):
        """Callback for real-time transcription updates."""
        if text:
            self._process_text(text)
    
    def _on_recording_start(self):
        """Callback when recording starts."""
        self.console.print("[cyan]Speech detected - Recording started[/cyan]")
        # Update tray icon state
        self.tray.set_color('red', self.config.send_enter)
    
    def _on_recording_stop(self):
        """Callback when recording stops."""
        self.console.print("[cyan]Recording stopped[/cyan]")
        # Update tray icon state back to 'blue' for listening mode
        self.tray.set_color('blue', self.config.send_enter)
    
    def start(self):
        """Start real-time transcription."""
        if self.is_running:
            self.console.print("[bold yellow]Real-time transcription already running![/bold yellow]")
            return
        
        self.console.print("[bold green]Starting real-time transcription...[/bold green]")
        
        # Initialize the real-time recorder
        if not self._setup_realtime_recorder():
            self.console.print("[bold red]Failed to set up real-time transcription.[/bold red]")
            return
        
        # Update tray icon
        self.tray.set_color('blue', self.config.send_enter)
        
        # Start real-time transcription
        self.is_running = True
        self.stop_event.clear()
        
        self.console.print("[bold green]Real-time transcription started![/bold green]")
    
    def stop(self):
        """Stop real-time transcription."""
        if not self.is_running:
            self.console.print("[bold yellow]Real-time transcription not running![/bold yellow]")
            return
        
        self.console.print("[bold yellow]Stopping real-time transcription...[/bold yellow]")
        
        # Signal to stop and clean up
        self.is_running = False
        self.stop_event.set()
        
        # Clean up and shut down the recorder
        if self.realtime_recorder:
            self.realtime_recorder.abort()
            self.realtime_recorder.shutdown()
            self.realtime_recorder = None
        
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

        # If already running, restart with the new audio source
        was_running = self.is_running
        if was_running:
            self.stop()
            
        # Update the tray icon to reflect the change
        self.tray.flash_white('gray', self.config.send_enter)
        
        # Restart if it was running before
        if was_running:
            # Small delay to ensure proper shutdown before restarting
            time.sleep(0.5)
            self.start()