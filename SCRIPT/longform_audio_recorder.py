# longform_audio_recorder.py
#
# Handles audio recording and transcription for long-form speech
#
# This module:
# - Captures audio from microphone or system audio
# - Provides manual control for starting and stopping recording
# - Manages transcription of recorded audio
# - Sends transcribed text to clipboard and optionally presses Enter
# - Provides visual feedback through the tray icon during operations
# - Uses RealtimeSTT library for efficient transcription
#
# Long-form mode allows users to control exactly when recording starts and stops,
# ideal for dictating longer content

import time
import os
import threading
import pyperclip
import keyboard
from rich.panel import Panel
from RealtimeSTT import AudioToTextRecorderClient

class LongFormAudioRecorder:
    def __init__(self, config, console, transcriber, tray):
        self.config = config
        self.console = console
        self.transcriber = transcriber
        self.tray = tray
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Recording state
        self.recording = False
        self.recording_thread = None
        self.recorder = None
        self.text_ready = threading.Event()
        self.final_text = ""
    
    def start(self):
        """Start recording audio."""
        if self.recording:
            self.console.print("[bold yellow]Already recording![/bold yellow]")
            return

        self.console.print("[bold green]Starting a new recording session[/bold green]")
        
        # Reset state
        self.final_text = ""
        self.text_ready.clear()
        
        # Initialize the recorder - but don't start recording automatically
        try:
            self.recorder = AudioToTextRecorderClient(
                model=self.config.longform_model,
                language=self.config.longform_language,
                input_device_index=self.config.input_device_index if self.config.longform_use_system_audio else None,
                enable_realtime_transcription=False,  # Not using real-time updates in long-form mode
                on_recording_start=self._on_recording_start,
                on_recording_stop=self._on_recording_stop,
                spinner=False,  # We'll handle UI feedback with our tray
                silero_sensitivity=0.5,
                post_speech_silence_duration=0.8,  # Slightly longer for long-form
                pre_recording_buffer_duration=0.5,
                debug_mode=False
            )
            
            # Update tray and state
            self.tray.set_color('red', self.config.send_enter)
            self.recording = True
            
            # Set the recording_start event to begin recording
            if hasattr(self.recorder, 'recording_start'):
                self.recorder.recording_start.set()
            
            self.console.print("[green]Recording started.[/green]")
        except Exception as e:
            self.console.print(f"[bold red]Failed to start recording: {e}[/bold red]")
            self._cleanup_resources()
    
    def _on_recording_start(self):
        """Callback when recording starts."""
        self.console.print("[cyan]Recording started[/cyan]")
    
    def _on_recording_stop(self):
        """Callback when recording stops."""
        self.console.print("[cyan]Recording stopped[/cyan]")
    
    def _cleanup_resources(self):
        """Clean up resources and reset state."""
        if self.recorder:
            try:
                self.recorder.shutdown()
            except Exception as e:
                self.console.print(f"[red]Error shutting down recorder: {e}[/red]")
            finally:
                self.recorder = None
        
        self.recording = False
    
    def _transcription_thread_func(self):
        """Thread function to handle transcription after stopping recording."""
        try:
            # Get transcription text using the text() method
            # This will automatically start the transcription process
            self.final_text = self.recorder.text()
            
            # Clean up resources
            self._cleanup_resources()
            
            # Signal that text is ready
            self.text_ready.set()
            
        except Exception as e:
            self.console.print(f"[bold red]Transcription error: {e}[/bold red]")
            self._cleanup_resources()
            self.text_ready.set()  # Set the event even on error to prevent hanging
    
    def stop_and_transcribe(self):
        """Stop recording and transcribe the audio."""
        if not self.recording:
            self.console.print("[italic bold yellow]Recording not in progress[/italic bold yellow]")
            return

        self.console.print("[bold blue]Stopping recording and transcribing...[/bold blue]")
        
        # Update tray icon
        self.tray.set_color('blue', self.config.send_enter)
        
        # Stop recording - clear the recording_start event
        if self.recorder and hasattr(self.recorder, 'recording_start'):
            self.recorder.recording_start.clear()
        
        # Start transcription in a separate thread
        self.transcription_thread = threading.Thread(target=self._transcription_thread_func)
        self.transcription_thread.daemon = True
        self.transcription_thread.start()
        
        # Wait for transcription to complete
        self.text_ready.wait()
        
        # Process the result
        if self.final_text:
            # Display the result
            panel = Panel(
                f"[bold magenta]Final Transcription:[/bold magenta] {self.final_text}",
                title="Transcription",
                border_style="yellow"
            )
            self.console.print(panel)

            # Copy to clipboard and paste
            pyperclip.copy(self.final_text)
            keyboard.send('ctrl+v')
            if self.config.send_enter:
                keyboard.send('enter')
                self.console.print("[yellow]Sent an ENTER keystroke after transcription.[/yellow]")
        else:
            self.console.print("[yellow]No transcription result available.[/yellow]")

        self.console.print("[italic green]Done.[/italic green]")
        self.tray.set_color('gray', self.config.send_enter)