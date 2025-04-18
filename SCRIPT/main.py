# main.py
# Main entry point for the Speech to Text application with Faster Whisper
#
# This module initializes all components and orchestrates the application:
# - Loads and manages configuration (language settings, audio sources, models)
# - Creates and coordinates all system components (recorder, transcriber, UI)
# - Handles hotkey commands via TCP server (F1, F2, ...)
# - Manages application lifecycle (startup, shutdown, resource handling)
# - Provides command handlers for all user interactions
# - Toggles between different transcription modes
#
# The application supports three transcription pipelines:
# 1. Long-form transcription: For extended recordings with chunking support
# 2. Real-time transcription: For immediate feedback during speech
# 3. Static file transcription: For processing pre-recorded audio files
#
# The application handles translation depending on the model type (applies to real-time models only):
# Turbo models, which are faster (e.g. deepdml/faster-whisper-large-v3-turbo-ct2 is faster than
# Systran/faster-whisper-medium) but can only transcribe and not translate, therefore:
# - For turbo models, translation requests are delegated to the long-form model
# - Non-turbo models handle both transcription and translation natively

import time
import sys
import pyaudio
import os
import threading
from rich.console import Console
from rich.panel import Panel
import re
import socket
import subprocess
import psutil
import platform
from dataclasses import dataclass
from typing import List

# Import modules
from system_tray_icon_manager import TrayManager
from transcription_engine import Transcriber
from longform_audio_recorder import LongFormAudioRecorder
from static_file_processor import StaticFileProcessor
from realtime_transcription_handler import RealtimeTranscriptionHandler
from unified_configuration_dialog import UnifiedConfigDialog

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
@dataclass
class Config:
    # Audio settings for Long Form STT
    longform_use_system_audio: bool = False           # For long-form STT (default to microphone)
    
    # Audio settings for Real-Time STT
    realtime_use_system_audio: bool = True            # For real-time STT (default to system audio)
    
    # Common audio settings
    input_device_index: int = 3
    format: int = pyaudio.paInt16
    channels: int = 1
    rate: int = 16000
    chunk: int = 1024
    
    # Separate language settings
    longform_language: str = "el"                     # Default Greek for Long Form STT
    realtime_language: str = "en"                     # Default English for Real-Time STT
    task: str = "transcribe"

    # Model settings
    longform_model: str = "Systran/faster-whisper-large-v3"  # Default model for long-form STT
    realtime_model: str = "deepdml/faster-whisper-large-v3-turbo-ct2"  # Default model for real-time STT
    
    # Detection settings
    threshold: int = 500
    silence_limit_sec: float = 1.5
    chunk_split_interval: int = 60
    
    # Transcription settings
    send_enter: bool = False
    
    # System settings
    hotkey_script: str = "linux_hotkeys.py" if platform.system() != 'Windows' else "AHK_script-hotkeys_handling.ahk"
    
    # Derived properties
    @property
    def chunks_per_second(self) -> int:
        return self.rate // self.chunk
    
    # For backwards compatibility
    @property
    def language(self) -> str:
        return self.longform_language
    
    @language.setter
    def language(self, value: str):
        self.longform_language = value
    
    @property
    def use_system_audio(self) -> bool:
        return self.longform_use_system_audio
    
    @use_system_audio.setter
    def use_system_audio(self, value: bool):
        self.longform_use_system_audio = value
    
    # Hallucination filters
    @property
    def hallucinations_regex(self) -> List[re.Pattern]:
        return [
            re.compile(r"\bΥπότιτλοι\s+AUTHORWAVE\b[^\w]*", re.IGNORECASE),
            re.compile(r"\bΣας\s+ευχαριστώ\b[^\w]*", re.IGNORECASE),
        ]

    # User Configuration File
    def save_to_file(self, file_path="userdata.config"):
        """Save current configuration to a file."""
        import json
        import os

        # Get directory of the script
        dir_path = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(dir_path, file_path)

        config_data = {
            # Save all preferences
            "longform_use_system_audio": self.longform_use_system_audio,
            "realtime_use_system_audio": self.realtime_use_system_audio,
            "longform_language": self.longform_language,
            "realtime_language": self.realtime_language,
            "send_enter": self.send_enter,
            "task": self.task,
            "longform_model": self.longform_model,
            "realtime_model": self.realtime_model
        }

        try:
            with open(full_path, 'w') as f:
                json.dump(config_data, f, indent=4)
            return True
        except Exception as e:
            print(f"Error saving configuration: {e}")
            return False

    @classmethod
    def load_from_file(cls, file_path="userdata.config"):
        """Load configuration from a file, or return default if file doesn't exist."""
        import json
        import os

        # Create a default configuration
        config = cls()

        # Get directory of the script
        dir_path = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(dir_path, file_path)

        # If file doesn't exist, return default config
        if not os.path.exists(full_path):
            return config

        try:
            with open(full_path, 'r') as f:
                config_data = json.load(f)

            # Update config with values from file
            for key, value in config_data.items():
                if hasattr(config, key) and value is not None:
                    setattr(config, key, value)

            return config
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return config

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
            # Add socket option to reuse address to avoid "address already in use" errors
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
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
        if command == "OPEN_CONFIG_DIALOG":
            self.app.open_config_dialog()
        if command == "TOGGLE_LANGUAGE":
            self.app.toggle_language()
        elif command == "START_RECORDING":
            self.app.start_recording()
        elif command == "STOP_AND_TRANSCRIBE":
            self.app.stop_and_transcribe()
        elif command == "TOGGLE_ENTER":
            self.app.toggle_enter()
        elif command == "RESET_TRANSCRIPTION":
            self.app.reset_transcription()
        elif command == "TRANSCRIBE_STATIC":
            self.app.transcribe_static()
        elif command == "TOGGLE_REALTIME_TRANSCRIPTION":
            self.app.toggle_realtime_transcription()
        elif command == "QUIT":
            self.console.print("[bold red]Received QUIT command[/bold red]")
            self.stop()
    
    def stop(self) -> None:
        """Stop the server and application."""
        self.keep_running = False

        # If a static transcription is in progress, request abort
        if hasattr(self.app, 'static_processor') and self.app.static_processor.is_transcribing():
            self.app.static_processor.request_abort()

        self.app.shutdown()

# --------------------------------------------------------------------------------------
# Main Application
# --------------------------------------------------------------------------------------
class STTApp:
    def __init__(self):
        self.console = Console()
        self.config = Config.load_from_file()
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.ahk_pid = None
        
        # Ensure temp_audio directory exists
        self.temp_dir = os.path.join(self.script_dir, "temp_audio")
        if not os.path.exists(self.temp_dir):
            try:
                os.makedirs(self.temp_dir)
                self.console.print(f"[green]Created temporary audio directory: {self.temp_dir}[/green]")
            except Exception as e:
                self.console.print(f"[red]Failed to create temp directory: {e}[/red]")
        
        # Initialize components
        self.tray = TrayManager(self.console)
        self.transcriber = Transcriber(self.config, self.console, model_id=self.config.longform_model)
        self.recorder = LongFormAudioRecorder(self.config, self.console, self.transcriber, self.tray)
        self.static_processor = StaticFileProcessor(self.config, self.console, self.transcriber, self.tray)
        self.realtime_handler = RealtimeTranscriptionHandler(self.config, self.console, self.transcriber, self.tray, model_name=self.config.realtime_model)
        self.config_dialog = UnifiedConfigDialog(self.config, self.console, self.realtime_handler, self.transcriber)
        self.server = CommandServer(self)
    
    def _display_info(self) -> None:
        """Display startup information."""
        panel_content = (
            f"[bold yellow]Model[/bold yellow]: {self.transcriber.model_id}\n"
            f"[bold yellow]Hotkeys[/bold yellow]: Controlled by {'AutoHotKey' if platform.system() == 'Windows' else 'Python'} script '{self.config.hotkey_script}'\n"
            " F1  -> Open configuration dialog\n"
            " F2  -> Toggle live transcription\n"
            " F3  -> Start recording\n"
            " F4  -> Stop & transcribe\n"
            " F5  -> Toggle enter\n"
            " F6  -> Reset transcription\n"
            " F7  -> Quit\n"
            " F10 -> Static file transcription\n"
            f"[bold yellow]Long Form STT[/bold yellow]:\n"
            f"  Language: {self.config.longform_language} (Task: {self.config.task})\n"
            f"  Audio Source: {'System Audio' if self.config.longform_use_system_audio else 'Microphone'}\n"
            f"[bold yellow]Real-time STT[/bold yellow]:\n"
            f"  Language: {self.config.realtime_language}\n"
            f"  Audio Source: {'System Audio' if self.config.realtime_use_system_audio else 'Microphone'}\n"
            f"  Model: {self.realtime_handler.realtime_model_name}"
        )
        panel = Panel(panel_content, title="Information", border_style="green")
        self.console.print(panel)

    def open_config_dialog(self) -> None:
        """Open the configuration dialog."""
        self.console.print("[bold yellow]OPEN_CONFIG_DIALOG command received[/bold yellow]")

        # Stop all running transcriptions

        # Check if live recording is in progress
        if self.recorder.recording:
            self.console.print("[bold yellow]Stopping live transcription...[/bold yellow]")
            self.recorder.recording = False
            if self.recorder.recording_thread:
                self.recorder.recording_thread.join()
            self.recorder._cleanup_resources()

        # Check if real-time transcription is running
        if self.realtime_handler.is_running:
            self.console.print("[bold yellow]Stopping real-time transcription...[/bold yellow]")
            self.realtime_handler.stop()

        # Check if static transcription is in progress
        if self.static_processor.is_transcribing():
            self.console.print("[bold yellow]Stopping static transcription...[/bold yellow]")
            self.static_processor.request_abort()

        # Flash the tray icon for visual feedback
        self.tray.flash_white('gray', self.config.send_enter)

        # Show the configuration dialog
        result = self.config_dialog.show_dialog()

        # Process the results if the dialog wasn't cancelled
        if result is not None:
            changes_made = False

            # Apply Long Form language change
            if result["longform_language"] and result["longform_language"] != self.config.longform_language:
                old_lang = self.config.longform_language
                self.config.longform_language = result["longform_language"]
                lang_name = self.config_dialog.languages.get(result["longform_language"], "Unknown")

                # Auto-switch task based on language
                if result["longform_language"] not in ["en", "el"]:
                    self.config.task = "translate"
                else:
                    self.config.task = "transcribe"

                self.console.print(f"[yellow]Long Form language changed from {old_lang} to {result['longform_language']} ({lang_name})[/yellow]")
                changes_made = True

            # Apply Real Time language change
            if result["realtime_language"] and result["realtime_language"] != self.config.realtime_language:
                old_lang = self.config.realtime_language
                self.config.realtime_language = result["realtime_language"]
                lang_name = self.config_dialog.languages.get(result["realtime_language"], "Unknown")

                self.console.print(f"[yellow]Real Time language changed from {old_lang} to {result['realtime_language']} ({lang_name})[/yellow]")
                changes_made = True

            # Apply audio source changes
            if result["longform_use_system_audio"] != self.config.longform_use_system_audio:
                self.config.longform_use_system_audio = result["longform_use_system_audio"]
                source_str = "system audio" if self.config.longform_use_system_audio else "microphone"
                self.console.print(f"[cyan]Long Form audio source changed to: {source_str}[/cyan]")
                changes_made = True

            if result["realtime_use_system_audio"] != self.config.realtime_use_system_audio:
                self.config.realtime_use_system_audio = result["realtime_use_system_audio"]
                source_str = "system audio" if self.config.realtime_use_system_audio else "microphone"
                self.console.print(f"[cyan]Real Time audio source changed to: {source_str}[/cyan]")
                changes_made = True

            # Apply model change
            if result["longform_model_name"] and result["longform_model_name"] != self.transcriber.model_id:
                old_model = self.transcriber.model_id
                self.transcriber.model_id = result["longform_model_name"]
                self.config.longform_model = result["longform_model_name"]
                self.console.print(f"[cyan]Long Form model changed from {old_model} to {result['longform_model_name']}[/cyan]")
                changes_made = True

            if result["realtime_model_name"] and result["realtime_model_name"] != self.realtime_handler.realtime_model_name:
                old_model = self.realtime_handler.realtime_model_name
                self.realtime_handler.realtime_model_name = result["realtime_model_name"]
                self.config.realtime_model = result["realtime_model_name"]
                self.realtime_handler.realtime_model_loaded = False  # Force reload of model
                self.console.print(f"[cyan]Real Time model changed from {old_model} to {result['realtime_model_name']}[/cyan]")
                changes_made = True

            # Apply Enter key toggle
            if result["send_enter"] != self.config.send_enter:
                self.config.send_enter = result["send_enter"]
                self.console.print(f"[cyan]Send Enter after transcription changed to: {self.config.send_enter}[/cyan]")
                changes_made = True

            # Save config if changes were made
            if changes_made:
                # Update the tray icon to reflect the new send_enter state
                self.tray.flash_white('gray', self.config.send_enter)

                # Save configuration to file
                self.config.save_to_file()
                self.console.print("[green]Configuration saved to file.[/green]")
        else:
            self.console.print("[yellow]Configuration dialog cancelled. No changes made.[/yellow]")

    def open_language_menu(self) -> None:
        """Open the language selection menu dialog."""
        self.console.print("[bold yellow]OPEN_LANGUAGE_MENU command received[/bold yellow]")
        
        # Use the unified config dialog instead
        self.open_config_dialog()

    def open_audio_source_menu(self) -> None:
        """Open the audio source selection menu dialog."""
        self.console.print("[bold yellow]OPEN_AUDIO_SOURCE_MENU command received[/bold yellow]")

        # Use the unified config dialog instead
        self.open_config_dialog()
    
    def toggle_longform_audio_source(self) -> None:
        """Toggle between system audio and microphone for long-form STT."""
        self.console.print("[bold yellow]TOGGLE_LONGFORM_AUDIO_SOURCE command received[/bold yellow]")
        
        # Toggle the audio source
        self.config.use_system_audio = not self.config.use_system_audio
        source_str = "system audio" if self.config.use_system_audio else "microphone"
        self.console.print(f"[cyan]Toggled long-form audio source to: {source_str}[/cyan]")
        
        # Flash the tray icon for visual feedback
        self.tray.flash_white('gray', self.config.send_enter)

    def _kill_leftover_hotkey_process(self) -> None:
        """Kill any existing hotkey processes using our script."""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if platform.system() == 'Windows':
                    # Windows: Look for AutoHotkey
                    if (
                        proc.info['name'] == 'AutoHotkeyU64.exe'
                        and proc.info['cmdline'] is not None
                        and self.config.hotkey_script in ' '.join(proc.info['cmdline'])
                    ):
                        self.console.print(f"[yellow]Killing leftover AHK process with PID={proc.pid}[/yellow]")
                        psutil.Process(proc.pid).kill()
                else:
                    # Linux: Look for Python process running our hotkey script
                    if (
                        (proc.info['name'] == 'python3' or proc.info['name'] == 'python')
                        and proc.info['cmdline'] is not None
                        and self.config.hotkey_script in ' '.join(proc.info['cmdline'])
                    ):
                        self.console.print(f"[yellow]Killing leftover hotkey process with PID={proc.pid}[/yellow]")
                        psutil.Process(proc.pid).kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
    
    def _start_hotkey_script(self) -> None:
        """Launch the hotkey script and track its PID."""
        # Record existing script PIDs before launching
        pre_pids = set()
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if platform.system() == 'Windows':
                    if proc.info['name'] == 'AutoHotkeyU64.exe':
                        pre_pids.add(proc.info['pid'])
                else:
                    if proc.info['name'] in ['python3', 'python']:
                        pre_pids.add(proc.info['pid'])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        # Launch the hotkey script
        script_path = os.path.join(self.script_dir, self.config.hotkey_script)
        
        if platform.system() == 'Windows':
            self.console.print("[green]Launching AHK script...[/green]")
            subprocess.Popen(
                [script_path],
                creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
                shell=True
            )
        else:
            self.console.print("[green]Launching Linux hotkey script...[/green]")
            # Make sure the script is executable
            os.chmod(script_path, 0o755)
            # Start the Python script
            subprocess.Popen(['python3', script_path], 
                            stdout=subprocess.DEVNULL, 
                            stderr=subprocess.DEVNULL, 
                            start_new_session=True)

        # Give it a moment to start
        time.sleep(1.0)
        
        # Find the new process
        post_pids = set()
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if platform.system() == 'Windows':
                    if proc.info['name'] == 'AutoHotkeyU64.exe':
                        post_pids.add(proc.info['pid'])
                else:
                    if proc.info['name'] in ['python3', 'python']:
                        if proc.info['cmdline'] is not None and self.config.hotkey_script in ' '.join(proc.info['cmdline']):
                            post_pids.add(proc.info['pid'])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        # Store the PID of the new process
        new_pids = post_pids - pre_pids
        if len(new_pids) == 1:
            self.ahk_pid = new_pids.pop()
            self.console.print(f"[green]Detected new hotkey script PID: {self.ahk_pid}[/green]")
        else:
            self.console.print("[red]Could not detect a single new hotkey script PID. No PID stored.[/red]")
            self.ahk_pid = None
    
    def start(self) -> None:
        """Start the application."""
        self._display_info()
        self._kill_leftover_hotkey_process()
        self._start_hotkey_script()        
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
        # Check if any transcription is already running
        if self.realtime_handler.is_running:
            self.console.print("[bold yellow]Cannot start recording while real-time transcription is running.[/bold yellow]")
            self.tray.flash_white('gray', self.config.send_enter)
            return

        if self.static_processor.is_transcribing():
            self.console.print("[bold yellow]Cannot start recording while static transcription is running.[/bold yellow]")
            self.tray.flash_white('gray', self.config.send_enter)
            return

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
    
    def reset_transcription(self) -> None:
        """Reset any in-progress transcription and return to ready state."""
        self.console.print("[bold yellow]RESET_TRANSCRIPTION command received[/bold yellow]")

        # Check if live recording is in progress
        if self.recorder.recording:
            # Flash white icon briefly for visual feedback on reset
            self.tray.flash_white('gray', self.config.send_enter)
            
            self.console.print("[bold yellow]Resetting live transcription...[/bold yellow]")
            # Stop recording without transcribing
            self.recorder.recording = False
            if self.recorder.recording_thread:
                self.recorder.recording_thread.join()
            
            # Clean up resources
            self.recorder._cleanup_resources()
            
            # Clear partial transcripts
            self.recorder.partial_transcripts.clear()
            self.recorder.buffer.clear()

            self.console.print("[green]Live transcription reset. Ready for new commands.[/green]")
        else:
            # Check if static transcription is in progress
            if self.static_processor.is_transcribing():
                # Request abort for static transcription
                self.static_processor.request_abort()
            else:
                self.console.print("[yellow]No transcription in progress to reset.[/yellow]")
                
                # Flash the tray icon white briefly to acknowledge the command
                self.tray.flash_white('gray', self.config.send_enter)
    
    def transcribe_static(self) -> None:
        """Transcribe a static audio file."""

        # Check if any transcription is already running
        if self.recorder.recording:
            self.console.print("[bold yellow]Cannot start static transcription while recording is in progress.[/bold yellow]")
            self.tray.flash_white('gray', self.config.send_enter)
            return
            
        if self.realtime_handler.is_running:
            self.console.print("[bold yellow]Cannot start static transcription while real-time transcription is running.[/bold yellow]")
            self.tray.flash_white('gray', self.config.send_enter)
            return

        self.static_processor.transcribe_file()
    
    def toggle_realtime_transcription(self) -> None:
        """Toggle real-time transcription on/off."""
        self.console.print("[bold yellow]TOGGLE_REALTIME_TRANSCRIPTION command received[/bold yellow]")

        # If real-time transcription is already running, stop it
        if self.realtime_handler.is_running:
            self.realtime_handler.stop()
            return

        # Check if any other transcription is running
        if self.recorder.recording:
            self.console.print("[bold yellow]Cannot start real-time transcription while recording is in progress.[/bold yellow]")
            self.tray.flash_white('gray', self.config.send_enter)
            return

        if self.static_processor.is_transcribing():
            self.console.print("[bold yellow]Cannot start real-time transcription while static transcription is running.[/bold yellow]")
            self.tray.flash_white('gray', self.config.send_enter)
            return

        self.realtime_handler.toggle()

    def toggle_audio_source(self) -> None:
        """Toggle between system audio and microphone for real-time transcription."""
        self.console.print("[bold yellow]TOGGLE_AUDIO_SOURCE command received[/bold yellow]")
        self.realtime_handler.toggle_audio_source()    
    
    def shutdown(self) -> None:
        """Shut down the application gracefully."""
        self.console.print("[bold red]Shutting down gracefully...[/bold red]")

        # Stop real-time transcription if running
        if hasattr(self, 'realtime_handler'):
            self.realtime_handler.stop()

        # Save configuration before exiting
        try:
            self.config.save_to_file()
            self.console.print("[green]Configuration saved to file.[/green]")
        except Exception as e:
            self.console.print(f"[red]Failed to save configuration: {e}[/red]")

        # Kill AHK script if we know its PID
        if self.ahk_pid is not None:
            self.console.print(f"[bold red]Killing hotkey script with PID={self.ahk_pid}[/bold red]")
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