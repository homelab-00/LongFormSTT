# static_file_processor.py
#
# Processes and transcribes pre-recorded audio/video files
#
# This module:
# - Handles selecting audio/video files via a file dialog
# - Converts various media formats to 16kHz mono WAV using FFmpeg
# - Applies Voice Activity Detection (WebRTC VAD) to remove silence
# - Transcribes the processed audio using the transcription engine
# - Saves transcription results alongside the original file
# - Manages temporary files and resource cleanup
# - Provides methods to abort transcription in progress
# - Updates system tray to indicate transcription status
#
# This component allows transcription of existing media files
# rather than just real-time microphone input

import os
import sys
import threading
import subprocess
import shutil
import wave
import ctypes
import time
from typing import Optional
import tkinter
from tkinter import filedialog
from rich.panel import Panel

# Optional dependencies
try:
    import webrtcvad
    WEBRTC_VAD_AVAILABLE = True
except ImportError:
    WEBRTC_VAD_AVAILABLE = False

class StaticFileProcessor:
    def __init__(self, config, console, transcriber, tray):
        self.config = config
        self.console = console
        self.transcriber = transcriber
        self.tray = tray
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.temp_dir = os.path.join(self.script_dir, "temp_audio")
        
        # Ensure temp_audio directory exists
        if not os.path.exists(self.temp_dir):
            try:
                os.makedirs(self.temp_dir)
                self.console.print(f"[green]Created temporary audio directory: {self.temp_dir}[/green]")
            except Exception as e:
                self.console.print(f"[red]Failed to create temp directory: {e}[/red]")
                # Fall back to script directory if temp directory creation fails
                self.temp_dir = self.script_dir

        self.transcription_thread = None
        self.static_transcription_lock = threading.Lock()
        self.abort_static_transcription = False
    
    def is_transcribing(self) -> bool:
        """Check if a static transcription is currently in progress."""
        return self.transcription_thread is not None and self.transcription_thread.is_alive()

    def _terminate_thread(self, thread):
        """Force terminate a thread using ctypes (Windows-specific solution)."""
        if not thread.is_alive():
            return
            
        # Use ctypes to terminate thread (Windows-specific)
        if sys.platform == "win32":
            thread_id = thread.ident
            if thread_id:
                res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                    ctypes.c_long(thread_id), 
                    ctypes.py_object(SystemExit)
                )
                if res > 1:
                    # If more than one thread was affected, undo the damage
                    ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread_id), None)
                    self.console.print("[red]Failed to terminate thread correctly.[/red]")
                return
        
        self.console.print("[red]Thread termination not supported on this platform.[/red]")
    
    def request_abort(self) -> None:
        """Immediately abort any in-progress static transcription."""
        with self.static_transcription_lock:
            if not self.is_transcribing():
                self.console.print("[yellow]No static transcription in progress to abort.[/yellow]")
                return
                
            # Set the abort flag
            self.abort_static_transcription = True
            
            # Immediately set the tray icon to gray to indicate we're stopping
            self.tray.flash_white('gray', self.config.send_enter)
            
            # Store a reference to the thread for forced termination
            thread_to_terminate = self.transcription_thread
        
        self.console.print("[bold yellow]Static transcription abort requested.[/bold yellow]")
        self.console.print("[bold yellow]Forcing immediate termination of transcription...[/bold yellow]")
        
        # First try to terminate the thread forcibly
        self._terminate_thread(thread_to_terminate)
        
        # Give it a short grace period
        grace_period = 0.5  # seconds
        start_time = time.time()
        # Use a copy of the thread reference to avoid race conditions
        thread_ref = thread_to_terminate
        while time.time() - start_time < grace_period and self.is_transcribing():
            time.sleep(0.1)
        
        # If the thread is still running, we'll try to clean up as much as possible
        if self.is_transcribing():
            self.console.print("[red]Could not terminate transcription thread gracefully.[/red]")
            self.console.print("[yellow]Cleaning up resources and marking transcription as complete.[/yellow]")
            
            with self.static_transcription_lock:
                # Force the thread to be marked as None so is_transcribing() returns False
                self.transcription_thread = None
        else:
            self.console.print("[green]Transcription aborted successfully![/green]")
        
        # Ensure we clean up any temporary files
        self._cleanup_temp_files()
        
        # Ensure tray icon is reset to gray
        self.tray.set_color('gray', self.config.send_enter)
        
        self.console.print("[green]Reset complete. Ready for new commands.[/green]")

    def _transcribe_in_thread(self, file_path: str) -> None:
        """Perform transcription in a separate thread."""
        try:
            # Reset abort flag at start
            with self.static_transcription_lock:
                self.abort_static_transcription = False
            
            # Check for abort frequently
            def should_abort():
                with self.static_transcription_lock:
                    return self.abort_static_transcription
            
            # Step 1: Convert to WAV format if needed
            wav_path = self._ensure_wav_format(file_path)
            if not wav_path or not os.path.exists(wav_path):
                self.console.print("[bold red]Failed to convert audio file. Aborting.[/bold red]")
                self.tray.set_color('gray', self.config.send_enter)
                return
            
            # Check abort flag after conversion
            if should_abort():
                self.console.print("[bold yellow]Static transcription aborted after conversion.[/bold yellow]")
                self.tray.set_color('gray', self.config.send_enter)
                return
            
            # Step 2: Apply VAD to remove non-speech sections
            voice_wav = self._apply_vad(wav_path, aggressiveness=2)
            
            # Check abort flag after VAD
            if should_abort():
                self.console.print("[bold yellow]Static transcription aborted after VAD.[/bold yellow]")
                self.tray.set_color('gray', self.config.send_enter)
                return
            
            # Step 3: Transcribe the processed audio
            self.console.print("[blue]Beginning transcription with voice-only data...[/blue]")
            
            # Check if we should abort before starting transcription
            if should_abort():
                self.console.print("[bold yellow]Static transcription aborted before starting.[/bold yellow]")
                self.tray.set_color('gray', self.config.send_enter)
                return
                
            final_text = self.transcriber.transcribe(voice_wav)
            
            # Check abort flag after transcription
            if should_abort():
                self.console.print("[bold yellow]Static transcription completed but results discarded due to abort request.[/bold yellow]")
                self.tray.set_color('gray', self.config.send_enter)
                return
            
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
        
        except SystemExit:
            self.console.print("[yellow]Transcription thread was terminated by user request.[/yellow]")
        except Exception as e:
            self.console.print(f"[bold red]Static transcription failed: {e}[/bold red]")
        
        finally:
            self.tray.set_color('gray', self.config.send_enter)
            with self.static_transcription_lock:
                self.transcription_thread = None

    def _cleanup_temp_files(self) -> None:
        """Remove temporary files used for static transcription."""
        temp_files = [
            os.path.join(self.temp_dir, "temp_static_file.wav"),
            os.path.join(self.temp_dir, "temp_static_silence_removed.wav")
        ]
        
        for f in temp_files:
            if os.path.exists(f):
                try:
                    os.remove(f)
                    self.console.print(f"[yellow]Deleted temp file: {os.path.basename(f)}[/yellow]")
                except Exception as e:
                    self.console.print(f"[red]Failed to delete {os.path.basename(f)}: {e}[/red]")
    
    def _ensure_wav_format(self, input_path: str) -> Optional[str]:
        """Convert input file (audio or video) to 16kHz mono WAV."""
        temp_wav = os.path.join(self.temp_dir, "temp_static_file.wav")

        # Check if the file is already a WAV in the correct format
        try:
            with wave.open(input_path, 'rb') as wf:
                channels = wf.getnchannels()
                rate = wf.getframerate()
                if channels == 1 and rate == 16000:
                    self.console.print("[blue]No conversion needed, copying to temp file.[/blue]")
                    shutil.copy(input_path, temp_wav)
                    return temp_wav
        except wave.Error:
            # Not a valid WAV file, needs conversion
            pass
        except Exception as e:
            self.console.print(f"[yellow]File check error: {e}. Will try conversion.[/yellow]")

        # Get file extension to determine if it's video or audio
        _, ext = os.path.splitext(input_path)
        ext = ext.lower()

        # Common video extensions
        video_exts = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
        is_video = ext in video_exts

        # Convert using FFmpeg
        if is_video:
            self.console.print(f"[cyan]Converting video file '{input_path}' to 16kHz mono WAV[/cyan]")
        else:
            self.console.print(f"[cyan]Converting audio file '{input_path}' to 16kHz mono WAV[/cyan]")

        try:
            subprocess.run([
                "ffmpeg",
                "-y",              # Overwrite output file if it exists
                "-i", input_path,  # Input file
                "-vn",             # Skip video stream (needed for video files)
                "-ac", "1",        # Mono
                "-ar", "16000",    # 16kHz
                temp_wav
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            self.console.print("[cyan]Conversion successful.[/cyan]")
            return temp_wav
        except Exception as e:
            self.console.print(f"[bold red]FFmpeg conversion error: {e}[/bold red]")
            return None
    
    def _apply_vad(self, in_wav_path: str, aggressiveness: int = 2) -> str:
        """Apply Voice Activity Detection to keep only speech frames."""
        if not WEBRTC_VAD_AVAILABLE:
            self.console.print("[red]webrtcvad not installed. Skipping VAD.[/red]")
            return in_wav_path

        out_wav = os.path.join(self.temp_dir, "temp_static_silence_removed.wav")

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

        # Check if already transcribing
        if self.is_transcribing():
            self.console.print("[bold yellow]Already transcribing a file. Please wait or reset.[/bold yellow]")
            return

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

        # Start transcription in a separate thread
        with self.static_transcription_lock:
            self.transcription_thread = threading.Thread(
                target=self._transcribe_in_thread,
                args=(file_path,),
                daemon=True
            )
            self.transcription_thread.start()