# audio_recorder.py
import numpy as np
import time
import pyaudio
import wave
import os
import threading
import pyperclip
import keyboard
import struct
import glob
from rich.console import Console
from rich.panel import Panel

class AudioRecorder:
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
        temp_files = glob.glob(os.path.join(self.temp_dir, "temp_audio_file*.wav"))
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
        first_file = os.path.join(self.temp_dir, f"temp_audio_file{self.current_chunk_index}.wav")
        self.active_filename = first_file

        try:
            # Open audio stream
            stream_params = {
                'format': self.config.format,
                'channels': self.config.channels,
                'rate': self.config.rate,
                'input': True,
                'frames_per_buffer': self.config.chunk,
                'input_device_index': self.config.input_device_index if self.config.longform_use_system_audio else None
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
            try:
                self.active_wave_file.close()
            except Exception as e:
                self.console.print(f"[red]Error closing wave file: {e}[/red]")
            self.active_wave_file = None
            
        if hasattr(self, 'stream') and self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                self.console.print(f"[red]Error closing audio stream: {e}[/red]")
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
        new_filename = os.path.join(self.temp_dir, f"temp_audio_file{self.current_chunk_index}.wav")
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