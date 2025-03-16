# Speech to Text with Faster Whisper
# - Handles both real-time transcription and static file transcription
# - Supports language toggling and voice activity detection

import collections
import numpy as np
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
import signal
import ctypes

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

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
@dataclass
class Config:
    # Audio settings
    use_system_audio: bool = False
    input_device_index: int = 3
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
    send_enter: bool = False
    
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
            'yellow': self._create_circle_icon(size, (255, 255, 0, 255), black_outline),
            'white': self._create_circle_icon(size, (255, 255, 255, 255), black_outline)
        }
        
        # Create icons with green outline (send_enter=True)
        self.icons_green = {
            'gray': self._create_circle_icon(size, (128, 128, 128, 255), green_outline),
            'red': self._create_circle_icon(size, (255, 0, 0, 255), green_outline),
            'blue': self._create_circle_icon(size, (0, 128, 255, 255), green_outline),
            'yellow': self._create_circle_icon(size, (255, 255, 0, 255), green_outline),
            'white': self._create_circle_icon(size, (255, 255, 255, 255), green_outline)
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

    def flash_white(self, final_color: str, send_enter: bool = False) -> None:
        """Flash the tray icon white and then return to specified color."""
        # Set to white
        self.set_color('white', send_enter)
        
        # Schedule changing back to final color after 0.5 seconds
        def revert_icon():
            time.sleep(0.5)
            self.set_color(final_color, send_enter)
        
        threading.Thread(target=revert_icon, daemon=True).start()

# --------------------------------------------------------------------------------------
# Language Selection Dialog
# --------------------------------------------------------------------------------------
class LanguageSelector:
    def __init__(self, config: Config, console: Console):
        self.config = config
        self.console = console
        
        # All supported languages from Whisper
        self.languages = {
            "en": "english", "zh": "chinese", "de": "german", "es": "spanish",
            "ru": "russian", "ko": "korean", "fr": "french", "ja": "japanese",
            "pt": "portuguese", "tr": "turkish", "pl": "polish", "ca": "catalan",
            "nl": "dutch", "ar": "arabic", "sv": "swedish", "it": "italian",
            "id": "indonesian", "hi": "hindi", "fi": "finnish", "vi": "vietnamese",
            "he": "hebrew", "uk": "ukrainian", "el": "greek", "ms": "malay",
            "cs": "czech", "ro": "romanian", "da": "danish", "hu": "hungarian",
            "ta": "tamil", "no": "norwegian", "th": "thai", "ur": "urdu",
            "hr": "croatian", "bg": "bulgarian", "lt": "lithuanian", "la": "latin",
            "mi": "maori", "ml": "malayalam", "cy": "welsh", "sk": "slovak",
            "te": "telugu", "fa": "persian", "lv": "latvian", "bn": "bengali",
            "sr": "serbian", "az": "azerbaijani", "sl": "slovenian", "kn": "kannada",
            "et": "estonian", "mk": "macedonian", "br": "breton", "eu": "basque",
            "is": "icelandic", "hy": "armenian", "ne": "nepali", "mn": "mongolian",
            "bs": "bosnian", "kk": "kazakh", "sq": "albanian", "sw": "swahili",
            "gl": "galician", "mr": "marathi", "pa": "punjabi", "si": "sinhala",
            "km": "khmer", "sn": "shona", "yo": "yoruba", "so": "somali",
            "af": "afrikaans", "oc": "occitan", "ka": "georgian", "be": "belarusian",
            "tg": "tajik", "sd": "sindhi", "gu": "gujarati", "am": "amharic",
            "yi": "yiddish", "lo": "lao", "uz": "uzbek", "fo": "faroese",
            "ht": "haitian creole", "ps": "pashto", "tk": "turkmen", "nn": "nynorsk",
            "mt": "maltese", "sa": "sanskrit", "lb": "luxembourgish", "my": "myanmar",
            "bo": "tibetan", "tl": "tagalog", "mg": "malagasy", "as": "assamese",
            "tt": "tatar", "haw": "hawaiian", "ln": "lingala", "ha": "hausa",
            "ba": "bashkir", "jw": "javanese", "su": "sundanese", "yue": "cantonese"
        }
        
        # Language code to display name mapping
        self.language_display = {}
        for code, name in self.languages.items():
            self.language_display[code] = f"{name.title()} ({code})"
    
    def show_dialog(self):
        """Show a language selection dialog with search and autocomplete."""
        self.console.print("[bold blue]Opening language selection dialog...[/bold blue]")

        # Create the main dialog window
        root = tkinter.Tk()
        root.title("Select Language")

        # Dark theme colors
        bg_color = "#333333"        # Dark grey background
        text_color = "#FFFFFF"      # White text
        entry_bg = "#555555"        # Slightly lighter grey for input fields
        button_bg = "#444444"       # Medium grey for buttons
        highlight_color = "#007ACC" # Blue highlight color

        # Make dialog shorter
        root.geometry("400x380")    # Reduced height from 500 to 380
        root.configure(bg=bg_color)

        # Make dialog appear on top
        root.attributes('-topmost', True)

        # Center the window on screen
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x = (screen_width - 400) // 2
        y = (screen_height - 380) // 2
        root.geometry(f"400x380+{x}+{y}")

        # Frame for search
        search_frame = tkinter.Frame(root, bg=bg_color)
        search_frame.pack(pady=10, padx=20, fill="x")

        # Search label
        search_label = tkinter.Label(search_frame, text="Search Language:", bg=bg_color, fg=text_color, font=("Arial", 12))
        search_label.pack(anchor="w", pady=(0, 5))

        # Current language label
        current_lang_name = self.languages.get(self.config.language, "Unknown")
        current_lang_label = tkinter.Label(
            search_frame, 
            text=f"Current: {current_lang_name.title()} ({self.config.language})", 
            bg=bg_color, 
            fg=text_color, 
            font=("Arial", 10, "italic")
        )
        current_lang_label.pack(anchor="w", pady=(0, 10))

        # Search entry
        search_var = tkinter.StringVar()
        search_entry = tkinter.Entry(
            search_frame, 
            textvariable=search_var, 
            font=("Arial", 12), 
            width=30, 
            bg=entry_bg, 
            fg=text_color, 
            insertbackground=text_color  # Cursor color
        )
        search_entry.pack(fill="x", pady=5)
        search_entry.focus_set()

        # Listbox for languages
        listbox_frame = tkinter.Frame(root, bg=bg_color)
        listbox_frame.pack(pady=10, padx=20, fill="both", expand=True)

        scrollbar = tkinter.Scrollbar(listbox_frame, bg=button_bg, troughcolor=bg_color)
        scrollbar.pack(side="right", fill="y")

        listbox = tkinter.Listbox(
            listbox_frame, 
            yscrollcommand=scrollbar.set, 
            font=("Arial", 12), 
            selectmode="single",
            height=10,  # Reduced from 15 to 10 to make the dialog shorter
            bg=entry_bg,
            fg=text_color,
            selectbackground=highlight_color,
            selectforeground=text_color
        )
        listbox.pack(side="left", fill="both", expand=True)

        scrollbar.config(command=listbox.yview)

        # Populate listbox with all languages initially
        sorted_display = sorted(self.language_display.values())
        for display_name in sorted_display:
            listbox.insert(tkinter.END, display_name)

        # Result variable to store the selected language code
        result = {"selected_code": None}

        # Function to update listbox based on search
        def update_listbox(*args):
            search_text = search_var.get().lower()
            listbox.delete(0, tkinter.END)

            # First search by language code
            matching_codes = [code for code, name in self.languages.items() 
                             if search_text in code.lower()]

            # Then search by language name
            matching_names = [code for code, name in self.languages.items() 
                             if search_text in name.lower() and code not in matching_codes]

            # Combine both lists
            matching_codes.extend(matching_names)

            # Display top matches (max 30)
            count = 0
            for code in matching_codes[:30]:
                listbox.insert(tkinter.END, self.language_display[code])
                count += 1

            # If there are matches, select the first one
            if count > 0:
                listbox.selection_set(0)

        # Bind search entry to update function
        search_var.trace("w", update_listbox)

        # Function to handle selection
        def on_select(event=None):
            try:
                # Get the selected display name from listbox
                selection_idx = listbox.curselection()[0]
                selection = listbox.get(selection_idx)

                # Extract language code from display name
                code = selection.split("(")[1].split(")")[0]

                # Update result
                result["selected_code"] = code
                root.destroy()
            except (IndexError, Exception) as e:
                self.console.print(f"[red]Selection error: {e}[/red]")

        # Function to handle cancel
        def on_cancel():
            root.destroy()

        # Bind double-click to select
        listbox.bind("<Double-1>", on_select)

        # Bind Enter key to select
        root.bind("<Return>", on_select)

        # Bind Escape key to cancel
        root.bind("<Escape>", lambda e: on_cancel())

        # Create buttons
        button_frame = tkinter.Frame(root, bg=bg_color)
        button_frame.pack(pady=15, padx=20, fill="x")

        select_button = tkinter.Button(
            button_frame, 
            text="Select", 
            command=on_select,
            bg="#4CAF50",
            fg=text_color,
            font=("Arial", 12),
            width=10,
            activebackground="#3e8e41",
            activeforeground=text_color
        )
        select_button.pack(side="right", padx=5)

        cancel_button = tkinter.Button(
            button_frame, 
            text="Cancel", 
            command=on_cancel,
            bg="#f44336",
            fg=text_color,
            font=("Arial", 12),
            width=10,
            activebackground="#d32f2f",
            activeforeground=text_color
        )
        cancel_button.pack(side="right", padx=5)

        # Run the dialog
        root.mainloop()

        # Check if a language was selected
        selected_code = result["selected_code"]
        if selected_code:
            return selected_code
        else:
            return None

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
    
    def transcribe_audio_data(self, audio):
        """Transcribe audio data directly."""
        try:
            segments, info = self.model.transcribe(
                audio, 
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
            self.console.print(f"[bold red]Transcription failed: {e}[/bold red]")
            return ""
    
    def toggle_language(self) -> None:
        """Toggle between Greek and English."""
        old_lang = self.config.language
        old_task = self.config.task

        # Toggle the language
        self.config.language = "en" if self.config.language == "el" else "el"

        # Always use transcribe for English and Greek
        self.config.task = "transcribe"

        # Log the changes
        task_msg = ""
        if old_task != self.config.task:
            task_msg = f" and task from '{old_task}' to '{self.config.task}'"

        self.console.print(f"[yellow]Language toggled from {old_lang} to {self.config.language}{task_msg}[/yellow]")
    
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
        self.abort_static_transcription = False
        self.transcription_thread = None
        self.static_transcription_lock = threading.Lock()
    
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
            os.path.join(self.script_dir, "temp_static_file.wav"),
            os.path.join(self.script_dir, "temp_static_silence_removed.wav")
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

# --------------------------------------------------------------------------------------
# Real-time Transcription Handler
# --------------------------------------------------------------------------------------
class RealtimeTranscriptionHandler:
    def __init__(self, config: Config, console: Console, transcriber: Transcriber, tray: TrayManager):
        self.config = config
        self.console = console
        self.transcriber = transcriber
        self.tray = tray
        
        # Real-time transcription state
        self.is_running = False
        self.thread = None
        self.stop_event = threading.Event()
        
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
        self.realtime_model_name = "deepdml/faster-whisper-large-v3-turbo-ct2"
        
        # Audio input stream
        self.audio = None
        self.stream = None
    
    def _load_realtime_model(self):
        """Lazy-load the real-time transcription model."""
        if not self.realtime_model_loaded:
            self.console.print(f"[bold green]Loading real-time transcription model: {self.realtime_model_name}[/bold green]")
            
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                compute_type = "float16" if device == "cuda" else "float32"
                
                self.realtime_model = WhisperModel(
                    self.realtime_model_name,
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
            'input_device_index': self.config.input_device_index if self.config.use_system_audio else None
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
        """Toggle between system audio and microphone."""
        was_running = self.is_running
        if was_running:
            self.stop()
        
        self.config.use_system_audio = not self.config.use_system_audio
        source_str = "system audio" if self.config.use_system_audio else "microphone"
        self.console.print(f"[cyan]Toggled audio source to: {source_str}[/cyan]")
        
        # Update the tray icon to reflect the new state
        self.tray.flash_white('gray', self.config.send_enter)
        
        # Restart if it was running
        if was_running:
            self.start()
    
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
                            # Speech just started
                            self.console.print("[cyan]Speech detected[/cyan]")
                            self.is_speech_active = True
                            silence_start_time = 0
                            
                        # Add data to current segment
                        current_segment.append(data)
                    else:
                        # No speech in this chunk
                        if self.is_speech_active:
                            # We were in active speech, track silence
                            if silence_start_time == 0:
                                silence_start_time = current_time
                                
                            # Still add the silent audio to maintain context
                            current_segment.append(data)
                            
                            # Check if silence duration exceeds threshold
                            if (current_time - silence_start_time) * 1000 >= self.silence_threshold_ms:
                                # End of speech segment detected
                                self.console.print("[cyan]End of speech segment detected[/cyan]")
                                self.is_speech_active = False
                                
                                # Process this segment if not empty
                                if current_segment:
                                    # Add to accumulated speech
                                    accumulated_speech.extend(current_segment)
                                    current_segment = []
                                    
                                    # Convert to numpy array for processing
                                    audio_data = np.frombuffer(b''.join(accumulated_speech), dtype=np.int16)
                                    audio_float = audio_data.astype(np.float32) / 32768.0
                                    
                                    # Transcribe using the real-time model
                                    text = ""
                                    if self.realtime_model_loaded:
                                        try:
                                            segments, info = self.realtime_model.transcribe(
                                                audio_float, 
                                                language=self.config.language, 
                                                task=self.config.task
                                            )
                                            text = "".join(segment.text for segment in segments)
                                        except Exception as e:
                                            self.console.print(f"[red]Real-time model transcription error: {e}[/red]")
                                            # Fall back to main transcriber
                                            text = self.transcriber.transcribe_audio_data(audio_float)
                                    else:
                                        # Use main transcriber as fallback
                                        text = self.transcriber.transcribe_audio_data(audio_float)
                                    
                                    # Process and display the result
                                    if text:
                                        self._process_text(text)
                                    
                                    # Clear accumulated speech but keep a small context window
                                    # to maintain continuity between utterances
                                    keep_frames = min(20, len(accumulated_speech))
                                    accumulated_speech = accumulated_speech[-keep_frames:]
                        
                        # If not in active speech and silence continues, just discard this chunk
                    
                    # Small sleep to prevent CPU overuse
                    time.sleep(0.01)
                    
                except Exception as e:
                    self.console.print(f"[bold red]Error reading audio: {e}[/bold red]")
                    time.sleep(0.1)
                
        except Exception as e:
            self.console.print(f"[bold red]Error in real-time transcription: {e}[/bold red]")
        finally:
            self._cleanup_audio()

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
        elif command == "OPEN_LANGUAGE_MENU":
            self.app.open_language_menu()
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
        elif command == "TOGGLE_AUDIO_SOURCE":
            self.app.toggle_audio_source()
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
        self.config = Config()
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.ahk_pid = None
        
        # Initialize components
        self.tray = TrayManager(self.console)
        self.transcriber = Transcriber(self.config, self.console)
        self.recorder = AudioRecorder(self.config, self.console, self.transcriber, self.tray)
        self.static_processor = StaticFileProcessor(self.config, self.console, self.transcriber, self.tray)
        self.realtime_handler = RealtimeTranscriptionHandler(self.config, self.console, self.transcriber, self.tray)
        self.language_selector = LanguageSelector(self.config, self.console)
        self.server = CommandServer(self)
    
    def _display_info(self) -> None:
        """Display startup information."""
        panel_content = (
            f"[bold yellow]Model[/bold yellow]: {self.transcriber.model_id}\n"
            f"[bold yellow]Hotkeys[/bold yellow]: Controlled by AutoHotKey script '{self.config.hotkey_script}'\n"
            " F2  -> Toggle language\n"
            " F3  -> Start recording\n"
            " F4  -> Stop & transcribe\n"
            " F5  -> Toggle enter\n"
            " F6  -> Reset transcription\n"
            " F7  -> Quit\n"
            " F8  -> Toggle audio source (system audio/microphone)\n"
            " F9  -> Toggle real-time transcription\n"
            " F10 -> Static file transcription\n"
            f"[bold yellow]Language[/bold yellow]: {self.config.language}\n"
            f"[bold yellow]Audio Source[/bold yellow]: {'System Audio' if self.config.use_system_audio else 'Microphone'}"
        )
        panel = Panel(panel_content, title="Information", border_style="green")
        self.console.print(panel)
    
    def open_language_menu(self) -> None:
        """Open the language selection menu dialog."""
        self.console.print("[bold yellow]OPEN_LANGUAGE_MENU command received[/bold yellow]")
        
        # Pause any real-time transcription if it's running
        realtime_was_running = False
        if hasattr(self, 'realtime_handler') and self.realtime_handler.is_running:
            realtime_was_running = True
            self.realtime_handler.stop()
        
        # Show the language selection dialog
        selected_language = self.language_selector.show_dialog()
        
        if selected_language:
            old_lang = self.config.language
            old_task = self.config.task
            
            self.config.language = selected_language
            language_name = self.language_selector.languages.get(selected_language, "Unknown")
            
            # Auto-switch task based on language
            if selected_language not in ["en", "el"]:
                self.config.task = "translate"
            else:
                self.config.task = "transcribe"
            
            # Log the changes
            task_msg = ""
            if old_task != self.config.task:
                task_msg = f" and task from '{old_task}' to '{self.config.task}'"
                
            self.console.print(f"[yellow]Language changed from {old_lang} to {selected_language} ({language_name}){task_msg}[/yellow]")
            
            # Flash the tray icon for visual feedback
            self.tray.flash_white('gray', self.config.send_enter)
        else:
            self.console.print("[yellow]Language selection cancelled or no language selected.[/yellow]")
        
        # Resume real-time transcription if it was running
        if realtime_was_running:
            self.realtime_handler.start()

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
        self.static_processor.transcribe_file()
    
    def toggle_realtime_transcription(self) -> None:
        """Toggle real-time transcription on/off."""
        self.console.print("[bold yellow]TOGGLE_REALTIME_TRANSCRIPTION command received[/bold yellow]")
        self.realtime_handler.toggle()

    def toggle_audio_source(self) -> None:
        """Toggle between system audio and microphone."""
        self.console.print("[bold yellow]TOGGLE_AUDIO_SOURCE command received[/bold yellow]")
        self.realtime_handler.toggle_audio_source()    
    
    def shutdown(self) -> None:
        """Shut down the application gracefully."""
        self.console.print("[bold red]Shutting down gracefully...[/bold red]")

        # Stop real-time transcription if running
        if hasattr(self, 'realtime_handler'):
            self.realtime_handler.stop()

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