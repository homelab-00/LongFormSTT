# transcription_engine.py
import torch
import re
from rich.console import Console
from faster_whisper import WhisperModel

class Transcriber:
    def __init__(self, config, console: Console, model_id: str = None):
        self.config = config
        self.console = console
        
        # Initialize the model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id if model_id else "Systran/faster-whisper-large-v3"
        self.model = WhisperModel(
            self.model_id, 
            device=self.device, 
            compute_type="float16" if self.device == "cuda" else "float32"
        )
    
    def transcribe_audio_data(self, audio):
        """Transcribe audio data directly."""
        try:
            # FIX: Use a separate variable for real-time language so we
            # don't accidentally pick up the long-form setting:
            stt_language = self.config.realtime_language

            # Decide whether we do 'transcribe' or 'translate':
            #    faster_whisper automatically translates to EN if task="translate"
            #    (no need to pass translate_to=...).
            stt_task = "transcribe"
            if stt_language not in ["en", "el"]:
                stt_task = "translate"

            # Now just call transcribe without translate_to=...
            segments, info = self.model.transcribe(
                audio,
                language=stt_language,
                task=stt_task
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
    
    def transcribe(self, audio_path: str, use_realtime_language: bool = False) -> str:
        """Transcribe audio file and clean up the result."""
        try:
            language = self.config.realtime_language if use_realtime_language else self.config.longform_language
            task = "transcribe"
            if language not in ["en", "el"]:
                task = "translate"  # Translates to English automatically

            segments, info = self.model.transcribe(
                audio_path,
                language=language,
                task=task
            )

            text = "".join(s.text for s in segments)

            for pattern in self.config.hallucinations_regex:
                text = pattern.sub("", text)

            if text and text[0].isspace():
                text = text[1:]

            return text
        except Exception as e:
            self.console.print(f"[bold red]Transcription failed for {audio_path}: {e}[/bold red]")
            return ""