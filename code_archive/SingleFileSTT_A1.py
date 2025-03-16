# ==============================
# Static Audio Transcription Script
# ==============================
# This script prompts the user to select a static audio file, converts it to 16 kHz,
# single-channel WAV if necessary, transcribes the entire file at once using FasterWhisper,
# saves the final transcription to a .txt file, prints it to the terminal, removes the
# temporary converted file, and then exits.
#
# Dependencies:
#   pip install tkinter pydub faster-whisper rich
#   Also requires ffmpeg installed on the system for pydub to handle conversions.

import os
import re
import time
import torch
import shutil
import tkinter as tk
from tkinter import filedialog
from rich.console import Console
from rich.panel import Panel
from pydub import AudioSegment
from faster_whisper import WhisperModel

#####################
# Configuration
#####################

# Hallucination filtering with regex (optional)
HALLUCINATIONS_REGEX = [
    re.compile(r"\bΥπότιτλοι\s+AUTHORWAVE\b[^\w]*", re.IGNORECASE),
    re.compile(r"\bΣας\s+ευχαριστώ\b[^\w]*", re.IGNORECASE),
    # Add more patterns if needed
]

console = Console()
script_dir = os.path.dirname(os.path.abspath(__file__))

# Device selection
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "Systran/faster-whisper-large-v3"

# We keep the same language & task from the original script.
language = "el"
task = "transcribe"

#####################
# Main Transcription Logic
#####################

def choose_audio_file():
    """
    Open a GUI file dialog to let the user select an audio file.
    Returns the path of the chosen file or None if canceled.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    console.print("[blue]Please select an audio file to transcribe...[/blue]")
    selected_file = filedialog.askopenfilename(
        title="Select audio file",
        filetypes=[("Audio Files", "*.wav *.mp3 *.flac *.m4a *.ogg *.wma *.aac")]
    )
    root.destroy()

    if not selected_file:
        console.print("[bold red]No file selected. Exiting...[/bold red]")
        return None

    return selected_file


def convert_audio_if_needed(input_path, output_path):
    """
    Convert the input file to 16 kHz mono WAV if needed using pydub.
    Always writes to 'output_path'.
    """
    console.print(f"[green]Loading audio file:[/green] {input_path}")

    # Load audio with pydub (ffmpeg required)
    try:
        sound = AudioSegment.from_file(input_path)
    except Exception as e:
        console.print(f"[bold red]Error loading file: {e}[/bold red]")
        return False

    # Print some debug info
    console.print(
        f"[cyan]Original sample rate:[/cyan] {sound.frame_rate} Hz, "
        f"[cyan]channels:[/cyan] {sound.channels}, "
        f"[cyan]duration:[/cyan] {sound.duration_seconds:.2f} s"
    )

    # We want 16k, mono
    sound = sound.set_frame_rate(16000)
    sound = sound.set_channels(1)

    # Save as WAV
    try:
        sound.export(output_path, format="wav")
        console.print(f"[green]Successfully converted to:[/green] {output_path}")
    except Exception as e:
        console.print(f"[bold red]Error converting file: {e}[/bold red]")
        return False

    return True


def transcribe_audio(model, audio_path):
    """
    Transcribe the given WAV file with FasterWhisper.
    Remove hallucinations. Return the text.
    """
    console.print(f"[blue]Beginning transcription for file:[/blue] {audio_path}")

    try:
        segments, info = model.transcribe(audio_path, language=language, task=task)
        text = "".join(s.text for s in segments)
        console.print("[green]Transcription complete.[/green]")
    except Exception as e:
        console.print(f"[bold red]Error during transcription: {e}[/bold red]")
        return ""

    # Remove any known hallucinations
    for pattern in HALLUCINATIONS_REGEX:
        text = pattern.sub("", text)

    return text.strip()


def save_transcription_to_file(transcription, base_name):
    """
    Save the transcription text to a .txt file with the given base_name in the script directory.
    """
    txt_filename = os.path.join(script_dir, f"{base_name}.txt")
    try:
        with open(txt_filename, "w", encoding="utf-8") as f:
            f.write(transcription)
        console.print(f"[green]Transcription saved to:[/green] {txt_filename}")
    except Exception as e:
        console.print(f"[bold red]Error saving transcription: {e}[/bold red]")


def main():
    console.print("[bold magenta]Static File Transcription Script[/bold magenta]")
    console.print(
        Panel(
            f"[bold yellow]Model ID:[/bold yellow] {model_id}\n"
            f"[bold yellow]Language:[/bold yellow] {language}\n"
            "Always converting input to 16 kHz, mono WAV.",
            title="Configuration",
            border_style="green"
        )
    )

    # 1) Let user choose an audio file
    selected_file = choose_audio_file()
    if not selected_file:
        return  # No file chosen, we exit

    # 2) Prepare model
    console.print("[cyan]Loading FasterWhisper model...[/cyan]")
    model = WhisperModel(
        model_id,
        device=device,
        compute_type="float16" if device == "cuda" else "float32"
    )

    # 3) Convert file if needed (always do it here for simplicity)
    base_name = os.path.splitext(os.path.basename(selected_file))[0]
    temp_wav_path = os.path.join(script_dir, f"temp_{base_name}.wav")

    ok = convert_audio_if_needed(selected_file, temp_wav_path)
    if not ok:
        console.print("[bold red]Conversion failed. Exiting...[/bold red]")
        return

    # 4) Transcribe
    transcription = transcribe_audio(model, temp_wav_path)

    # 5) Print the final transcription
    panel = Panel(
        f"[bold magenta]{transcription}[/bold magenta]",
        title="Final Transcription",
        border_style="yellow"
    )
    console.print(panel)

    # 6) Save transcription to file
    save_transcription_to_file(transcription, base_name)

    # 7) Delete the temp file
    try:
        os.remove(temp_wav_path)
        console.print(f"[yellow]Deleted temp file:[/yellow] {temp_wav_path}")
    except Exception as e:
        console.print(f"[bold red]Failed to delete temp file: {e}[/bold red]")

    # 8) Done
    console.print("[bold green]All done. Exiting script.[/bold green]")


if __name__ == "__main__":
    main()
