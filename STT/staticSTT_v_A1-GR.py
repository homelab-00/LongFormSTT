EXTENDED_LOGGING = False

if __name__ == '__main__':

    import sys
    import threading
    import time

    if EXTENDED_LOGGING:
        import logging
        logging.basicConfig(level=logging.DEBUG)

    from rich.console import Console
    from rich.live import Live
    console = Console()
    console.print("System initializing, please wait")

    import os
    import numpy as np

    # Initialize Rich Console and Live
    live = Live(console=console, refresh_per_second=10, screen=False)
    live.start()

    # Events to signal threads to exit
    exit_event = threading.Event()

    # Recorder configuration parameters (kept as per your request)
    end_of_sentence_detection_pause = 0.45
    unknown_sentence_detection_pause = 0.7
    mid_sentence_detection_pause = 2.0

    recorder_config = {
        'spinner': False,
        'model': 'large-v3',  # Systran/faster-whisper-large-v3 or ...
        'language': 'el',
        'silero_sensitivity': 0.05,
        'webrtc_sensitivity': 3,
        'post_speech_silence_duration': unknown_sentence_detection_pause,
        'min_length_of_recording': 1.1,
        'min_gap_between_recordings': 0,
        'enable_realtime_transcription': False,
        'realtime_processing_pause': 0.02,
        'silero_deactivity_detection': True,
        'early_transcription_on_silence': 0,
        'beam_size': 5,
        'beam_size_realtime': 5,
        'no_log_file': True,
        'device': 'cuda',          # Device configuration
        'compute_type': 'float16',  # Compute type configuration
        'initial_prompt': (
            "End incomplete sentences with ellipses.\n"
            "Examples:\n"
            "Complete: The sky is blue.\n"
            "Incomplete: When the sky...\n"
            "Complete: She walked home.\n"
            "Incomplete: Because he...\n"
        )
    }

    if EXTENDED_LOGGING:
        recorder_config['level'] = logging.DEBUG

    # Function to transcribe the existing audio file
    def transcribe_audio_file():
        console.print("[bold green]Starting transcription of existing audio file...[/bold green]")

        audio_filename = 'recording.wav'  # The name of the audio file to transcribe
        output_filename = 'transcribed_audio.txt'  # The name of the output transcription file

        if not os.path.exists(audio_filename):
            console.print(f"[bold red]Audio file '{audio_filename}' not found in the current directory.[/bold red]")
            return

        try:
            # Load the audio file
            console.print("Loading audio file...")
            import soundfile as sf
            audio_data, sample_rate = sf.read(audio_filename)
            if sample_rate != 16000:
                console.print(f"[bold yellow]Warning: Expected sample rate of 16000 Hz, but got {sample_rate} Hz.[/bold yellow]")
            if audio_data.ndim > 1:
                console.print(f"[bold yellow]Warning: Audio file has more than one channel; using the first channel.[/bold yellow]")
                audio_data = audio_data[:, 0]

            # Convert audio data to the expected format
            audio_array = audio_data.astype(np.float32)

            # Transcribe the audio
            console.print("Transcribing audio...")

            try:
                from faster_whisper import WhisperModel
            except ImportError:
                console.print("[bold red]faster_whisper is not installed. Please install it to use transcription.[/bold red]")
                return

            # Load the model using recorder_config
            model_size = recorder_config['model']
            device = recorder_config['device']
            compute_type = recorder_config['compute_type']

            console.print("Loading transcription model... This may take a moment.")
            try:
                model = WhisperModel(model_size, device=device, compute_type=compute_type)
            except Exception as e:
                console.print(f"[bold red]Failed to load transcription model: {e}[/bold red]")
                return

            # Transcribe the audio
            try:
                segments, info = model.transcribe(audio_array, beam_size=recorder_config['beam_size'], language=recorder_config['language'])
                transcription = ' '.join([segment.text for segment in segments]).strip()
            except Exception as e:
                console.print(f"[bold red]Error during transcription: {e}[/bold red]")
                return

            # Save the transcription to a txt file
            with open(output_filename, 'w', encoding='utf-8') as f:
                f.write(transcription)

            console.print(f"[bold green]Transcription completed. Saved to '{output_filename}'.[/bold green]")

            # Display the transcription
            console.print("Transcription:")
            console.print(f"[bold cyan]{transcription}[/bold cyan]")

        except Exception as e:
            console.print(f"[bold red]Error during transcription: {e}[/bold red]")

    # Start the transcription process
    transcribe_audio_file()

    # Stop the Live console
    live.stop()

    console.print("[bold green]Script completed.[/bold green]")
    sys.exit(0)
