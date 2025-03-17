import os
import sys
import time
import socket
import threading
import subprocess
import platform
import tkinter as tk
from tkinter import filedialog
import tempfile
from RealtimeSTT import AudioToTextRecorder

class STTApplication:
    def __init__(self):
        # Configuration
        self.model = "Systran/faster-whisper-large-v3"
        self.language = "el"  # Greek language
        self.task = "transcribe"
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.hotkey_script = "AHK_script-hotkeys_handling.ahk"
        
        # States
        self.is_running = True
        self.ahk_pid = None
        self.recording_active = False
        self.realtime_active = False
        self.static_transcription_active = False
        
        # Initialize main recorder
        self.initialize_recorder()
        
        # Create temp directory for file processing
        self.temp_dir = os.path.join(tempfile.gettempdir(), "stt_temp")
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
        
        # Set up command server
        self.server = CommandServer(self)
    
    def initialize_recorder(self):
        """Initialize the main recorder."""
        self.recorder = AudioToTextRecorder(
            model=self.model,
            language=self.language,
            on_recording_start=self.on_recording_start,
            on_recording_stop=self.on_recording_stop
        )
    
    def start(self):
        """Start the application."""
        self._display_info()
        self._start_ahk_script()
        self.server.start()
        
        # Keep the main thread alive
        try:
            while self.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.shutdown()
    
    def _display_info(self):
        """Display startup information."""
        print("\n==== Speech-to-Text Application ====")
        print(f"Model: {self.model}")
        print(f"Language: {self.language}")
        print("Hotkeys:")
        print("  F2 -> Toggle real-time transcription")
        print("  F3 -> Start long-form recording")
        print("  F4 -> Stop & transcribe long-form")
        print("  F10 -> Static file transcription")
        print("  F7 -> Quit")
        print("==================================\n")
    
    def _start_ahk_script(self):
        """Launch the AHK script for hotkey handling."""
        if platform.system() != 'Windows':
            print("AutoHotkey is only supported on Windows.")
            return
            
        ahk_path = os.path.join(self.script_dir, self.hotkey_script)
        print(f"Launching AHK script: {ahk_path}")
        
        try:
            subprocess.Popen(
                [ahk_path],
                creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
                shell=True
            )
            print("AHK script launched successfully.")
        except Exception as e:
            print(f"Error launching AHK script: {e}")
    
    # Long-form transcription methods
    def start_recording(self):
        """Start long-form recording."""
        if self.realtime_active:
            print("Cannot start long-form recording while real-time transcription is active.")
            return
            
        if self.static_transcription_active:
            print("Cannot start long-form recording while static file transcription is in progress.")
            return
            
        if self.recording_active:
            print("Recording is already active.")
            return
            
        print("\n----- LONG-FORM RECORDING -----")
        print("Starting recording...")
        
        # Reset recorder if needed
        if not hasattr(self, 'recorder') or self.recorder is None:
            self.initialize_recorder()
            
        # Start recording
        self.recorder.start()
        self.recording_active = True
        
    def stop_and_transcribe(self):
        """Stop recording and transcribe the audio."""
        if not self.recording_active:
            print("No active recording to stop.")
            return
            
        print("Stopping recording and transcribing...")
        self.recorder.stop()
        self.recording_active = False
        
        # Get transcription and print to terminal
        try:
            print("Processing transcription. This may take a moment...")
            transcription = self.recorder.text()
            print("\n======== TRANSCRIPTION ========")
            print(transcription)
            print("==============================\n")
        except Exception as e:
            print(f"Error during transcription: {e}")
        
        # Create a new recorder for future use without completely shutting down
        try:
            # NOTE: Don't call shutdown() on the recorder as it causes the "Shutting down" message
            # Instead, just create a new recorder
            self.initialize_recorder()
        except Exception as e:
            print(f"Error resetting recorder: {e}")
    
    # Real-time transcription methods
    def toggle_realtime_transcription(self):
        """Toggle real-time transcription on/off."""
        if self.realtime_active:
            print("\n----- STOPPING REAL-TIME TRANSCRIPTION -----")
            
            if hasattr(self, 'realtime_recorder'):
                self.realtime_recorder.abort()  # Use abort() instead of shutdown()
                delattr(self, 'realtime_recorder')
                
            self.realtime_active = False
            print("Real-time transcription stopped.")
        else:
            # Check if other processes are active
            if self.recording_active:
                print("Cannot start real-time transcription while long-form recording is active.")
                return
                
            if self.static_transcription_active:
                print("Cannot start real-time transcription while static file transcription is in progress.")
                return
                
            print("\n----- STARTING REAL-TIME TRANSCRIPTION -----")
            print("Speak and see real-time results...")
            
            # This callback function is critical for real-time output
            def realtime_update(text):
                # Print the text without newline, overwriting the current line
                print(f"\r[REAL-TIME]: {text:<100}", end="", flush=True)
            
            # Set up real-time recorder - with explicit parameters for realtime
            self.realtime_recorder = AudioToTextRecorder(
                model=self.model,
                language=self.language,
                enable_realtime_transcription=True,
                realtime_processing_pause=0.05,  # Lower value for more frequent updates
                on_realtime_transcription_update=realtime_update,
                use_main_model_for_realtime=False,  # Use a separate model for real-time
                realtime_model_type="tiny"  # Use a smaller model for speed
            )
            
            # Start listening for real-time transcription
            def realtime_listen():
                while self.realtime_active:
                    try:
                        # Use text() method in a loop to keep getting updates
                        self.realtime_recorder.text()
                        time.sleep(0.1)  # Small pause to prevent CPU overuse
                    except Exception as e:
                        print(f"\nError in real-time transcription: {e}")
                        break
            
            # Start real-time in a separate thread
            self.realtime_active = True
            threading.Thread(target=realtime_listen, daemon=True).start()
    
    # Static file transcription
    def transcribe_static_file(self):
        """Open file dialog and transcribe selected file."""
        if self.realtime_active:
            print("Cannot perform static transcription while real-time transcription is active.")
            return
            
        if self.recording_active:
            print("Cannot perform static transcription while long-form recording is active.")
            return
            
        if self.static_transcription_active:
            print("Static file transcription is already in progress.")
            return
            
        print("\n----- STATIC FILE TRANSCRIPTION -----")
        
        # Create root window for file dialog
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        
        # Open file dialog
        file_path = filedialog.askopenfilename(
            title="Select audio or video file",
            filetypes=[
                ("Audio/Video files", "*.mp3 *.wav *.mp4 *.avi *.mov *.m4a *.flac *.ogg *.mkv *.webm"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            print("No file selected. Aborting.")
            return
            
        print(f"Selected file: {file_path}")
        
        # Process in a separate thread to keep the UI responsive
        threading.Thread(target=self._process_static_file, args=(file_path,), daemon=True).start()
    
    def _process_static_file(self, file_path):
        """Process and transcribe a static file."""
        self.static_transcription_active = True
        
        try:
            # Convert file to WAV format suitable for transcription
            temp_wav = self._convert_to_wav(file_path)
            if not temp_wav:
                print("Failed to convert file. Aborting.")
                self.static_transcription_active = False
                return
            
            print("Converting complete. Starting transcription...")
            
            # Create a completely separate recorder instance for static transcription
            # Very important: set use_microphone=False
            static_recorder = None
            try:
                static_recorder = AudioToTextRecorder(
                    model=self.model,
                    language=self.language,
                    use_microphone=False
                )
                
                # Read the audio file as binary data
                with open(temp_wav, 'rb') as f:
                    audio_data = f.read()
                
                # Feed the audio data into the recorder
                print("Feeding audio data to transcriber...")
                static_recorder.feed_audio(audio_data)
                
                # Process transcription synchronously
                print("Transcribing... (this may take some time)")
                transcription = static_recorder.text()
                
                # Display result
                print("\n======== STATIC FILE TRANSCRIPTION ========")
                print(transcription)
                print("==========================================\n")
                
                # Offer to save the transcription
                self._save_transcription(transcription, file_path)
                
            finally:
                # Make sure to clean up properly
                if static_recorder:
                    try:
                        # Use abort to prevent "shutting down" message
                        static_recorder.abort()
                    except:
                        pass
                
        except Exception as e:
            print(f"Error processing static file: {e}")
        finally:
            self._cleanup_temp_files()
            self.static_transcription_active = False
    
    def _convert_to_wav(self, input_path):
        """Convert input file to 16kHz mono WAV format."""
        output_path = os.path.join(self.temp_dir, "temp_audio.wav")
        
        try:
            # Check if ffmpeg is available
            subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            
            # Convert file using ffmpeg
            subprocess.run([
                "ffmpeg",
                "-y",  # Overwrite output file if it exists
                "-i", input_path,  # Input file
                "-vn",  # Skip video stream
                "-ar", "16000",  # 16kHz sample rate
                "-ac", "1",  # Mono
                "-c:a", "pcm_s16le",  # 16-bit PCM
                output_path
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            
            print(f"Successfully converted file to 16kHz mono WAV.")
            return output_path
            
        except subprocess.CalledProcessError as e:
            print(f"Error converting file: {e}")
            return None
        except FileNotFoundError:
            print("ffmpeg not found. Please install ffmpeg to transcribe files.")
            return None
    
    def _save_transcription(self, transcription, original_file):
        """Offer to save the transcription to a text file."""
        root = tk.Tk()
        root.withdraw()
        
        # Default save path: same directory as original file, but with .txt extension
        default_path = os.path.splitext(original_file)[0] + ".txt"
        
        save_path = filedialog.asksaveasfilename(
            title="Save transcription",
            initialfile=os.path.basename(default_path),
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if save_path:
            try:
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(transcription)
                print(f"Transcription saved to: {save_path}")
            except Exception as e:
                print(f"Error saving transcription: {e}")
    
    def _cleanup_temp_files(self):
        """Remove temporary files."""
        try:
            temp_wav = os.path.join(self.temp_dir, "temp_audio.wav")
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
        except Exception as e:
            print(f"Error cleaning up temporary files: {e}")
    
    # Callback methods
    def on_recording_start(self):
        """Callback when recording starts."""
        print("Recording started!")
    
    def on_recording_stop(self):
        """Callback when recording stops."""
        print("Recording stopped!")
    
    def shutdown(self):
        """Shut down the application gracefully."""
        print("Shutting down...")
        self.is_running = False
        
        try:
            # Clean up recorder resources safely
            if hasattr(self, 'recorder') and self.recorder is not None:
                try:
                    self.recorder.abort()  # Using abort instead of shutdown
                except:
                    pass
                    
            if hasattr(self, 'realtime_recorder') and self.realtime_recorder is not None:
                try:
                    self.realtime_recorder.abort()  # Using abort instead of shutdown
                except:
                    pass
                    
        except Exception as e:
            print(f"Error during shutdown: {e}")
        
        # Clean up temp directory
        self._cleanup_temp_files()
        
        # Exit system
        os._exit(0)  # Force exit to avoid thread issues


class CommandServer:
    """Server for receiving commands from AHK script."""
    def __init__(self, app):
        self.app = app
        self.keep_running = True
        self.server_thread = None
    
    def start(self):
        """Start the TCP server in a separate thread."""
        self.server_thread = threading.Thread(target=self._run_server, daemon=True)
        self.server_thread.start()
    
    def _run_server(self):
        """Run the TCP server to receive commands from AHK."""
        host = '127.0.0.1'
        port = 34909
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # Add socket option to reuse address
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
            sock.listen(5)
            sock.settimeout(1)  # Allow checking keep_running flag every second

            print(f"TCP server listening on {host}:{port}")

            while self.keep_running:
                try:
                    conn, addr = sock.accept()
                    data = conn.recv(1024).decode('utf-8').strip()
                    print(f"Received command: '{data}'")

                    # Process command
                    self._handle_command(data)
                    conn.close()
                    
                except socket.timeout:
                    continue  # Just a timeout, check keep_running and continue
                except Exception as e:
                    print(f"Socket error: {e}")
                    
        except Exception as e:
            print(f"Server error: {e}")
            
        finally:
            if 'sock' in locals():
                sock.close()
    
    def _handle_command(self, command):
        """Process commands received from AHK."""
        if command == "TOGGLE_REALTIME_TRANSCRIPTION":
            self.app.toggle_realtime_transcription()
        elif command == "START_RECORDING":
            self.app.start_recording()
        elif command == "STOP_AND_TRANSCRIBE":
            self.app.stop_and_transcribe()
        elif command == "TRANSCRIBE_STATIC":
            self.app.transcribe_static_file()
        elif command == "QUIT":
            print("Received QUIT command")
            self.keep_running = False
            self.app.shutdown()


if __name__ == "__main__":
    app = STTApplication()
    app.start()