import os
import sys
import time
import socket
import threading
import subprocess
import platform
from RealtimeSTT import AudioToTextRecorder

class STTApplication:
    def __init__(self):
        # Configuration
        self.model = "Systran/faster-whisper-large-v3"
        self.language = "el"  # Greek language
        self.task = "transcribe"
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.hotkey_script = "AHK_script-hotkeys_handling.ahk"
        
        # Initialize RealtimeSTT recorder
        self.recorder = AudioToTextRecorder(
            model=self.model,
            language=self.language,
            on_recording_start=self.on_recording_start,
            on_recording_stop=self.on_recording_stop,
        )
        
        # State
        self.is_running = True
        self.ahk_pid = None
        
        # Set up command server
        self.server = CommandServer(self)
        
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
        print("\n==== LongFormSTT using RealtimeSTT Library ====")
        print(f"Model: {self.model}")
        print(f"Language: {self.language}")
        print("Hotkeys:")
        print("  F3 -> Start recording")
        print("  F4 -> Stop & transcribe")
        print("=========================================\n")
    
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
    
    def start_recording(self):
        """Start recording audio."""
        print("Starting recording...")
        self.recorder.start()
        
    def stop_and_transcribe(self):
        """Stop recording and transcribe the audio."""
        print("Stopping recording and transcribing...")
        self.recorder.stop()
        
        # Get transcription and print to terminal
        transcription = self.recorder.text()
        print("\n======== TRANSCRIPTION ========")
        print(transcription)
        print("==============================\n")
    
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
            # Clean up recorder resources
            self.recorder.shutdown()
        except Exception as e:
            print(f"Error shutting down recorder: {e}")
        
        sys.exit(0)


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
        if command == "START_RECORDING":
            self.app.start_recording()
        elif command == "STOP_AND_TRANSCRIBE":
            self.app.stop_and_transcribe()
        elif command == "QUIT":
            print("Received QUIT command")
            self.keep_running = False
            self.app.shutdown()


if __name__ == "__main__":
    app = STTApplication()
    app.start()