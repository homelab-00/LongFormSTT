#!/usr/bin/env python3
# linux_hotkeys.py - Linux replacement for AHK_script-hotkeys_handling.ahk

import socket
import keyboard
import sys
import threading
import time
import signal

def send_command(command):
    """Send a command to the TCP server."""
    try:
        # Connect to the local TCP server
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('127.0.0.1', 34909))
        
        # Send the command
        sock.sendall(command.encode('utf-8'))
        
        # Close the connection
        sock.close()
    except Exception as e:
        print(f"Error sending command: {e}", file=sys.stderr)

# Define hotkey mappings
hotkey_commands = {
    'f1': 'OPEN_CONFIG_DIALOG',
    'f2': 'TOGGLE_REALTIME_TRANSCRIPTION',
    'f3': 'START_RECORDING',
    'f4': 'STOP_AND_TRANSCRIBE',
    'f5': 'TOGGLE_ENTER',
    'f6': 'RESET_TRANSCRIPTION',
    'f7': 'QUIT',
    'f10': 'TRANSCRIBE_STATIC'
}

# Handle SIGTERM for clean shutdown
def signal_handler(sig, frame):
    print("Received termination signal, exiting...")
    keyboard.unhook_all()
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)

# Register all hotkeys
for key, command in hotkey_commands.items():
    # Use lambda with default argument to capture the current command value
    keyboard.add_hotkey(key, lambda cmd=command: send_command(cmd))

print("Linux hotkey handler started. Press Ctrl+C to exit.")

# Keep the script running
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Keyboard interrupt received, exiting...")
    keyboard.unhook_all()