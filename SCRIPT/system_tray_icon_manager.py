# system_tray_icon_manager.py
#
# Manages the system tray icon to provide visual feedback on application state
#
# This module:
# - Creates and manages colored circular icons in the system tray
# - Provides visual indication of the application's current state:
#   * Gray: Idle/Ready
#   * Red: Recording
#   * Blue: Processing/Transcribing
#   * Yellow: Waiting for static file transcription
#   * White: Temporary flash to indicate state change
# - Uses a green outline to indicate when the "send Enter" option is enabled
# - Provides flash animation for visual feedback on various operations
# - Gracefully handles missing dependencies (pystray, PIL) with fallbacks
#
# The tray icon helps users quickly understand the current state of the
# application without needing to check the console output

from rich.console import Console
from typing import Tuple
import time
import threading

# Optional dependencies with graceful fallbacks
try:
    import pystray
    from PIL import Image, ImageDraw
    TRAY_AVAILABLE = True
except ImportError:
    TRAY_AVAILABLE = False

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