# unified_configuration_dialog.py
class UnifiedConfigDialog:
    def __init__(self, config, console, realtime_handler, transcriber):
        self.config = config
        self.console = console
        self.realtime_handler = realtime_handler
        self.transcriber = transcriber
        
        # Languages to display first with flag icons
        self.priority_languages = {
            "en": "🇬🇧 English (en)",
            "el": "🇬🇷 Greek (el)",
            "ru": "🇷🇺 Russian (ru)",
            "zh": "🇨🇳 Chinese (zh)",
            "de": "🇩🇪 German (de)",
            "fr": "🇫🇷 French (fr)"
        }
        
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
        
        # Language code to display name mapping (for non-priority languages)
        self.language_display = {}
        for code, name in self.languages.items():
            if code not in self.priority_languages:
                self.language_display[code] = f"{name.title()} ({code})"
    
    def _get_available_models(self):
        """Get a list of available Whisper models from the HuggingFace cache."""
        import os
        import glob
        
        models = []
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")

        # Helper function to normalize model names for comparison
        def normalize_model_name(name):
            # Remove 'models--' prefix and replace '--' with '/'
            if name.startswith("models--"):
                parts = name.split("--")
                if len(parts) >= 3:
                    return f"{parts[1]}/{parts[2]}"
            return name
        
        # Track normalized names to avoid duplicates
        normalized_models = set()
        
        if os.path.exists(cache_dir):
            # Look for model directories directly in the hub folder
            model_dirs = [d for d in os.listdir(cache_dir) if os.path.isdir(os.path.join(cache_dir, d))]
            for model_dir in model_dirs:
                # Only add model directories that have "whisper" in the name
                if "whisper" in model_dir.lower():
                    norm_name = normalize_model_name(model_dir)
                    if norm_name not in normalized_models:
                        normalized_models.add(norm_name)
                        models.append(norm_name)  # Use normalized name instead of directory name
        
        # Add default models if not found
        default_models = [
            "deepdml/faster-whisper-large-v3-turbo-ct2",
            "Systran/faster-whisper-medium",
            "Systran/faster-whisper-large-v3"
        ]
        
        for model in default_models:
        # Check if this model (or a variant) is already in the list
            norm_name = normalize_model_name(model)
            if norm_name not in normalized_models:
                if model not in models:
                    models.append(model)
                    normalized_models.add(norm_name)
        
        return sorted(models)
    
    def show_dialog(self):
        """Show the unified config dialog."""
        self.console.print("[bold blue]Opening configuration dialog...[/bold blue]")

        import tkinter as tk
        from tkinter import ttk

        # Create the main dialog window
        root = tk.Tk()
        root.title("Configuration Settings")

        # Dark theme colors
        bg_color = "#333333"        # Dark grey background
        text_color = "#FFFFFF"      # White text
        entry_bg = "#555555"        # Slightly lighter grey for input fields
        button_bg = "#444444"       # Medium grey for buttons
        highlight_color = "#007ACC" # Blue highlight color

        # Size and position
        root.geometry("700x600")
        root.configure(bg=bg_color)

        # Make dialog appear on top
        root.attributes('-topmost', True)

        # Center the window on screen
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x = (screen_width - 700) // 2
        y = (screen_height - 600) // 2
        root.geometry(f"700x600+{x}+{y}")

        # Main content frame with tabs
        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Style for the notebook
        style = ttk.Style()
        style.configure("TNotebook", background=bg_color, borderwidth=0)
        style.configure("TNotebook.Tab", background=button_bg, foreground="black", padding=[10, 5])
        style.map("TNotebook.Tab", background=[("selected", highlight_color)], foreground=[("selected", "black")])
        
        # Create tabs
        language_tab = tk.Frame(notebook, bg=bg_color)
        audio_tab = tk.Frame(notebook, bg=bg_color)
        models_tab = tk.Frame(notebook, bg=bg_color)
        
        notebook.add(language_tab, text="Language")
        notebook.add(audio_tab, text="Audio Source")
        notebook.add(models_tab, text="Models")
        
        # Language Tab
        language_frame = tk.Frame(language_tab, bg=bg_color)
        language_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create a frame for the two-column layout
        columns_frame = tk.Frame(language_frame, bg=bg_color)
        columns_frame.pack(fill="both", expand=True)

        # Long Form STT Column
        longform_frame = tk.Frame(columns_frame, bg=bg_color)
        longform_frame.pack(side="left", fill="both", expand=True, padx=10)

        longform_title = tk.Label(
            longform_frame, 
            text="Long Form STT", 
            font=("Arial", 12, "bold"), 
            bg=bg_color, 
            fg=text_color
        )
        longform_title.pack(pady=(0, 5))

        # Current Long Form language
        longform_lang_name = self.languages.get(self.config.longform_language, "Unknown")
        current_longform_label = tk.Label(
            longform_frame, 
            text=f"Current: {longform_lang_name.title()} ({self.config.longform_language})", 
            bg=bg_color, 
            fg=text_color, 
            font=("Arial", 10, "italic")
        )
        current_longform_label.pack(pady=(0, 10))

        # Long Form search
        longform_search_label = tk.Label(
            longform_frame, 
            text="Search:", 
            bg=bg_color, 
            fg=text_color, 
            font=("Arial", 11)
        )
        longform_search_label.pack(anchor="w")

        longform_search_var = tk.StringVar()
        longform_search_entry = tk.Entry(
            longform_frame, 
            textvariable=longform_search_var, 
            font=("Arial", 11), 
            bg=entry_bg, 
            fg=text_color, 
            insertbackground=text_color
        )
        longform_search_entry.pack(fill="x", pady=5)

        # Long Form listbox
        longform_listbox_frame = tk.Frame(longform_frame, bg=bg_color)
        longform_listbox_frame.pack(fill="both", expand=True, pady=5)

        longform_scrollbar = tk.Scrollbar(longform_listbox_frame, bg=button_bg, troughcolor=bg_color)
        longform_scrollbar.pack(side="right", fill="y")

        longform_listbox = tk.Listbox(
            longform_listbox_frame, 
            yscrollcommand=longform_scrollbar.set, 
            font=("Segoe UI Emoji", 11), 
            selectmode="single",
            height=12,
            bg=entry_bg,
            fg=text_color,
            selectbackground=highlight_color,
            selectforeground=text_color
        )
        longform_listbox.pack(side="left", fill="both", expand=True)
        longform_scrollbar.config(command=longform_listbox.yview)

        # Real Time STT Column
        realtime_frame = tk.Frame(columns_frame, bg=bg_color)
        realtime_frame.pack(side="right", fill="both", expand=True, padx=10)

        realtime_title = tk.Label(
            realtime_frame, 
            text="Real Time STT", 
            font=("Arial", 12, "bold"), 
            bg=bg_color, 
            fg=text_color
        )
        realtime_title.pack(pady=(0, 5))

        # Current Real Time language
        realtime_lang_name = self.languages.get(self.config.realtime_language, "Unknown")
        current_realtime_label = tk.Label(
            realtime_frame, 
            text=f"Current: {realtime_lang_name.title()} ({self.config.realtime_language})", 
            bg=bg_color, 
            fg=text_color, 
            font=("Arial", 10, "italic")
        )
        current_realtime_label.pack(pady=(0, 10))

        # Real Time search
        realtime_search_label = tk.Label(
            realtime_frame, 
            text="Search:", 
            bg=bg_color, 
            fg=text_color, 
            font=("Arial", 11)
        )
        realtime_search_label.pack(anchor="w")

        realtime_search_var = tk.StringVar()
        realtime_search_entry = tk.Entry(
            realtime_frame, 
            textvariable=realtime_search_var, 
            font=("Arial", 11), 
            bg=entry_bg, 
            fg=text_color, 
            insertbackground=text_color
        )
        realtime_search_entry.pack(fill="x", pady=5)

        # Real Time listbox
        realtime_listbox_frame = tk.Frame(realtime_frame, bg=bg_color)
        realtime_listbox_frame.pack(fill="both", expand=True, pady=5)

        realtime_scrollbar = tk.Scrollbar(realtime_listbox_frame, bg=button_bg, troughcolor=bg_color)
        realtime_scrollbar.pack(side="right", fill="y")

        realtime_listbox = tk.Listbox(
            realtime_listbox_frame, 
            yscrollcommand=realtime_scrollbar.set, 
            font=("Segoe UI Emoji", 11), 
            selectmode="single",
            height=12,
            bg=entry_bg,
            fg=text_color,
            selectbackground=highlight_color,
            selectforeground=text_color
        )
        realtime_listbox.pack(side="left", fill="both", expand=True)
        realtime_scrollbar.config(command=realtime_listbox.yview)

        # Populate language listboxes
        # First add priority languages
        for code, display_name in self.priority_languages.items():
            longform_listbox.insert(tk.END, display_name)
            realtime_listbox.insert(tk.END, display_name)
        
        # Add a separator
        longform_listbox.insert(tk.END, "───────────────────")
        realtime_listbox.insert(tk.END, "───────────────────")
        
        # Then add the rest in alphabetical order
        sorted_display = sorted(self.language_display.values())
        for display_name in sorted_display:
            longform_listbox.insert(tk.END, display_name)
            realtime_listbox.insert(tk.END, display_name)

        # Audio Tab
        audio_frame = tk.Frame(audio_tab, bg=bg_color)
        audio_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create table-like layout
        table_frame = tk.Frame(audio_frame, bg=bg_color)
        table_frame.pack(fill="both", expand=True, padx=10, pady=30)

        # Header row
        header_frame = tk.Frame(table_frame, bg=bg_color)
        header_frame.pack(fill="x", pady=(0, 10))

        # Empty corner cell
        corner_cell = tk.Frame(header_frame, width=150, bg=bg_color)
        corner_cell.pack(side="left", padx=5)

        # Long Form STT header
        longform_header = tk.Label(
            header_frame, 
            text="Long Form STT", 
            font=("Arial", 12, "bold"), 
            width=15,
            bg=bg_color, 
            fg=text_color
        )
        longform_header.pack(side="left", padx=5)

        # Real Time STT header
        realtime_header = tk.Label(
            header_frame, 
            text="Real Time STT", 
            font=("Arial", 12, "bold"), 
            width=15,
            bg=bg_color, 
            fg=text_color
        )
        realtime_header.pack(side="left", padx=5)

        # Audio source row
        audio_row = tk.Frame(table_frame, bg=bg_color)
        audio_row.pack(fill="x", pady=10)

        # Audio source label
        audio_label = tk.Label(
            audio_row, 
            text="Audio Source:", 
            font=("Arial", 11), 
            width=15,
            anchor="w",
            bg=bg_color, 
            fg=text_color
        )
        audio_label.pack(side="left", padx=5)

        # Long Form audio source dropdown
        longform_audio_var = tk.StringVar(value="Microphone" if not self.config.longform_use_system_audio else "System Audio")
        longform_audio_dropdown = tk.OptionMenu(
            audio_row, 
            longform_audio_var, 
            "Microphone", 
            "System Audio"
        )
        longform_audio_dropdown.config(
            bg=entry_bg, 
            fg=text_color, 
            highlightbackground=bg_color,
            activebackground=highlight_color,
            activeforeground=text_color,
            font=("Arial", 10),
            width=12
        )
        longform_audio_dropdown["menu"].config(
            bg=entry_bg,
            fg=text_color,
            activebackground=highlight_color,
            activeforeground=text_color
        )
        longform_audio_dropdown.pack(side="left", padx=5)

        # Real Time audio source dropdown
        realtime_audio_var = tk.StringVar(value="Microphone" if not self.config.realtime_use_system_audio else "System Audio")
        realtime_audio_dropdown = tk.OptionMenu(
            audio_row, 
            realtime_audio_var, 
            "Microphone", 
            "System Audio"
        )
        realtime_audio_dropdown.config(
            bg=entry_bg, 
            fg=text_color, 
            highlightbackground=bg_color,
            activebackground=highlight_color,
            activeforeground=text_color,
            font=("Arial", 10),
            width=12
        )
        realtime_audio_dropdown["menu"].config(
            bg=entry_bg,
            fg=text_color,
            activebackground=highlight_color,
            activeforeground=text_color
        )
        realtime_audio_dropdown.pack(side="left", padx=5)
        
        # Models Tab
        models_frame = tk.Frame(models_tab, bg=bg_color)
        models_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create a frame for the two-column layout
        models_columns_frame = tk.Frame(models_frame, bg=bg_color)
        models_columns_frame.pack(fill="both", expand=True)
        
        # Long Form STT Model Column
        longform_model_frame = tk.Frame(models_columns_frame, bg=bg_color)
        longform_model_frame.pack(side="left", fill="both", expand=True, padx=10)
        
        longform_model_title = tk.Label(
            longform_model_frame, 
            text="Long Form Model", 
            font=("Arial", 12, "bold"), 
            bg=bg_color, 
            fg=text_color
        )
        longform_model_title.pack(pady=(20, 10))
        
        # Get available models
        available_models = self._get_available_models()
        
        # Current Long Form model
        current_longform_model_label = tk.Label(
            longform_model_frame, 
            text=f"Current Model: {self.transcriber.model_id}", 
            bg=bg_color, 
            fg=text_color, 
            font=("Arial", 10, "italic")
        )
        current_longform_model_label.pack(pady=(0, 20))
        
        # Long Form model dropdown
        longform_model_var = tk.StringVar(value=self.transcriber.model_id)
        longform_model_dropdown = tk.OptionMenu(
            longform_model_frame, 
            longform_model_var,
            *available_models
        )
        longform_model_dropdown.config(
            bg=entry_bg, 
            fg=text_color, 
            highlightbackground=bg_color,
            activebackground=highlight_color,
            activeforeground=text_color,
            font=("Arial", 10),
            width=40
        )
        longform_model_dropdown["menu"].config(
            bg=entry_bg,
            fg=text_color,
            activebackground=highlight_color,
            activeforeground=text_color
        )
        longform_model_dropdown.pack(pady=10)
        
        # Real Time STT Model Column
        realtime_model_frame = tk.Frame(models_columns_frame, bg=bg_color)
        realtime_model_frame.pack(side="right", fill="both", expand=True, padx=10)
        
        realtime_model_title = tk.Label(
            realtime_model_frame, 
            text="Real Time Model", 
            font=("Arial", 12, "bold"), 
            bg=bg_color, 
            fg=text_color
        )
        realtime_model_title.pack(pady=(20, 10))
        
        # Current Real Time model
        current_realtime_model_label = tk.Label(
            realtime_model_frame, 
            text=f"Current Model: {self.realtime_handler.realtime_model_name}", 
            bg=bg_color, 
            fg=text_color, 
            font=("Arial", 10, "italic")
        )
        current_realtime_model_label.pack(pady=(0, 20))
        
        # Real Time model dropdown
        realtime_model_var = tk.StringVar(value=self.realtime_handler.realtime_model_name)
        realtime_model_dropdown = tk.OptionMenu(
            realtime_model_frame, 
            realtime_model_var, 
            *available_models
        )
        realtime_model_dropdown.config(
            bg=entry_bg, 
            fg=text_color, 
            highlightbackground=bg_color,
            activebackground=highlight_color,
            activeforeground=text_color,
            font=("Arial", 10),
            width=40
        )
        realtime_model_dropdown["menu"].config(
            bg=entry_bg,
            fg=text_color,
            activebackground=highlight_color,
            activeforeground=text_color
        )
        realtime_model_dropdown.pack(pady=10)
        
        # Add Enter key toggle
        enter_frame = tk.Frame(models_frame, bg=bg_color)
        enter_frame.pack(fill="x", pady=30)
        
        enter_var = tk.BooleanVar(value=self.config.send_enter)
        enter_checkbox = tk.Checkbutton(
            enter_frame,
            text="Send Enter key after transcription",
            variable=enter_var,
            bg=bg_color,
            fg=text_color,
            selectcolor=button_bg,
            activebackground=bg_color,
            activeforeground=text_color,
            font=("Arial", 11)
        )
        enter_checkbox.pack()

        # Function to update listbox based on search
        def update_longform_listbox(*args):
            search_text = longform_search_var.get().lower()
            longform_listbox.delete(0, tk.END)
            
            # First display priority languages that match
            matching_priority = []
            for code, display_name in self.priority_languages.items():
                if search_text in code.lower() or search_text in self.languages[code].lower():
                    matching_priority.append(display_name)
            
            for display_name in matching_priority:
                longform_listbox.insert(tk.END, display_name)
            
            # Add separator if there are priority matches
            if matching_priority:
                longform_listbox.insert(tk.END, "───────────────────")
            
            # Find matching non-priority languages
            matching_normal = []
            for code, name in self.languages.items():
                if code not in self.priority_languages and (search_text in code.lower() or search_text in name.lower()):
                    matching_normal.append(self.language_display[code])
            
            # Sort and add them
            for display_name in sorted(matching_normal):
                longform_listbox.insert(tk.END, display_name)
            
            # If there are matches, select the first one
            if matching_priority or matching_normal:
                longform_listbox.selection_set(0)

        def update_realtime_listbox(*args):
            search_text = realtime_search_var.get().lower()
            realtime_listbox.delete(0, tk.END)
            
            # First display priority languages that match
            matching_priority = []
            for code, display_name in self.priority_languages.items():
                if search_text in code.lower() or search_text in self.languages[code].lower():
                    matching_priority.append(display_name)
            
            for display_name in matching_priority:
                realtime_listbox.insert(tk.END, display_name)
            
            # Add separator if there are priority matches
            if matching_priority:
                realtime_listbox.insert(tk.END, "───────────────────")
            
            # Find matching non-priority languages
            matching_normal = []
            for code, name in self.languages.items():
                if code not in self.priority_languages and (search_text in code.lower() or search_text in name.lower()):
                    matching_normal.append(self.language_display[code])
            
            # Sort and add them
            for display_name in sorted(matching_normal):
                realtime_listbox.insert(tk.END, display_name)
            
            # If there are matches, select the first one
            if matching_priority or matching_normal:
                realtime_listbox.selection_set(0)

        # Bind search entries to update functions
        longform_search_var.trace("w", update_longform_listbox)
        realtime_search_var.trace("w", update_realtime_listbox)

        # Result dictionary to store selected values
        result = {
            "longform_language": None,
            "realtime_language": None,
            "longform_use_system_audio": None,
            "realtime_use_system_audio": None,
            "longform_model_name": None,
            "realtime_model_name": None,
            "send_enter": None
        }

        # Flag to track if Apply was clicked
        apply_clicked = [False]  # Using a list for mutability in nested functions

        # Functions to extract code from display name
        def get_code_from_display(display_name):
            # Check if it's a priority language with flag
            for code, priority_name in self.priority_languages.items():
                if display_name == priority_name:
                    return code
            
            # Check if it's a separator
            if "───" in display_name:
                return None
                
            try:
                return display_name.split("(")[1].split(")")[0]
            except (IndexError, Exception):
                return None

        # Function to apply selections
        def apply_selections():
            # Get Long Form language selection
            longform_selection = None
            if longform_listbox.curselection():
                selection_idx = longform_listbox.curselection()[0]
                display_name = longform_listbox.get(selection_idx)
                longform_selection = get_code_from_display(display_name)
            
            # Get Real Time language selection
            realtime_selection = None
            if realtime_listbox.curselection():
                selection_idx = realtime_listbox.curselection()[0]
                display_name = realtime_listbox.get(selection_idx)
                realtime_selection = get_code_from_display(display_name)
            
            # Store results
            result["longform_language"] = longform_selection
            result["realtime_language"] = realtime_selection
            result["longform_use_system_audio"] = (longform_audio_var.get() == "System Audio")
            result["realtime_use_system_audio"] = (realtime_audio_var.get() == "System Audio")
            result["longform_model_name"] = longform_model_var.get()
            result["realtime_model_name"] = realtime_model_var.get()
            result["send_enter"] = enter_var.get()

            # Set flag to indicate Apply was clicked
            apply_clicked[0] = True

            root.destroy()

        # Function to handle cancel
        def on_cancel():
            root.destroy()

        # Button frame
        button_frame = tk.Frame(root, bg=bg_color)
        button_frame.pack(pady=15, padx=20, fill="x")

        apply_button = tk.Button(
            button_frame, 
            text="Apply", 
            command=apply_selections,
            bg="#4CAF50",
            fg=text_color,
            font=("Arial", 12),
            width=10,
            activebackground="#3e8e41",
            activeforeground=text_color
        )
        apply_button.pack(side="right", padx=5)

        cancel_button = tk.Button(
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

        # Bind Enter key to apply
        root.bind("<Return>", lambda e: apply_selections())

        # Bind Escape key to cancel
        root.bind("<Escape>", lambda e: on_cancel())

        # Handle window close event (X button) as cancellation
        root.protocol("WM_DELETE_WINDOW", on_cancel)

        # Run the dialog
        root.mainloop()

        # Return None if Apply wasn't clicked
        if not apply_clicked[0]:
            return None

        # Return the results
        return result

    def stop_all_transcriptions(self) -> None:
        """Stop all running transcription processes."""
        # Check if live recording is in progress
        if hasattr(self, 'recorder') and self.recorder.recording:
            self.console.print("[bold yellow]Stopping live transcription...[/bold yellow]")
            self.recorder.recording = False
            if self.recorder.recording_thread:
                self.recorder.recording_thread.join()
            self.recorder._cleanup_resources()
        
        # Check if real-time transcription is running
        if hasattr(self, 'realtime_handler') and self.realtime_handler.is_running:
            self.console.print("[bold yellow]Stopping real-time transcription...[/bold yellow]")
            self.realtime_handler.stop()
        
        # Check if static transcription is in progress
        if hasattr(self, 'static_processor') and self.static_processor.is_transcribing():
            self.console.print("[bold yellow]Stopping static transcription...[/bold yellow]")
            self.static_processor.request_abort()
        
        # Flash the tray icon for visual feedback
        self.tray.flash_white('gray', self.config.send_enter)
        self.console.print("[green]All transcription processes stopped.[/green]")