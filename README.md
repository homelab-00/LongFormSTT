# LongFormSTT

[[[NOTE: The README is depreceated. Will update in the future.]]]

A python script that utilizes `faster-whisper` and `pytorch` for long form transcription. Uses silence detection with RMS/peak value. Has global hotkeys for easy use.

---

Transcription happens in 1 min windows (configurable). This means that every 1 min of recording time, a `temp_audio_file*.wav` file is generated
(in the same folder as the script) and sent to be transcribed by the model. When you stop recording, the script finishes whatever transcription
still has left and then prints the collated full transcript of all the audio files.

The temp audio files are auto deleted every time a new recording starts. This is a fail-safe feature, so that if the script crashes for some reason
at least you don't lose the recordings. If that happens, then you can use a much simpler script that just transcribes a single audio file and get
your data that way. Maybe combine the temp audio files in Audacity (though I'm not sure if the `whisper` models like it when the audio files get too long).

The model I'm using by default is [*Systran/faster-whisper-large-v3*](https://huggingface.co/Systran/faster-whisper-large-v3) (configurable). That means it's the *faster-whisper* version of *whisper-larger-v3*. There are other versions of
*whisper-large-v3* (such as *distil*) and of course other models. This one works best for my hardware and use case. (Specifically the *distil* version doesn't
work well with Greek, which is what I'm mostly speaking to it. I think it auto-translates or something. But for English it's great.)

I've designed this on my Windows 10 pc with an RTX 3060. It runs fine on my system but if you want better performance you can load a smaller model.
Built with CUDA in mind. It can run on CPU alone and also theoretically on newer AMD cards, though only on Linux. That is because `pytorch` only supports ROCm (AMD's version of CUDA) on Linux.

I'm using CUDA Toolkit 12.4 since that is the latest CUDA version supported by `pytorch 2.5.1` (which is the latest version of `pytorch`). If your Nvidia card
doesn't support this CUDA version, you can also use versions 11.8 and 12.1 (older versions supported by `pytorch 2.5.1`), or an older `pytorch` version altogether, though I haven't tested any of these configurations. 

---

This script was inspired by the excellent work of [@KoljaB](https://github.com/KoljaB) on [RealtimeSTT](https://github.com/KoljaB/RealtimeSTT). Check out his repo if you're more
interested in real time transcription (as well as his other cool projects).

The idea for this project is that I prefer static transcription instead of real time. The context window is bigger; although `whisper` breaks audio files
down to smaller chunks anyway, I'm sure a 1 min static transcription has better context than real time transcription. In addition I'm pretty sure it works
better for more fragmented speech with a lot of pauses. And also I don't like seeing the transcription happening constantly, it makes me feel anxious like
someone is watching me lol


## Installation

* First off, you need to install Toolkit 12.4 (skip if not using Nvidia GPU).  
Download it from: https://developer.nvidia.com/cuda-12-4-0-download-archive  
You will also need cuDNN, get it from: https://developer.nvidia.com/cudnn-downloads

* Then you need to install Python. I specifically used version [3.10.11](https://www.python.org/downloads/release/python-31011/) with this project.

* Next step is to create a Python virtual environment. It makes managing dependencies easier and keeps packages and configurations for each project separate. To do that:

    * To begin with, open a cmd window and navigate to the root folder where you want to create the venv:  
    `cd "C:\Users\Bill\PurePython_envs"`

    * Run this command to create a venv based on the Python 3.10.11 you installed:  
    `"C:\Program Files\Python310\python.exe" -m venv LongFormSTT-3.10.11-12.4-pip`  

        *Note: The naming scheme for the venv is just to remind myself that it's using Python 3.10.11, CUDA Toolkit 12.4 and that it's managed only by `pip`, not `anaconda`.*

    * To activate the venv:  
    Go to `C:\Users\Bill\PurePython_envs\LongForm-3.10.11-12.4-pip\Scripts`, open a cmd window and enter `activate.bat`.  
    Or just go to `C:\Users\Bill\PurePython_envs\LongFormSTT-3.10.11-12.4-pip` and enter in cmd `scripts\activate`.

        If successfully activated you should now see the name of the venv before the folder location in cmd, e.g. `(venv) C:\path\to\folder`.

        *Note: You can't just run the `activate.bat` file from the File Explorer directly because:  
        When you double-click the file from the File Explorer, it briefly opens a Command Prompt window, then immediately closes. This happens
        because .bat scripts are designed to run inside an active command line session. When you double-click from the Explorer, Windows opens a new
        Command Prompt, runs the script, and then closes it right after, without keeping the session open.*

    * To run commands (e.g. to list `pip` installed packages):  
    `python.exe -m pip list`

        *Note: You need to add `python.exe -m` before commands when working with the venv to make sure the command is run in the venv and not the normal Python installation.  
        On Windows there are aliases for apps, so using `python -m pip` instead is also fine.*

    * To deactivate the env:  
    Run `deactivate` or simply close the cmd window

* Now we need to install all the required packages listed in `requirements.txt`:  
*(The commands listed below should be run in the cmd window with the venv we just created active.)*

    * Before installing packages it might be a good idea to upgrade the default packages that come with the environment (they should only be `pip` and maybe `setuptools` and `wheel`).
    
        To do that run:  
         `python -m pip install --upgrade pip setuptools wheel`
    
    * Then we need to install `pytorch`. The commands to install the various versions can be found on their [website](https://pytorch.org/).

        * If you have CUDA Toolkit 12.4 installed, run the following command to install `pytorch 2.5.1` with CUDA 12.4 integration:  
        
            `python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124`
        
        * For the CPU only version, run:  
            
            `python -m pip install torch torchvision torchaudio`

    * Now we need to install all the other packages required by the script. To do that:
        * Download the [`requirements.txt`](https://github.com/homelab-00/LongFormSTT/blob/main/requirements.txt) file
        
        * Go to the download location by running:

            `cd C:\Bill\Downloads`

        * Then run:

            `python -m pip install -r requirements.txt`

* Next step is logging in to Hugging Face, so that we can download the model.

    * First create an account on the site and also create an access token.

    * Then you need to run the following script. First create a new project folder, open that folder in VS Code, then select as the Python interpreter the `python.exe` file in the `Scripts` folder of your venv.
      Finally run the script in VS Code. Your login credentials are saved for future use.

        ```
        from huggingface_hub import login

        huggingface_token = "your_huggingface_token"

        try:
            # Log in to Hugging Face
            login(token=huggingface_token)
            print("Successfully logged in to Hugging Face!")
        except Exception as e:
            print(f"Failed to log in: {e}")
        ```
        *Note: Now any Python command you run in a terminal inside this project in VS Code will automatically use the `python.exe` of the venv.* 

And that's it you're done!

Now just download the [latest version](https://github.com/homelab-00/LongFormSTT/tree/main/LongFormSTT%20-%20Latest%20version) of the script (the folder contains a preconfigured English version and Greek version),
place it in the VS Code project folder and run it from the app (or directly from cmd with the venv active).

*Note: The model is downloaded the first time you run the script (or when you change the model in the script afterwards). By default it's saved in `C:\Users\Bill\.cache\huggingface\hub`.*

## Usage

The script has hotkeys for easy use:
* F2 toggles typing on/off
* F3 starts recording
* F4 stops recording and transcribes

These hotkeys are global and suppressed. This means that they work even if VS Code isn't in focus and that when pressed (while the script is running) they won't be sent to the app in focus but instead sent only to the script.
That also means that while the script is running you won't be able to use those keys for other things (although you can configure them to any key you like).

By default, when the scripts finishes transcribing the text is automatically pasted in the focused app. This can be toggled with F2.

Also there's a bug that I haven't managed to fix (without breaking global hotkeys) regarding the hotkeys.  
You start a recording with F3 and then after a while you stop it with F4. The script does the transcription normally and prints it (the script will also print 'Done' in terminal when it's done). The issue is that afterwards
the hotkeys stop working. To fix it, simply press F4 again after the transcription is done. If you're working with VS Code minimized and typing enabled, then when you see the final transcription pasted in the app you have
in focus, that means the transcription is done and you can press F4 again to unlock the hotkeys.

*Note: For the duration that the hotkeys are stuck they're no longer suppressed.*

### Parameters

Most parameters are commented inside the script. The ones you'll mostly care about are:

* `THRESHOLD` (*default=500*): Defines how aggressive silence detection is.
* `CHUNK_SPLIT_INTERVAL` (*default=60*): Defines how long each temp audio chunk is in seconds.
* `model_id` (*default="Systran/faster-whisper-large-v3"*): The model used for transcription. You can change it by simply copying the name of the model you're interested in from Hugging Face. Note that they need to be in CTranslate2 format in order to be compatible with `faster-whisper` (as noted [here](https://github.com/SYSTRAN/faster-whisper#faster-whisper-transcription-with-ctranslate2)).
* `language` (*default="en" for the English version*): The main language you'll be speaking to the script. This doesn't mean that it won't recognize anything spoken in other languages; I use the script with the language set to "el" (Greek) and I can throw some English words and even small sentences in with no problem. Though this depends on the model and language mix.  
The full list of languages (and their associated codes) for the *whisper* family of models can be found [here](https://github.com/openai/whisper/blob/main/whisper/tokenizer.py).

## Preview

*Note: Sorry about the bad audio quality, I'm playing the audio from my phone to the pc microphone. Still it shows that the script works great even with choppy audio.*

https://github.com/user-attachments/assets/45ff1aab-d983-48f8-91ce-d7df72cac1fa

https://github.com/user-attachments/assets/d793f819-9336-4c66-8b40-547c25ae5335
