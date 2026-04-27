# AI_doll

## Project Description

This project implements a local voice-based RAG assistant with a face-recognition access gate. The system first opens the webcam and checks whether a predefined user is recognized. Once the target user is detected, it plays a short reference voice clip and then starts an interactive voice conversation loop.

During each interaction, the user presses Enter to start recording and presses Enter again to stop. The recorded audio is captured from the microphone, converted into a NumPy waveform, and transcribed with OpenAI Whisper. The transcribed text is then passed to a local RAG query function, `src.query.query_rag()`, which generates a text response based on the project’s retrieval pipeline. Finally, the response is synthesized into speech with Coqui XTTS v2 and played back through the system audio device.

The project integrates computer vision, speech recognition, retrieval-augmented generation, and text-to-speech synthesis into a simple terminal-driven prototype. It is suitable for experimenting with personalized voice assistants, local multimodal interaction, face-gated access control, and voice-based knowledge retrieval workflows. The current implementation is designed as a research or prototype system rather than a production-ready assistant. It assumes local access to a webcam, microphone, speaker, a known face image, a reference speaker audio file, and a working RAG backend.

## Features

- Face-recognition gate before starting the assistant
- Webcam-based identity detection with `face_recognition` and OpenCV
- Push-to-record terminal interaction
- Microphone audio capture with `sounddevice`
- Speech-to-text transcription using Whisper
- Retrieval-augmented response generation through `src.query.query_rag()`
- Chinese speech synthesis using Coqui XTTS v2
- Local audio playback through the default system audio device
- GPU acceleration support for TTS when CUDA is available

## System Workflow

```text
Start program
   ↓
Open webcam
   ↓
Detect known face
   ↓
Play reference voice clip
   ↓
Wait for user to start recording
   ↓
Record microphone audio
   ↓
Transcribe audio with Whisper
   ↓
Send text to RAG backend
   ↓
Generate assistant response
   ↓
Synthesize response with XTTS v2
   ↓
Play generated speech
   ↓
Repeat conversation loop
```

## Project Structure

A recommended project structure is shown below:

```text
project-root/
├── main.py
├── README.md
├── imgs/
│   └── biden.jpg
├── target_voice/
│   └── xinmeng_audio.wav
└── src/
    ├── __init__.py
    └── query.py
```

The current code expects the following files or modules to exist:

- `imgs/biden.jpg`: reference face image used for identity recognition
- `target_voice/xinmeng_audio.wav`: reference voice used for playback and XTTS voice cloning
- `src/query.py`: local RAG module containing the function `query_rag(text)`

## Requirements

The code depends on the following Python packages:

```text
numpy
openai-whisper
sounddevice
rich
torch
TTS
face_recognition
opencv-python
```

Additional system-level dependencies may be required for audio, video, and face recognition. In particular, `face_recognition` depends on `dlib`, which may require CMake and a compatible C++ build environment.

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install Python dependencies:

```bash
pip install numpy openai-whisper sounddevice rich torch TTS face_recognition opencv-python
```

If `face_recognition` fails to install, install CMake and dlib first:

```bash
pip install cmake
pip install dlib
pip install face_recognition
```

## Configuration

Before running the program, update the reference assets and user name in the code if needed.

### Reference Face

The program loads a known face image from:

```python
face_recognition.load_image_file("imgs/biden.jpg")
```

Replace this file with your own reference image, or update the file path.

### Recognized User Name

The recognized identity is currently stored as:

```python
known_face_names = ["Reze"]
```

Change this value to match the intended user label.

### Reference Voice

The program uses the following audio file as the reference speaker voice:

```python
"target_voice/xinmeng_audio.wav"
```

This file is used both for the initial playback and as the `speaker_wav` input for XTTS voice synthesis.

### RAG Backend

The response generation step depends on:

```python
response = qy.query_rag(text)
```

Make sure that `src/query.py` exists and implements a compatible function:

```python
def query_rag(text: str) -> str:
    ...
```

The function should accept a user query string and return a response string.

## Usage

Run the assistant:

```bash
python main.py
```

When the program starts, it will open the webcam and attempt to recognize the predefined user. After successful recognition, the assistant enters the voice interaction loop.

In each round:

1. Press Enter to start recording.
2. Speak into the microphone.
3. Press Enter again to stop recording.
4. Wait for transcription, RAG response generation, speech synthesis, and audio playback.

Press `Ctrl+C` to exit.

## Notes and Limitations

- The code uses the default webcam through `cv2.VideoCapture(0)`.
- The code uses the default microphone and speaker through `sounddevice`.
- Face recognition currently processes every other frame to reduce computation.
- The webcam window is not explicitly displayed with `cv2.imshow()` in the current implementation, although bounding boxes are drawn on frames internally.
- The code comments mention BGR-to-RGB conversion, but the current assignment keeps the OpenCV frame unchanged. For stricter color handling, use `rgb_small_frame = small_frame[:, :, ::-1]`.
- Whisper is loaded with the `turbo` model. Change this if a smaller or larger model is preferred.
- TTS uses `tts_models/multilingual/multi-dataset/xtts_v2`, which may require significant memory and may run slowly on CPU.
- The script is intended as a prototype and does not include authentication hardening, exception handling for missing devices, or robust production logging.

## Possible Improvements

- Add a graphical preview window for face detection.
- Add configuration through a `.env` or YAML file.
- Add command-line arguments for model selection and file paths.
- Improve error handling for missing webcam, microphone, model files, and reference assets.
- Support multiple known users and multiple voice profiles.
- Add voice activity detection to avoid manual Enter-based recording.
- Add conversation memory and session logging.
- Package the RAG backend with clear indexing and retrieval instructions.

## License

No license has been specified yet. Add a license file before publishing the repository if the project will be shared publicly.
