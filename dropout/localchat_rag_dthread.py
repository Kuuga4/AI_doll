import time
import datetime
import threading
import numpy as np
import whisper
import sounddevice as sd
import wave
from queue import Queue
from rich.console import Console
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
import torch
from TTS.api import TTS
import uuid
import src.query as qy
import queue  

console = Console()

# SAR
stt = whisper.load_model("turbo")

# TTS
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def record_audio(stop_event, data_queue):
    def callback(indata, frames, time, status):
        if status:
            console.print(status)
        data_queue.put(bytes(indata))

    with sd.RawInputStream(
        samplerate=16000, dtype="int16", channels=1, callback=callback
    ):
        while not stop_event.is_set():
            time.sleep(0.1)

def transcribe(audio_np: np.ndarray) -> str:
    result = stt.transcribe(audio_np, fp16=True)  # Set fp16=True if using a GPU
    text = result["text"].strip()
    return text

def play_audio(sample_rate, audio_array):
    sd.play(audio_array, sample_rate)
    sd.wait()

def tts_and_play(text_queue):
    while True:
        text = text_queue.get()
        if text is None:  # Check for termination signal
            break
        # Generate audio from text and save to file
        out_path = f"./chatdata/{uuid.uuid4()}.wav"  # Generate a unique file name
        tts.tts_to_file(text=text, file_path=out_path, language="zh-cn", speaker_wav="target_voice/jarvis.mp3",)  # Save audio to file
        # Play the generated audio
        with wave.open(out_path, 'rb') as wf:
            data = wf.readframes(wf.getnframes())
            sample_rate = wf.getframerate()
        data = np.frombuffer(data, dtype=np.int16)  # Adjust dtype as necessary
        play_audio(sample_rate, data)  # Play the generated audio

if __name__ == "__main__":
    console.print("[cyan]Assistant started! Press Ctrl+C to exit.")

    try:
        while True:
            console.input(
                "Press Enter to start recording, then press Enter again to stop."
            )

            data_queue = Queue() 
            text_queue = Queue() 
            stop_event = threading.Event()
            #out_path = f"./chatdata/{uuid.uuid4()}.wav"
            recording_thread = threading.Thread(
                target=record_audio,
                args=(stop_event, data_queue),
            )
            recording_thread.start()

            input() #停程序的执行，等待用户的输入
            stop_event.set()
            recording_thread.join()

            audio_data = b"".join(list(data_queue.queue))
            audio_np = (
                np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            )

            if audio_np.size > 0:
                with console.status("Transcribing...", spinner="earth"):
                    text = transcribe(audio_np)
                console.print(f"[yellow]You: {text}")

                # tts线程
                tts_thread = threading.Thread(target=tts_and_play, args=(text_queue,))
                tts_thread.start()

                with console.status("Generating response...", spinner="earth"):
                    response = qy.query_rag(text)

                # 将 response 每 20 个字符分行
                for i in range(0, len(response), 20):
                    text_queue.put(response[i:i+20])  # 每 20 个字符发送一次
                
                tts_thread.join()  # 等待tts线程结束

            else:
                console.print(
                    "[red]No audio recorded. Please ensure your microphone is working."
                )

    except KeyboardInterrupt:
        console.print("\n[red]Exiting...")

    console.print("[blue]Session ended.")