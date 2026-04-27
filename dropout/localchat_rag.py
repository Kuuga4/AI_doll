import torch
import time
import numpy as np
import threading
from queue import Queue
import whisper
import sounddevice as sd
from rich.console import Console
from TTS.api import TTS
import uuid
import os
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

console = Console()
load_dotenv()

# TTS
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# SAR
stt = whisper.load_model("turbo", device=device)

# Chroma path
directory_path = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = directory_path + "\src\chroma"

# 提示词模板
PROMPT_TEMPLATE = """
以下面检索到的内容为补充，需要充分借鉴提供的信息：

{context}

---

充分借鉴上面的内容来回答这个问题,尽量简洁明了,不超过50字：

{question}
"""

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
    result = stt.transcribe(audio_np, fp16=False)
    text = result["text"].strip()
    return text

def play_audio(sample_rate, audio_array):
    sd.play(audio_array, sample_rate)
    sd.wait()

def play_audio_from_buffer(audio_buffer):
    while True:
        if audio_buffer.qsize() >= 2:  # 确保缓冲区中至少有两段音频
            data1 = audio_buffer.get()
            play_audio(24000, data1)  # 播放第一段音频
        else:
            time.sleep(0.01)  # 如果缓冲区不足，稍等再检查，减少等待时间以提高响应速度

def tts_generate(text_queue, audio_buffer):
    while True:
        text = text_queue.get()
        if text is None:  # Check for termination signal
            break
        if not text.strip():  # 检查文本是否为空
            console.print("[red]Received empty text for TTS. Skipping...")
            continue  # 跳过空文本
        
        # 直接生成音频数据
        audio_data = tts.tts(text=text, language="zh-cn", speaker_wav="target_voice/jarvis.mp3")  # 生成音频数据

        # 将生成的音频数据放入缓冲区
        audio_buffer.put(audio_data)

def query_rag(query_text: str, text_queue):
    embedding_function = OllamaEmbeddings(model="nomic-embed-text")

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        base_url="https://api.guidaodeng.com/v1",
    )

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # 实例化检索器
    retriever = db.as_retriever(
        search_kwargs={
            "k": 3}
    )

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt_template
        | llm
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

    answers = []  # 用于存储生成的答案
    for chunk in rag_chain_with_source.stream(query_text):
        for key in chunk:
            answer = chunk.get("answer")
            if answer is not None:
                answers.append(answer)  # 将每个答案添加到列表中
                print(answers)

                # 每10个字作为一组发送
                if len(answers) == 10:
                    combined_answer = " ".join(answers)  # 将10个答案合并为一组
                    text_queue.put(combined_answer)  # 发送到 TTS 线程
                    answers = []  # 重置答案列表

    # 发送剩余的答案（如果有）
    if answers:
        combined_answer = " ".join(answers)
        text_queue.put(combined_answer)  # 发送最后一组答案

    text_queue.put(None)  # 发送终止信号

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

if __name__ == "__main__":
    console.print("[cyan]Assistant started! Press Ctrl+C to exit.")

    try:
        while True:
            console.input(
                "Press Enter to start recording, then press Enter again to stop."
            )

            data_queue = Queue() 
            text_queue = Queue() 
            audio_buffer = Queue()  # 音频缓冲区
            stop_event = threading.Event()
            recording_thread = threading.Thread(
                target=record_audio,
                args=(stop_event, data_queue),
            )
            recording_thread.start()

            input()  # 暂停程序的执行，等待用户的输入
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

                # 启动查询线程
                query_thread = threading.Thread(target=query_rag, args=(text, text_queue))
                query_thread.start()

                # 启动 TTS 线程
                tts_thread = threading.Thread(target=tts_generate, args=(text_queue, audio_buffer))
                tts_thread.start()

                # 启动播放线程
                play_thread = threading.Thread(target=play_audio_from_buffer, args=(audio_buffer,))
                play_thread.start()

                # 等待查询线程结束
                query_thread.join()
                text_queue.put(None)  # 发送终止信号
                tts_thread.join()  # 等待 TTS 线程结束
                play_thread.join()

            else:
                console.print(
                    "[red]No audio recorded. Please ensure your microphone is working."
                )

    except KeyboardInterrupt:
        console.print("\n[red]Exiting...")

    console.print("[blue]Session ended.")