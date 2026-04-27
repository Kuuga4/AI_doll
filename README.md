# AI_Doll

一个本地语音交互原型：通过**人脸识别门禁**启动会话，使用 **Whisper** 做语音识别，结合 **RAG** 检索生成回答，再用 **XTTS v2** 合成语音播报。

> 当前仓库更偏向课程/实验型项目，不是生产级系统。

---

## 1. 项目能力概览

- 摄像头识别指定用户（默认基于 `imgs/biden.jpg`）。
- 通过回车控制开始/停止录音。
- 使用 Whisper (`turbo`) 将语音转文本。
- 调用 `src/query.py` 中的 RAG 链路生成回答。
- 使用 Coqui TTS 的 `xtts_v2` 按参考音色合成中文语音。
- 终端内完成交互与状态提示（`rich`）。

---

## 2. 实际代码结构（按当前仓库）

```text
AI_Doll/
├── README.md
├── requirement.txt
├── localchat.py                 # 主程序：人脸识别 + 录音 + STT + RAG + TTS
├── imgs/
│   ├── biden.jpg                # 默认人脸参考图
│   └── obama.jpg
├── target_voice/
│   ├── xinmeng_audio.wav        # 默认说话人参考音频（TTS克隆声线）
│   ├── dobby.mp3
│   └── hemine.wav
└── src/
    ├── database.py              # 文档切分与向量库构建（Chroma）
    ├── query.py                 # RAG 查询入口 query_rag
    ├── document/
    │   └── 作品.docx            # 默认知识库文档
    └── chroma/                  # 已存在向量库目录
```

> `dropout/` 下文件看起来是历史实验版本，不是当前主流程入口。

---

## 3. 运行流程

```text
启动 localchat.py
  -> 打开摄像头并进行人脸识别
  -> 识别到目标用户后播放参考音频
  -> 回车开始录音，再次回车停止
  -> Whisper 转写
  -> query_rag() 检索并生成回答
  -> XTTS 合成语音并播放
  -> 循环下一轮对话
```

---

## 4. 环境准备

## 4.1 Python 版本建议

- 建议 Python 3.10 ~ 3.12（项目中依赖较多，版本过新/过旧都可能触发兼容问题）。

## 4.2 创建虚拟环境

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 4.3 安装依赖

本仓库提供了完整依赖锁定文件 `requirement.txt`（UTF-16 编码，带 BOM）。推荐先尝试：

```bash
pip install -r requirement.txt
```

如果你只想先跑主流程，可最小化安装：

```bash
pip install numpy openai-whisper sounddevice rich torch TTS face_recognition opencv-python \
            langchain langchain-openai langchain-community langchain-text-splitters chromadb \
            python-dotenv docx2txt
```

> `face_recognition` 依赖 `dlib`，部分系统需要提前安装 C/C++ 构建工具与 CMake。

---

## 5. 配置说明（与源码一致）

## 5.1 人脸识别配置（`localchat.py`）

- 默认人脸图片：`imgs/biden.jpg`
- 默认识别名称：`Reze`

可按需修改：

```python
biden_image = face_recognition.load_image_file("imgs/biden.jpg")
known_face_names = ["Reze"]
```

## 5.2 语音克隆参考音频（`localchat.py`）

默认使用：

```python
"target_voice/xinmeng_audio.wav"
```

该音频会用于：
- 人脸通过后的提示播放
- XTTS 的 `speaker_wav` 音色克隆

## 5.3 RAG 与模型配置（`src/query.py`）

- 向量库目录：`src/chroma`
- Embedding：`OllamaEmbeddings(model="nomic-embed-text")`
- LLM：`ChatOpenAI(model="gpt-4o")`
- 自定义 API Base URL：`https://api.guidaodeng.com/v1`

请确保：
1. 本机可用 Ollama 且已拉取 `nomic-embed-text`。
2. 你的 OpenAI 兼容接口密钥环境变量已配置（通常是 `OPENAI_API_KEY`）。

---

## 6. 知识库构建（可选但推荐）

当你更新 `src/document/` 下资料后，建议重建向量库：

```bash
cd src
python database.py --reset
python database.py
```

说明：
- `--reset` 会删除 `src/chroma` 后重建。
- 支持加载 `.pdf/.docx/.txt` 文档。

---

## 7. 启动项目

在仓库根目录执行：

```bash
python localchat.py
```

交互方式：
1. 先通过人脸识别。
2. 按一次回车开始录音。
3. 再按一次回车结束录音。
4. 等待转写、检索与语音播报。

按 `Ctrl + C` 退出。

---

## 8. 常见问题排查

### 8.1 摄像头打不开 / 无法识别人脸

- 确认摄像头权限已授予。
- 确认 `imgs/biden.jpg` 包含清晰正脸。
- 可先将目标人脸改为更容易识别的高清正脸图。

### 8.2 麦克风无输入

- 检查系统默认录音设备是否正确。
- 尝试关闭占用麦克风的其他应用。

### 8.3 TTS 很慢或显存不足

- 当前会自动选择 CUDA（若可用），否则走 CPU。
- CPU 下 `xtts_v2` 速度会明显下降，属于正常现象。

### 8.4 RAG 调用失败

重点检查：
- Ollama 服务是否已启动。
- `nomic-embed-text` 模型是否已拉取。
- API Key 与 `base_url` 是否可用。
- `src/chroma` 是否已构建成功。

---

## 9. 已知限制与后续优化建议

当前限制：
- 配置硬编码在 Python 文件中，不便切换环境。
- 缺少完善的异常处理和日志分级。
- 音频交互依赖“回车控制”，可用性一般。
- 没有多用户人脸与多音色管理。

建议优化方向：
- 增加 `.env`/YAML 配置系统。
- 增加设备检测与启动前自检。
- 增加 VAD（语音活动检测）实现自动起停录音。
- 为 `src/database.py` 和 `src/query.py` 增加单元测试。
- 将主流程拆分为可复用模块，便于 Web/API 化部署。
