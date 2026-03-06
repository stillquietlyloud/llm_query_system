# llm_query_system

Universal API client that sends **JSON or plain-text (toon) POST requests** to
local AI inference services and saves the results to `output_result/`.

## Supported services

| Service | Endpoint type | Default request format |
|---|---|---|
| **llama.cpp** | LLM | JSON |
| **Ollama** | LLM | JSON |
| **Chatterbox** (ResembleAI TTS) | TTS | JSON |
| **Coqui-TTS** | TTS | JSON |
| **Stable Diffusion** (AUTOMATIC1111 WebUI) | Image generation | JSON |

Any unknown service falls back to a generic JSON payload containing `prompt` and `model`.

---

## Quick start

### Prerequisites

- Python 3.9 or newer
- `pip install requests`

```bash
pip install -r requirements.txt
```

### GUI mode (default)

```bash
python llm_query_system.py
```

The GUI lets you:

1. **Browse** to an endpoint config `.ini` file
2. **Browse** to an input prompt `.txt` or `.md` file
3. Set the HTTP **timeout** (default 120 s)
4. Click **▶ Send Request** to call the API
5. Watch live log output scroll by
6. Open `output_result/` directly from the GUI

### CLI mode

```bash
python llm_query_system.py --cli endpoint_config.ini example_input.txt
```

---

## Configuration file

Copy `endpoint_config.ini`, rename it, and fill in the values.
Every key is documented with inline comments in the template.

### Sections

```ini
[service]
name = ollama          # service name (see table above)
type = llm             # llm | tts | image

[endpoint]
url = http://localhost:11434/api/generate

[model]
name = llama3          # model tag / checkpoint name

[request]
format = auto          # auto | json | text (toon)

[parameters]
max_tokens  = 512
temperature = 0.7
stream      = false
# ... (many more — see endpoint_config.ini for the full list)
```

**`format = text`** sends the raw prompt as a `text/plain` body (sometimes
called *toon* format).  
**`format = auto`** (default) chooses JSON for all built-in services.

---

## Input files

Write your prompt in any `.txt` or `.md` file:

```
# example_input.txt
Explain how transformers work in simple terms.
```

The system reads the file, strips leading/trailing whitespace, and builds
the correct request payload automatically.

---

## Output

All outputs land in `output_result/`:

| File | Content |
|---|---|
| `response_<session>.txt` | Generated text (LLM) |
| `audio_<session>.wav/.mp3/.ogg` | Synthesised speech (TTS) |
| `image_<session>.png` | Generated image (Stable Diffusion) |
| `session_<session>.log` | Detailed per-session log |
| `benchmark.csv` | Cumulative benchmark data (all sessions) |

### Benchmark log

Each session appends a human-readable block to its log file:

```
════════════════════════════════════════════════════════════════════════
  BENCHMARK REPORT  ·  Session 20240315_143022_12
════════════════════════════════════════════════════════════════════════
  Timestamp           : 2024-03-15T14:30:22.451
  Service Name        : ollama
  Service Type        : llm
  Endpoint URL        : http://localhost:11434/api/generate
  Model               : llama3
  Request Format      : json
  Input File          : example_input.txt
  Output File         : output_result/response_20240315_143022_12.txt
  ────────────────────────────────────────────────────────────────────
  ⏱  Elapsed Time      : 3.741 s
  📊  Est. Prompt Tok   : 73
  📊  Est. Completion   : 128
  📊  Total Tokens      : 201
  🚀  Tokens / Second   : 34.22
  📡  HTTP Status       : 200
  📦  Response Size     : 512 bytes
════════════════════════════════════════════════════════════════════════
```

`benchmark.csv` contains one row per session for easy import into Excel,
pandas, or any other analysis tool.

---

## Building a Windows EXE

```batch
build.bat
```

This installs PyInstaller and produces `dist\LLMQuerySystem.exe` — a
standalone single-file executable that runs without a Python installation.

On Linux / macOS:

```bash
chmod +x build.sh && ./build.sh
```

---

## Running tests

```bash
python -m pytest tests/ -v
```

---

## Project structure

```
llm_query_system/
├── llm_query_system.py      # main application (GUI + CLI + core logic)
├── endpoint_config.ini      # heavily-commented config template
├── example_input.txt        # sample LLM prompt
├── example_input.md         # sample Stable Diffusion prompt
├── requirements.txt         # Python dependencies
├── build.bat                # Windows exe build script
├── build.sh                 # Linux/macOS build script
├── tests/
│   └── test_llm_query_system.py
└── output_result/           # created automatically at runtime
    ├── response_*.txt
    ├── audio_*.wav
    ├── image_*.png
    ├── session_*.log
    └── benchmark.csv
```
