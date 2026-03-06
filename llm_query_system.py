#!/usr/bin/env python3
"""
LLM Query System
================
Universal API client for AI inference services.

Supported services
------------------
  * llama.cpp          – local LLM inference server
  * ollama             – Ollama LLM runner
  * Chatterbox         – ResembleAI Chatterbox TTS
  * Coqui-TTS          – coqui-ai TTS server
  * Stable Diffusion   – AUTOMATIC1111 WebUI image generation

Request formats
---------------
  * json  – structured JSON body (application/json)
  * text  – raw plain-text body (text/plain), sometimes called "toon" format

Usage
-----
  GUI mode (default)::

      python llm_query_system.py

  CLI mode::

      python llm_query_system.py --cli <config.ini> <input.txt>
"""

import base64
import csv
import datetime
import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

try:
    import configparser
except ImportError:
    raise SystemExit("configparser is required (Python standard library)")

try:
    import requests
except ImportError:
    raise SystemExit(
        "The 'requests' library is required.  "
        "Install it with:  pip install requests"
    )

# tkinter is only needed for the GUI; import lazily so the module can be used
# in headless environments (tests, CLI mode, CI pipelines) without errors.
tk = None
ttk = None
filedialog = None
messagebox = None
scrolledtext = None


def _import_tkinter():
    """Lazily import tkinter sub-modules (called only when the GUI starts)."""
    global tk, ttk, filedialog, messagebox, scrolledtext
    if tk is not None:
        return
    try:
        import tkinter as _tk
        from tkinter import ttk as _ttk
        from tkinter import filedialog as _fd
        from tkinter import messagebox as _mb
        from tkinter import scrolledtext as _st
    except ImportError as exc:
        raise SystemExit(
            "tkinter is required for the GUI but is not available in this "
            "Python installation.\n"
            "  • On Ubuntu/Debian: sudo apt-get install python3-tk\n"
            "  • On Windows: re-run the Python installer and enable Tcl/Tk\n"
            f"Original error: {exc}"
        )
    tk = _tk
    ttk = _ttk
    filedialog = _fd
    messagebox = _mb
    scrolledtext = _st

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR = Path("output_result")

# Maps service name → (endpoint_type, default_request_format)
KNOWN_SERVICES: dict[str, dict] = {
    "llama_cpp":        {"type": "llm",   "format": "json"},
    "llama.cpp":        {"type": "llm",   "format": "json"},
    "llamacpp":         {"type": "llm",   "format": "json"},
    "ollama":           {"type": "llm",   "format": "json"},
    "chatterbox":       {"type": "tts",   "format": "json"},
    "coqui_tts":        {"type": "tts",   "format": "json"},
    "coqui-tts":        {"type": "tts",   "format": "json"},
    "coquitts":         {"type": "tts",   "format": "json"},
    "stable_diffusion": {"type": "image", "format": "json"},
    "stable-diffusion": {"type": "image", "format": "json"},
    "stablediffusion":  {"type": "image", "format": "json"},
}


# ─────────────────────────────────────────────────────────────────────────────
# Logging helpers
# ─────────────────────────────────────────────────────────────────────────────

_LOG_FMT = "%(asctime)s | %(levelname)-8s | %(message)s"
_LOG_DATE = "%Y-%m-%d %H:%M:%S"


def _session_id() -> str:
    """Return a timestamp-based session identifier."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:20]


def setup_logger(session_id: str, log_callback=None) -> logging.Logger:
    """
    Create a per-session logger that writes to:
      * output_result/session_<id>.log  (DEBUG level)
      * stderr                          (INFO level)
      * GUI callback (if provided)      (INFO level)
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT_DIR / f"session_{session_id}.log"

    logger = logging.getLogger(f"llm_query.{session_id}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    fmt = logging.Formatter(_LOG_FMT, datefmt=_LOG_DATE)

    # File handler – full debug output
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Stderr handler – INFO+
    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # Optional GUI callback handler
    if log_callback is not None:
        cb_fmt = logging.Formatter("%(levelname)s | %(message)s")
        ch = _CallbackHandler(log_callback)
        ch.setLevel(logging.INFO)
        ch.setFormatter(cb_fmt)
        logger.addHandler(ch)

    return logger


class _CallbackHandler(logging.Handler):
    """Logging handler that forwards messages to a callable."""

    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def emit(self, record):
        try:
            self.callback(self.format(record) + "\n")
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Configuration loading
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    """
    Parse an INI configuration file and return a normalised dictionary.

    Required sections: [service], [endpoint]
    Optional sections: [model], [request], [parameters]
    """
    cfg = configparser.ConfigParser(
        inline_comment_prefixes=("#", ";"),
        default_section="DEFAULT",
    )
    read_files = cfg.read(config_path, encoding="utf-8")
    if not read_files:
        raise FileNotFoundError(f"Config file not found: {config_path}")

    for required in ("service", "endpoint"):
        if required not in cfg:
            raise ValueError(
                f"Config file is missing the required [{required}] section: "
                f"{config_path}"
            )

    service_name = cfg["service"].get("name", "").strip().lower()
    service_type = cfg["service"].get("type", "").strip().lower()

    if not service_name:
        raise ValueError("[service] name must not be empty")
    if service_type not in ("llm", "tts", "image", ""):
        raise ValueError(
            f"[service] type must be 'llm', 'tts', or 'image', got: {service_type!r}"
        )

    # Fall back to the known-service defaults when type is omitted
    if not service_type:
        known = KNOWN_SERVICES.get(service_name, {})
        service_type = known.get("type", "llm")

    endpoint_url = cfg["endpoint"].get("url", "").strip()
    if not endpoint_url:
        raise ValueError("[endpoint] url must not be empty")

    model = cfg["model"].get("name", "").strip() if "model" in cfg else ""
    req_format = (
        cfg["request"].get("format", "auto").strip().lower()
        if "request" in cfg
        else "auto"
    )
    parameters = dict(cfg["parameters"]) if "parameters" in cfg else {}

    return {
        "service_name":   service_name,
        "service_type":   service_type,
        "endpoint_url":   endpoint_url,
        "model":          model,
        "request_format": req_format,
        "parameters":     parameters,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Input reading
# ─────────────────────────────────────────────────────────────────────────────

def read_input_file(input_path: str) -> str:
    """
    Read a .txt or .md file and return its content as a stripped string.
    Raises ValueError for unsupported file extensions.
    """
    path = Path(input_path)
    if path.suffix.lower() not in (".txt", ".md"):
        raise ValueError(
            f"Input file must have a .txt or .md extension, got: {path.suffix!r}"
        )
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    return path.read_text(encoding="utf-8").strip()


# ─────────────────────────────────────────────────────────────────────────────
# Format detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_format(config: dict) -> str:
    """
    Determine whether to use 'json' or 'text' (plain/toon) request format.

    Priority:
      1. Explicit value in config [request] format  (if not 'auto')
      2. Default for the known service
      3. Fall back to 'json'
    """
    if config["request_format"] not in ("auto", ""):
        return config["request_format"]
    known = KNOWN_SERVICES.get(config["service_name"], {})
    return known.get("format", "json")


# ─────────────────────────────────────────────────────────────────────────────
# Request payload builders (one per service)
# ─────────────────────────────────────────────────────────────────────────────

def _param(params: dict, key: str, default):
    """Return a typed parameter from the params dict, coerced to default's type."""
    raw = params.get(key)
    if raw is None:
        return default
    try:
        return type(default)(raw)
    except (ValueError, TypeError):
        return default


def _build_llama_cpp(prompt: str, config: dict) -> dict:
    """Payload for llama.cpp  /completion  endpoint."""
    p = config["parameters"]
    return {
        "prompt":      prompt,
        "n_predict":   _param(p, "max_tokens", 512),
        "temperature": _param(p, "temperature", 0.7),
        "stream":      _param(p, "stream", "false").lower() == "true",
        "stop":        json.loads(p["stop"]) if "stop" in p else [],
    }


def _build_ollama(prompt: str, config: dict) -> dict:
    """Payload for Ollama  /api/generate  endpoint."""
    p = config["parameters"]
    return {
        "model":  config["model"] or "llama3",
        "prompt": prompt,
        "stream": _param(p, "stream", "false").lower() == "true",
        "options": {
            "temperature": _param(p, "temperature", 0.7),
            "num_predict": _param(p, "max_tokens", 512),
        },
    }


def _build_chatterbox(prompt: str, config: dict) -> dict:
    """Payload for Chatterbox TTS  /generate  endpoint."""
    p = config["parameters"]
    return {
        "text":        prompt,
        "exaggeration": _param(p, "exaggeration", 0.5),
        "cfg_weight":  _param(p, "cfg_weight", 0.5),
        "temperature": _param(p, "temperature", 0.8),
    }


def _build_coqui_tts(prompt: str, config: dict) -> dict:
    """Payload for Coqui-TTS  /api/tts  endpoint."""
    p = config["parameters"]
    return {
        "text":        prompt,
        "speaker_id":  p.get("speaker_id", ""),
        "language_id": p.get("language_id", ""),
        "style_wav":   p.get("style_wav", ""),
    }


def _build_stable_diffusion(prompt: str, config: dict) -> dict:
    """Payload for Stable Diffusion AUTOMATIC1111  /sdapi/v1/txt2img  endpoint."""
    p = config["parameters"]
    payload: dict = {
        "prompt":          prompt,
        "negative_prompt": p.get("negative_prompt", ""),
        "steps":           _param(p, "steps", 20),
        "width":           _param(p, "width", 512),
        "height":          _param(p, "height", 512),
        "cfg_scale":       _param(p, "cfg_scale", 7.0),
        "sampler_name":    p.get("sampler_name", "Euler a"),
        "batch_size":      _param(p, "batch_size", 1),
        "seed":            _param(p, "seed", -1),
    }
    if config["model"]:
        payload["override_settings"] = {"sd_model_checkpoint": config["model"]}
    return payload


_BUILDERS = {
    "llama_cpp":        _build_llama_cpp,
    "llama.cpp":        _build_llama_cpp,
    "llamacpp":         _build_llama_cpp,
    "ollama":           _build_ollama,
    "chatterbox":       _build_chatterbox,
    "coqui_tts":        _build_coqui_tts,
    "coqui-tts":        _build_coqui_tts,
    "coquitts":         _build_coqui_tts,
    "stable_diffusion": _build_stable_diffusion,
    "stable-diffusion": _build_stable_diffusion,
    "stablediffusion":  _build_stable_diffusion,
}


def build_request(prompt: str, config: dict, fmt: str) -> tuple:
    """
    Build the HTTP request components.

    Returns
    -------
    (payload, headers, is_json) where:
      * payload   is either a dict (JSON) or a str (text/toon)
      * headers   is a dict of HTTP headers
      * is_json   True  → call requests.post(json=payload)
                  False → call requests.post(data=payload)
    """
    if fmt == "text":
        return prompt, {"Content-Type": "text/plain; charset=utf-8"}, False

    builder = _BUILDERS.get(config["service_name"])
    if builder:
        payload = builder(prompt, config)
    else:
        # Generic fallback: minimal JSON with prompt and optional model
        payload = {"prompt": prompt}
        if config["model"]:
            payload["model"] = config["model"]

    return payload, {"Content-Type": "application/json"}, True


# ─────────────────────────────────────────────────────────────────────────────
# HTTP call
# ─────────────────────────────────────────────────────────────────────────────

def call_api(
    url: str,
    payload,
    headers: dict,
    is_json: bool,
    logger: logging.Logger,
    timeout: int = 120,
) -> tuple:
    """
    Execute the POST request and return (response, elapsed_seconds).
    Raises requests.HTTPError for non-2xx responses.
    """
    logger.info("POST %s", url)
    logger.debug("Headers: %s", headers)
    if is_json:
        logger.debug("Payload:\n%s", json.dumps(payload, indent=2))
    else:
        preview = payload[:300] + "…" if len(payload) > 300 else payload
        logger.debug("Payload (text/toon): %s", preview)

    t_start = time.perf_counter()
    if is_json:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    else:
        resp = requests.post(
            url, data=payload.encode("utf-8"), headers=headers, timeout=timeout
        )
    elapsed = time.perf_counter() - t_start

    logger.info(
        "Response: HTTP %s  |  %.3f s  |  %d bytes",
        resp.status_code,
        elapsed,
        len(resp.content),
    )
    resp.raise_for_status()
    return resp, elapsed


# ─────────────────────────────────────────────────────────────────────────────
# Response parsing & output saving
# ─────────────────────────────────────────────────────────────────────────────

def _extract_llm_text(resp: requests.Response) -> str:
    """
    Extract the generated text from an LLM response.
    Handles Ollama, llama.cpp, and OpenAI-compatible formats.
    """
    try:
        data = resp.json()
    except ValueError:
        return resp.text

    # Ollama /api/generate
    if "response" in data:
        return data["response"]

    # llama.cpp /v1/chat/completions  or  OpenAI-compatible
    choices = data.get("choices")
    if choices:
        choice = choices[0]
        # Chat completion
        msg = choice.get("message", {})
        if "content" in msg:
            return msg["content"]
        # Text completion
        if "text" in choice:
            return choice["text"]

    # llama.cpp /completion
    if "content" in data:
        return data["content"]

    # Fallback: pretty-print the whole JSON
    return json.dumps(data, indent=2, ensure_ascii=False)


def parse_and_save(
    resp: requests.Response,
    config: dict,
    session_id: str,
    logger: logging.Logger,
) -> Path:
    """
    Interpret the HTTP response according to service type and write the output
    to the output_result/ folder.

    Returns the Path of the primary output file.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stype = config["service_type"]

    # ── LLM ──────────────────────────────────────────────────────────────────
    if stype == "llm":
        text = _extract_llm_text(resp)
        out_path = OUTPUT_DIR / f"response_{session_id}.txt"
        out_path.write_text(text, encoding="utf-8")
        logger.info("LLM response saved → %s", out_path)
        return out_path

    # ── TTS ──────────────────────────────────────────────────────────────────
    if stype == "tts":
        ct = resp.headers.get("Content-Type", "")
        if "mpeg" in ct or "mp3" in ct:
            ext = "mp3"
        elif "ogg" in ct:
            ext = "ogg"
        else:
            ext = "wav"
        out_path = OUTPUT_DIR / f"audio_{session_id}.{ext}"
        out_path.write_bytes(resp.content)
        logger.info("TTS audio saved → %s", out_path)
        return out_path

    # ── Image ─────────────────────────────────────────────────────────────────
    if stype == "image":
        data = resp.json()
        images = data.get("images", [])
        if not images:
            raise ValueError(
                "Stable Diffusion returned no images. "
                "Check that the server is running and the prompt is valid."
            )
        primary = OUTPUT_DIR / f"image_{session_id}.png"
        primary.write_bytes(base64.b64decode(images[0]))
        logger.info("Image saved → %s", primary)

        for idx, img_b64 in enumerate(images[1:], start=1):
            extra = OUTPUT_DIR / f"image_{session_id}_{idx}.png"
            extra.write_bytes(base64.b64decode(img_b64))
            logger.info("Additional image saved → %s", extra)

        return primary

    # ── Unknown / generic ─────────────────────────────────────────────────────
    out_path = OUTPUT_DIR / f"output_{session_id}.bin"
    out_path.write_bytes(resp.content)
    logger.warning("Unknown service type %r; raw bytes saved → %s", stype, out_path)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark / performance report
# ─────────────────────────────────────────────────────────────────────────────

def write_benchmark(
    session_id: str,
    config: dict,
    input_path: str,
    output_path: Path,
    prompt: str,
    elapsed: float,
    resp: requests.Response,
    logger: logging.Logger,
):
    """
    Append a human-readable benchmark block to the session log and update
    the cumulative benchmark.csv in output_result/.
    """
    now = datetime.datetime.now().isoformat(timespec="milliseconds")

    # ── Token estimates ───────────────────────────────────────────────────────
    # Rough rule-of-thumb: 1 token ≈ 4 characters
    prompt_tokens = max(1, len(prompt) // 4)

    resp_data: dict = {}
    try:
        resp_data = resp.json()
    except ValueError:
        pass

    completion_tokens = (
        resp_data.get("usage", {}).get("completion_tokens")
        or resp_data.get("eval_count")
        or max(1, len(resp.text) // 4)
    )
    total_tokens = prompt_tokens + completion_tokens
    tps = completion_tokens / elapsed if elapsed > 0 else 0.0

    sep = "─" * 72
    report = "\n".join([
        "",
        "═" * 72,
        f"  BENCHMARK REPORT  ·  Session {session_id}",
        "═" * 72,
        f"  Timestamp           : {now}",
        f"  Service Name        : {config['service_name']}",
        f"  Service Type        : {config['service_type']}",
        f"  Endpoint URL        : {config['endpoint_url']}",
        f"  Model               : {config.get('model') or 'N/A'}",
        f"  Request Format      : {config.get('_resolved_format', 'json')}",
        f"  Input File          : {input_path}",
        f"  Output File         : {output_path}",
        sep,
        f"  ⏱  Elapsed Time      : {elapsed:.3f} s",
        f"  📊  Est. Prompt Tok   : {prompt_tokens:,}",
        f"  📊  Est. Completion   : {completion_tokens:,}",
        f"  📊  Total Tokens      : {total_tokens:,}",
        f"  🚀  Tokens / Second   : {tps:.2f}",
        f"  📡  HTTP Status       : {resp.status_code}",
        f"  📦  Response Size     : {len(resp.content):,} bytes",
        "═" * 72,
        "",
    ])
    logger.info(report)

    # ── Cumulative CSV ────────────────────────────────────────────────────────
    csv_path = OUTPUT_DIR / "benchmark.csv"
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "session_id", "timestamp", "service_name", "service_type",
                "model", "elapsed_s", "prompt_tokens", "completion_tokens",
                "tokens_per_sec", "http_status", "response_bytes",
                "input_file", "output_file",
            ])
        writer.writerow([
            session_id, now,
            config["service_name"], config["service_type"],
            config.get("model", ""),
            f"{elapsed:.3f}", prompt_tokens, completion_tokens,
            f"{tps:.2f}", resp.status_code, len(resp.content),
            input_path, output_path,
        ])
    logger.info("Benchmark CSV updated → %s", csv_path)


# ─────────────────────────────────────────────────────────────────────────────
# Core orchestration
# ─────────────────────────────────────────────────────────────────────────────

def run_query(
    config_path: str,
    input_path: str,
    log_callback=None,
    timeout: int = 120,
) -> tuple:
    """
    Execute a complete query cycle:
      load config → read input → build request → call API →
      save output → write benchmark log

    Parameters
    ----------
    config_path  : path to the INI configuration file
    input_path   : path to the .txt or .md input file
    log_callback : optional callable(str) for live log streaming (used by GUI)
    timeout      : HTTP request timeout in seconds

    Returns
    -------
    (output_path: Path, session_id: str)
    """
    sid = _session_id()
    logger = setup_logger(sid, log_callback)

    logger.info("═══ Session %s started ═══", sid)

    # 1. Load configuration
    config = load_config(config_path)
    logger.info(
        "Config loaded: service=%s  type=%s  url=%s",
        config["service_name"],
        config["service_type"],
        config["endpoint_url"],
    )

    # 2. Read input
    prompt = read_input_file(input_path)
    logger.info(
        "Input loaded: %d chars from %s", len(prompt), input_path
    )

    # 3. Detect format
    fmt = detect_format(config)
    config["_resolved_format"] = fmt
    logger.info(
        "Request format: %s  (%s)",
        fmt,
        "JSON payload" if fmt == "json" else "plain-text / toon",
    )

    # 4. Build request
    payload, headers, is_json = build_request(prompt, config, fmt)

    # 5. Call API
    resp, elapsed = call_api(
        config["endpoint_url"], payload, headers, is_json, logger, timeout
    )

    # 6. Save output
    out_path = parse_and_save(resp, config, sid, logger)

    # 7. Benchmark report
    write_benchmark(sid, config, input_path, out_path, prompt, elapsed, resp, logger)

    logger.info("═══ Session %s complete ═══", sid)
    return out_path, sid


# ─────────────────────────────────────────────────────────────────────────────
# GUI
# ─────────────────────────────────────────────────────────────────────────────

class App:
    """Main application window (tkinter GUI)."""

    def __init__(self):
        _import_tkinter()
        self.root = tk.Tk()
        self.root.title("LLM Query System")
        self.root.geometry("860x660")
        self.root.minsize(640, 480)
        self.root.resizable(True, True)
        self._build_ui()

    def mainloop(self):
        self.root.mainloop()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        root = self.root
        pad = {"padx": 10, "pady": 5}

        # Config file row
        tk.Label(root, text="Config File (.ini):", anchor="w").grid(
            row=0, column=0, sticky="w", **pad
        )
        self.config_var = tk.StringVar()
        tk.Entry(root, textvariable=self.config_var, width=64).grid(
            row=0, column=1, sticky="ew", **pad
        )
        tk.Button(root, text="Browse…", command=self._browse_config).grid(
            row=0, column=2, **pad
        )

        # Input file row
        tk.Label(root, text="Input File (.txt / .md):", anchor="w").grid(
            row=1, column=0, sticky="w", **pad
        )
        self.input_var = tk.StringVar()
        tk.Entry(root, textvariable=self.input_var, width=64).grid(
            row=1, column=1, sticky="ew", **pad
        )
        tk.Button(root, text="Browse…", command=self._browse_input).grid(
            row=1, column=2, **pad
        )

        # Timeout row
        tk.Label(root, text="Timeout (seconds):", anchor="w").grid(
            row=2, column=0, sticky="w", **pad
        )
        self.timeout_var = tk.StringVar(value="120")
        tk.Entry(root, textvariable=self.timeout_var, width=10).grid(
            row=2, column=1, sticky="w", **pad
        )

        # Action buttons
        btn_frame = tk.Frame(root)
        btn_frame.grid(row=3, column=0, columnspan=3, sticky="w", **pad)

        self.submit_btn = tk.Button(
            btn_frame,
            text="▶  Send Request",
            command=self._on_submit,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 11, "bold"),
            padx=12,
            pady=6,
        )
        self.submit_btn.pack(side="left", padx=(0, 10))

        tk.Button(
            btn_frame,
            text="📁  Open Output Folder",
            command=self._open_output,
            padx=8,
            pady=6,
        ).pack(side="left")

        tk.Button(
            btn_frame,
            text="🗑  Clear Log",
            command=self._log_clear,
            padx=8,
            pady=6,
        ).pack(side="left", padx=(10, 0))

        # Log area
        tk.Label(root, text="Log output:", anchor="w").grid(
            row=4, column=0, sticky="w", padx=10, pady=(8, 0)
        )
        self.log_area = scrolledtext.ScrolledText(
            root,
            height=24,
            width=100,
            state="disabled",
            font=("Courier New", 9),
            wrap="word",
        )
        self.log_area.grid(
            row=5, column=0, columnspan=3, sticky="nsew", padx=10, pady=5
        )

        # Status bar
        self.status_var = tk.StringVar(value="Ready.")
        tk.Label(
            root,
            textvariable=self.status_var,
            anchor="w",
            relief="sunken",
            bg="#f0f0f0",
        ).grid(row=6, column=0, columnspan=3, sticky="ew", padx=5, pady=(0, 5))

        root.columnconfigure(1, weight=1)
        root.rowconfigure(5, weight=1)

    # ── Event handlers ────────────────────────────────────────────────────────

    def _browse_config(self):
        path = filedialog.askopenfilename(
            title="Select Config File",
            filetypes=[("INI Config", "*.ini"), ("All Files", "*.*")],
        )
        if path:
            self.config_var.set(path)

    def _browse_input(self):
        path = filedialog.askopenfilename(
            title="Select Input File",
            filetypes=[
                ("Text / Markdown", "*.txt *.md"),
                ("All Files", "*.*"),
            ],
        )
        if path:
            self.input_var.set(path)

    def _on_submit(self):
        config_path = self.config_var.get().strip()
        input_path = self.input_var.get().strip()

        if not config_path:
            messagebox.showerror("Missing Input", "Please select a config file.")
            return
        if not input_path:
            messagebox.showerror("Missing Input", "Please select an input file.")
            return

        try:
            timeout = int(self.timeout_var.get())
        except ValueError:
            messagebox.showerror("Invalid Timeout", "Timeout must be an integer.")
            return

        self.submit_btn.config(state="disabled")
        self._log_clear()
        self.status_var.set("Running…  please wait.")

        thread = threading.Thread(
            target=self._run_in_thread,
            args=(config_path, input_path, timeout),
            daemon=True,
        )
        thread.start()

    def _run_in_thread(self, config_path: str, input_path: str, timeout: int):
        root = self.root
        try:
            out_path, session_id = run_query(
                config_path,
                input_path,
                log_callback=lambda msg: root.after(0, self._log_append, msg),
                timeout=timeout,
            )
            root.after(0, self.status_var.set, f"✓ Done — output: {out_path}")
            root.after(
                0,
                messagebox.showinfo,
                "Query Complete",
                f"Output saved to:\n{out_path}",
            )
        except Exception as exc:
            root.after(0, self.status_var.set, f"✗ Error: {exc}")
            root.after(0, messagebox.showerror, "Error", str(exc))
        finally:
            root.after(0, self.submit_btn.config, {"state": "normal"})

    def _log_append(self, text: str):
        self.log_area.config(state="normal")
        self.log_area.insert(tk.END, text)
        self.log_area.see(tk.END)
        self.log_area.config(state="disabled")

    def _log_clear(self):
        self.log_area.config(state="normal")
        self.log_area.delete("1.0", tk.END)
        self.log_area.config(state="disabled")

    def _open_output(self):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        target = str(OUTPUT_DIR.resolve())
        if sys.platform == "win32":
            subprocess.Popen(["explorer", target])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", target])
        else:
            subprocess.Popen(["xdg-open", target])


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def _cli_main():
    """Handle  --cli <config.ini> <input.txt/md>  mode."""
    args = sys.argv[sys.argv.index("--cli") + 1:]
    if len(args) < 2:
        print(
            "Usage: python llm_query_system.py --cli <config.ini> <input.txt>",
            file=sys.stderr,
        )
        sys.exit(1)
    out_path, session_id = run_query(args[0], args[1])
    print(f"\nSession : {session_id}")
    print(f"Output  : {out_path}")
    print(f"Logs    : {OUTPUT_DIR / f'session_{session_id}.log'}")


def main():
    if "--cli" in sys.argv:
        _cli_main()
    else:
        app = App()
        app.mainloop()


if __name__ == "__main__":
    main()
