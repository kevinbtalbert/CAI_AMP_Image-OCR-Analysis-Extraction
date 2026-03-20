import io
import os
import json
import base64
import uuid
import zipfile
import threading
import queue as thread_queue
from datetime import datetime
from pathlib import Path
from typing import Optional

import shutil
import subprocess

import requests
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from openai import OpenAI
from PIL import Image
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

try:
    BASE_DIR = Path(__file__).parent
except NameError:
    BASE_DIR = Path(os.getcwd()) / "3_application"

STATIC_DIR   = BASE_DIR / "static"
EXAMPLES_DIR = BASE_DIR.parent / "data" / "examples"
DATA_DIR     = Path(os.getenv("CDSW_HOME", "/home/cdsw")) / "data"
CONFIG_PATH  = Path(os.getenv("CDSW_HOME", "/home/cdsw")) / ".cai_image_analysis_config.json"

RESULTS_DIR  = DATA_DIR / "_results"   # saved .txt files from batch jobs

DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Vision-capable Ollama models.
# Ordered by OCR fidelity — only models that achieve high verbatim accuracy
# are listed.  qwen2.5vl is the recommended default: purpose-built for
# document understanding, scores 864/1000 on OCRBench and 95.7% on DocVQA.
LOCAL_MODEL_CATALOG = {
    "Qwen2.5-VL 7B  (~6.0 GB · recommended)":  "qwen2.5vl:7b",
    "Qwen2.5-VL 32B  (~21 GB · highest OCR quality)": "qwen2.5vl:32b",
    "Llama 3.2 Vision 11B  (~7.9 GB · legacy)": "llama3.2-vision:11b",
}
LOCAL_MODEL_DEFAULT = "qwen2.5vl:7b"
OLLAMA_BASE_URL     = "http://localhost:11434"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
PDF_EXTENSIONS   = {".pdf"}

USE_CASES = [
    "Transcribing Typed Text",
    "Transcribing Handwritten Text",
    "Transcribing Forms",
    "Complicated Document QA",
    "Unstructured Information → JSON",
    "Summarize Image",
]

# ---------------------------------------------------------------------------
# Per-use-case configuration
#
# schema:  JSON Schema passed to Ollama's `format` parameter.
#          Ollama enforces this using GBNF grammar-constrained sampling —
#          the model is physically unable to produce tokens outside the schema.
#          No post-hoc string matching needed.
# system:  Role and task framing for the model.
# user:    Task instruction sent alongside the image.
# options: Ollama sampling overrides.
# render:  Callable that converts the parsed JSON response into display text.
# ---------------------------------------------------------------------------

def _render_transcription(data: dict) -> str:
    return data.get("transcription", "")


def _render_form(data: dict) -> str:
    lines = []
    for field in data.get("fields", []):
        label = field.get("label", "").strip()
        value = field.get("value", "").strip() or "(blank)"
        if label:
            lines.append(f"{label}: {value}")
    return "\n".join(lines)


def _render_json(data: dict) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


def _render_text(data: dict) -> str:
    return data.get("text", "")


USE_CASE_CONFIG: dict[str, dict] = {
    "Transcribing Typed Text": {
        # No JSON schema — grammar constraints break on quotes/backslashes that appear
        # verbatim in source content (e.g. code strings, file paths).  Plain-text
        # streaming with strong prompts + low temperature handles all content correctly.
        "schema": None,
        "system": (
            "You are a precise OCR engine. Output the verbatim text from the image "
            "exactly as it appears — every character, symbol, line break, and "
            "punctuation mark. Do not add explanations, summaries, or any text that "
            "does not physically appear in the image. Stop when you reach the last "
            "visible character. "
            "IMPORTANT: if any section of text is too small, blurry, or visually "
            "complex to read with certainty (such as long base64 strings, cryptographic "
            "hashes, or densely packed technical codes), write [illegible] in place of "
            "that section. Never guess or reconstruct content you cannot clearly see."
        ),
        "user": "Transcribe all printed or typed text visible in this image.",
        "render": None,
        "options": {"temperature": 0.05, "repeat_penalty": 1.5, "repeat_last_n": 256,
                    "top_k": 10, "top_p": 0.5, "num_ctx": 16384},
    },
    "Transcribing Handwritten Text": {
        # Same reasoning as Transcribing Typed Text.
        "schema": None,
        "system": (
            "You are a precise handwriting OCR engine. Output the verbatim handwritten "
            "text from the image exactly as written, preserving line breaks. Mark "
            "illegible words as [illegible]. Do not add explanations, summaries, or any "
            "text that does not physically appear in the image. Stop when you reach "
            "the last visible word."
        ),
        "user": "Transcribe all handwritten text visible in this image.",
        "render": None,
        "options": {"temperature": 0.05, "repeat_penalty": 1.5, "repeat_last_n": 256,
                    "top_k": 10, "top_p": 0.5, "num_ctx": 16384},
    },
    "Transcribing Forms": {
        "schema": {
            "type": "object",
            "properties": {
                "fields": {
                    "type": "array",
                    "description": "All fields in the form, in document order.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string", "description": "Field label exactly as printed."},
                            "value": {"type": "string", "description": "Field value as filled in. Empty string if blank."},
                        },
                        "required": ["label", "value"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["fields"],
            "additionalProperties": False,
        },
        "system": (
            "You are a form data extractor. Populate the 'fields' array with every "
            "label/value pair from the form, in document order. Use the exact printed "
            "label text. Leave value as an empty string for blank fields."
        ),
        "user": "Extract every field and its filled-in value from this form.",
        "render": _render_form,
        "options": {"temperature": 0.05, "repeat_penalty": 1.3, "repeat_last_n": 64,
                    "top_k": 20, "top_p": 0.6, "num_ctx": 16384},
    },
    "Complicated Document QA": {
        "schema": {
            "type": "object",
            "properties": {
                "answer":  {"type": "string", "description": "Direct answer to the question."},
                "section": {"type": "string", "description": "Document section or location where the answer was found. Empty string if not applicable."},
            },
            "required": ["answer", "section"],
            "additionalProperties": False,
        },
        "system": (
            "You are a document analyst. Answer using ONLY information present in the "
            "document image. If the answer is not in the document, set answer to "
            "'Not found in document'. Cite the section or location where you found it."
        ),
        "user": "Answer the following question based on this document: {question}",
        "render": lambda d: (
            d.get("answer", "") +
            (f"\n\n[Source: {d['section']}]" if d.get("section") else "")
        ),
        "options": {"temperature": 0.2, "repeat_penalty": 1.15, "repeat_last_n": 64,
                    "top_k": 40, "top_p": 0.9},
    },
    "Unstructured Information → JSON": {
        # No fixed schema — the model produces domain-specific keys.
        # Ollama's format="json" still enforces valid JSON at the grammar level.
        "schema": "json",
        "system": (
            "You are a structured data extractor. Output valid JSON only — "
            "no markdown, no commentary. Identify all logical fields, group "
            "related information, and use snake_case keys. Never invent data "
            "not present in the image."
        ),
        "user": "Convert all information in this document into structured JSON.",
        "render": lambda d: json.dumps(d, indent=2, ensure_ascii=False),
        "options": {"temperature": 0.1, "repeat_penalty": 1.2, "repeat_last_n": 64,
                    "top_k": 20, "top_p": 0.7},
    },
    "Summarize Image": {
        "schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "Clear prose summary of the image content, purpose, and key information."},
            },
            "required": ["summary"],
            "additionalProperties": False,
        },
        "system": (
            "You are an expert at analysing images and documents. Populate the "
            "'summary' field with a clear, informative prose summary covering the "
            "content, purpose, and key takeaways of the image."
        ),
        "user": "Summarise the content of this image.",
        "render": lambda d: d.get("summary", ""),
        "options": {"temperature": 0.3, "repeat_penalty": 1.15, "repeat_last_n": 64,
                    "top_k": 40, "top_p": 0.9},
    },
}

# Flat prompt map — kept for backward-compat with batch worker
USE_CASE_PROMPTS = {k: v["user"] for k, v in USE_CASE_CONFIG.items()}


def _truncate_repetition(text: str, *, min_block: int = 2, max_repeats: int = 3) -> str:
    """
    Detect and remove repeating line-blocks that indicate the model has entered
    a hallucination loop (e.g. the same MIME header pair repeated hundreds of times).

    Strategy: slide a window of *min_block* consecutive lines and check whether
    the same block appears more than *max_repeats* times in a row.  If so,
    keep the first *max_repeats* occurrences and strip the rest.

    This handles both short patterns (2–4 lines) and exact single-line repeats.
    """
    if not text:
        return text

    lines = text.splitlines()
    n     = len(lines)

    # Try block sizes from small to medium
    for block_size in range(1, min(16, n // (max_repeats + 1) + 1)):
        i = 0
        while i <= n - block_size * (max_repeats + 1):
            block = lines[i : i + block_size]
            # Count how many consecutive times this block repeats after position i
            count = 1
            while i + block_size * (count + 1) <= n and \
                  lines[i + block_size * count : i + block_size * (count + 1)] == block:
                count += 1
            if count > max_repeats:
                # Keep the first max_repeats copies and truncate
                keep_end = i + block_size * max_repeats
                return "\n".join(lines[:keep_end]).rstrip()
            i += 1

    return text


def _parse_structured_response(raw: str, use_case: str) -> str:
    """
    Parse the JSON response from a schema-constrained Ollama call and
    convert it to a human-readable string using the use-case render function.
    Falls back to returning raw text if JSON parsing fails.
    """
    uc = USE_CASE_CONFIG.get(use_case, {})
    render = uc.get("render")
    if not render:
        return raw
    try:
        data = json.loads(raw)
        return render(data)
    except (json.JSONDecodeError, TypeError):
        return raw

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def default_config() -> dict:
    return {
        "local_model": os.getenv("LOCAL_MODEL", LOCAL_MODEL_DEFAULT),
        "max_tokens":  4096,
    }


def load_config() -> dict:
    cfg = default_config()
    if CONFIG_PATH.exists():
        try:
            cfg.update(json.loads(CONFIG_PATH.read_text()))
        except Exception:
            pass
    return cfg


def save_config(cfg: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------

def ollama_installed() -> bool:
    return shutil.which("ollama") is not None


def ollama_running() -> bool:
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _count_nvidia_gpus() -> int:
    """Count available NVIDIA GPUs using nvidia-smi (most reliable source)."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            return len([l for l in r.stdout.strip().splitlines() if l.strip()])
    except Exception:
        pass
    return 0


_OLLAMA_LOCAL    = Path.home() / ".local"
_OLLAMA_LIB_DIR  = _OLLAMA_LOCAL / "lib" / "ollama"
_OLLAMA_BIN_DIR  = _OLLAMA_LOCAL / "bin"


def _ollama_env() -> dict:
    """
    Build the environment for the Ollama subprocess.

    Newer Ollama releases (0.2+) moved CUDA runners to a 'runners/' subdir:
        ~/.local/lib/ollama/runners/cuda_v12/libggml-cuda.so
    Older Ollama stored them directly in lib/ollama/:
        ~/.local/lib/ollama/cuda_v12/libggml-cuda.so

    OLLAMA_RUNNERS_DIR must point to the directory that *contains* the
    cuda_v11 / cuda_v12 subdirectories. We auto-detect which layout is present.
    """
    env = os.environ.copy()

    # ── Detect runners directory layout ──────────────────────────────────────
    # Prefer the new layout (runners/ subdir); fall back to lib/ollama directly.
    _runners_subdir = _OLLAMA_LIB_DIR / "runners"
    if _runners_subdir.is_dir():
        runners_dir = _runners_subdir
    else:
        runners_dir = _OLLAMA_LIB_DIR

    env["OLLAMA_RUNNERS_DIR"] = str(runners_dir)

    # Log what we find inside the runners directory for diagnostics
    try:
        runner_entries = sorted(runners_dir.iterdir())
        runner_names   = [e.name for e in runner_entries] if runner_entries else []
        has_cuda = any("cuda" in n.lower() for n in runner_names)
    except Exception:
        runner_names, has_cuda = [], False

    # ── LD_LIBRARY_PATH — CUDA runners + system CUDA libs ────────────────────
    lib_paths = [
        str(runners_dir),               # runner .so files live here
        str(_OLLAMA_LIB_DIR),           # libggml.so / libllama.so
        str(_OLLAMA_LOCAL / "lib"),
        "/usr/local/cuda/lib64",
        "/usr/local/cuda-12/lib64",
        "/usr/local/cuda-11/lib64",
        "/usr/lib/x86_64-linux-gnu",
    ]
    existing  = env.get("LD_LIBRARY_PATH", "")
    new_paths = ":".join(p for p in lib_paths if p not in existing)
    env["LD_LIBRARY_PATH"] = f"{new_paths}:{existing}" if existing else new_paths

    # ── GPU layer offloading ──────────────────────────────────────────────────
    # CRITICAL: OLLAMA_NUM_GPU is the number of MODEL LAYERS to offload to GPU,
    # NOT the number of physical GPUs.  Setting it to 1 puts only 1 layer on GPU
    # and loads the remaining ~27 layers (~9 GB) into CPU RAM — causing the
    # "model requires more system memory" error when RAM < model size.
    #
    # Strategy:
    #   GPU detected  → do NOT set OLLAMA_NUM_GPU (Ollama auto-detects all layers)
    #                   OR set it to 999 to force all layers on GPU
    #   No GPU        → set OLLAMA_NUM_GPU=0 to explicitly select CPU mode
    n_gpu = _count_nvidia_gpus()
    if n_gpu > 0:
        # Leave OLLAMA_NUM_GPU unset so Ollama auto-detects the layer count,
        # which defaults to "all layers that fit in VRAM".
        env.pop("OLLAMA_NUM_GPU", None)
        # Ensure CUDA_VISIBLE_DEVICES is set so the CUDA runtime initialises.
        if not env.get("CUDA_VISIBLE_DEVICES"):
            env["CUDA_VISIBLE_DEVICES"] = "0"
        # Flash Attention reduces KV-cache memory significantly with no quality loss.
        env.setdefault("OLLAMA_FLASH_ATTENTION", "1")
        print(f"[ollama] nvidia-smi found {n_gpu} GPU(s) — CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}, OLLAMA_FLASH_ATTENTION=1")
        print("[ollama] OLLAMA_NUM_GPU not set — Ollama will auto-load all layers onto GPU")
    else:
        env["OLLAMA_NUM_GPU"] = "0"
        print("[ollama] No GPU detected via nvidia-smi — setting OLLAMA_NUM_GPU=0 (CPU mode)")

    # Keep VRAM reservation minimal so the model fits without unnecessary fallback
    env.setdefault("OLLAMA_GPU_OVERHEAD", "0")

    # ── Diagnostics ───────────────────────────────────────────────────────────
    print(f"[ollama] OLLAMA_RUNNERS_DIR={env['OLLAMA_RUNNERS_DIR']}")
    print(f"[ollama] runners/ subdir used: {runners_dir != _OLLAMA_LIB_DIR}")
    print(f"[ollama] runner entries: {runner_names[:8]}")
    print(f"[ollama] CUDA runners found: {has_cuda}")
    print(f"[ollama] lib/ollama exists={_OLLAMA_LIB_DIR.is_dir()}")
    print(f"[ollama] LD_LIBRARY_PATH={env['LD_LIBRARY_PATH'][:140]}…")

    return env


def ollama_stop() -> None:
    """
    Gracefully stop any running Ollama server so we can restart it with the
    correct GPU environment variables.
    """
    try:
        # SIGTERM the ollama process group if we can find it
        result = subprocess.run(
            ["pkill", "-f", "ollama serve"],
            timeout=5, capture_output=True,
        )
        if result.returncode == 0:
            import time; time.sleep(1)  # brief pause for it to stop
    except Exception:
        pass


def ollama_start() -> None:
    if not ollama_installed():
        return
    env = _ollama_env()
    # Ensure ~/.local/bin is on PATH so the ollama binary is found
    local_bin = str(_OLLAMA_BIN_DIR)
    path = env.get("PATH", "")
    if local_bin not in path.split(os.pathsep):
        env["PATH"] = local_bin + os.pathsep + path

    # Write Ollama server output to a log file so GPU detection messages are
    # visible. 'ollama serve' logs CUDA runner selection and GPU info to stderr.
    _log_path = Path.home() / ".local" / "share" / "ollama-serve.log"
    _log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(_log_path, "w")  # noqa: WPS515 (intentionally left open)
    print(f"[ollama] server logs → {_log_path}")

    subprocess.Popen(
        ["ollama", "serve"],
        stdout=log_fh,
        stderr=log_fh,
        start_new_session=True,
        env=env,
    )


def ollama_list_models() -> list[str]:
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


def ollama_ps() -> list[dict]:
    """
    Return currently loaded Ollama models via /api/ps.
    Each entry includes 'size_vram' (bytes on GPU) so we can confirm GPU use.
    Available in Ollama v0.1.33+.
    """
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/ps", timeout=5)
        if r.status_code == 200:
            return r.json().get("models", [])
    except Exception:
        pass
    return []


def get_gpu_info() -> dict:
    """
    Query nvidia-smi for GPU hardware info and utilization.
    Returns a dict with 'available' (bool) and a 'gpus' list.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,utilization.gpu,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            gpus = []
            for line in result.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 5:
                    gpus.append({
                        "name":            parts[0],
                        "memory_total_mb": int(parts[1]),
                        "memory_used_mb":  int(parts[2]),
                        "utilization_pct": int(parts[3]),
                        "driver_version":  parts[4],
                    })
            return {"available": True, "gpus": gpus}
    except FileNotFoundError:
        pass        # nvidia-smi not present — no GPU
    except Exception:
        pass
    return {"available": False, "gpus": []}


# Pull state — one pull at a time
_pull_state: dict = {"model": None, "running": False, "error": None}
_pull_lock = threading.Lock()


def _do_pull(model: str) -> None:
    with _pull_lock:
        _pull_state.update({"model": model, "running": True, "error": None})
    try:
        ollama_bin = shutil.which("ollama")
        if not ollama_bin:
            raise FileNotFoundError(
                "Ollama binary not found. Run the '2_setup_models' setup job first."
            )
        result = subprocess.run(
            [ollama_bin, "pull", model],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=7200,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ollama pull exited with code {result.returncode}. "
                + (result.stdout[-300:] if result.stdout else "")
            )
    except Exception as e:
        with _pull_lock:
            _pull_state.update({"running": False, "error": str(e)[:300]})
        return
    with _pull_lock:
        _pull_state.update({"running": False, "error": None})


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

# qwen2.5vl uses dynamic resolution — it divides images into 28×28 patches
# and can process up to 1280px on the longest side without degradation.
# Higher resolution improves OCR accuracy on dense text (code, forms, small print).
_MAX_IMAGE_PX = 1280
_JPEG_QUALITY = 92


def prepare_image(path: Path) -> tuple[str, str]:
    """
    Resize the image to fit within _MAX_IMAGE_PX on the longest side, convert
    to RGB JPEG, and return (base64_string, mime_type).

    This reduces payload size by ~5-20× for typical document scans and keeps
    the model's attention focused where it matters.
    """
    img = Image.open(path)

    # Flatten transparency / palette modes to plain RGB
    if img.mode not in ("RGB", "L"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "RGBA":
            background.paste(img, mask=img.split()[3])
        else:
            background.paste(img.convert("RGB"))
        img = background
    elif img.mode == "L":
        img = img.convert("RGB")

    # Downscale if needed (preserve aspect ratio, no upscaling)
    w, h = img.size
    if max(w, h) > _MAX_IMAGE_PX:
        scale = _MAX_IMAGE_PX / max(w, h)
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=_JPEG_QUALITY, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return b64, "image/jpeg"


def list_images(directory: Path) -> list[dict]:
    if not directory.exists():
        return []
    return [
        {"name": f.name, "size": f.stat().st_size,
         "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat()}
        for f in sorted(directory.iterdir())
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ]


# ---------------------------------------------------------------------------
# Vision LLM — Ollama
# ---------------------------------------------------------------------------

def _use_case_options(use_case: str, cfg: dict) -> dict:
    """
    Return merged Ollama sampling options for a given use case.

    For schema-constrained use cases the grammar is the natural stop signal,
    so num_predict is set to -2 (fill context) unless the user has set a
    lower explicit cap via the Configuration panel.  This prevents the output
    from being truncated before the model finishes the structured field.

    num_gpu is always set explicitly so inference requests force GPU usage
    even if the Ollama server was inadvertently started without GPU env vars.
    """
    uc = USE_CASE_CONFIG.get(use_case, {})
    has_schema = bool(uc.get("schema"))

    user_max = cfg.get("max_tokens")
    # Schema-constrained and plain-text transcription use cases both benefit from
    # filling the context window — let the model or grammar decide when to stop.
    is_open_ended = has_schema or uc.get("options", {}).get("num_ctx", 0) >= 16384
    if is_open_ended:
        base_predict = user_max if user_max else -2
    else:
        base_predict = user_max if user_max else 4096

    # num_gpu in the Ollama request = number of *model layers* to offload to GPU.
    # 999 means "all layers" on all platforms (safe large number; Ollama caps it
    # at the actual layer count). Do NOT use -1 — its behaviour on Linux varies
    # across Ollama versions and may default to CPU instead of all-GPU.
    base    = {"num_predict": base_predict, "num_gpu": 999}
    uc_opts = uc.get("options", {})
    return {**base, **uc_opts}


def _build_native_messages(image_path: Path, use_case: str, cfg: dict | None = None) -> list[dict]:
    """Build the Ollama-native /api/chat messages list for a use case."""
    uc     = USE_CASE_CONFIG.get(use_case, {})
    system = uc.get("system", "")
    user   = uc.get("user", "Describe this image.")
    if cfg and cfg.get("_qa_question") and "{question}" in user:
        user = user.replace("{question}", cfg["_qa_question"])
    b64, _ = prepare_image(image_path)

    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user, "images": [b64]})
    return msgs


def analyze_image(image_path: Path, use_case: str, cfg: dict) -> str:
    """
    Send an image to the local Ollama vision model (blocking, for batch jobs).

    schema=None  → plain-text, no format constraint (handles any special chars).
    schema="json" → valid JSON enforced by grammar, open keys.
    schema=<dict> → fully structured JSON schema, grammar-enforced.
    """
    model    = cfg.get("local_model", LOCAL_MODEL_DEFAULT)
    msgs     = _build_native_messages(image_path, use_case, cfg)
    opts     = _use_case_options(use_case, cfg)
    schema   = USE_CASE_CONFIG.get(use_case, {}).get("schema")

    payload: dict = {
        "model":    model,
        "messages": msgs,
        "stream":   False,
        "options":  opts,
    }
    if schema:
        payload["format"] = schema

    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json=payload,
        timeout=(10, 300),
    )
    resp.raise_for_status()
    raw = resp.json().get("message", {}).get("content", "")
    if schema:
        return _parse_structured_response(raw, use_case)
    return _truncate_repetition(raw)


def stream_analyze_image(image_path: Path, use_case: str, cfg: dict):
    """
    Yield SSE lines ('data: {...}\\n\\n') for the streaming endpoint.

    schema=None  (plain-text transcription):
        Stream tokens directly — no format constraint, handles any character
        including quotes, backslashes, and other special chars verbatim.

    schema="json" (Unstructured → JSON):
        Stream with format="json" so Ollama enforces syntactically valid JSON
        at the grammar level while allowing arbitrary keys.

    schema=<dict> (Forms, QA, Summarize):
        Blocking call with the full JSON Schema format constraint, then deliver
        the rendered result in one SSE token.
    """
    model  = cfg.get("local_model", LOCAL_MODEL_DEFAULT)
    msgs   = _build_native_messages(image_path, use_case, cfg)
    opts   = _use_case_options(use_case, cfg)
    schema = USE_CASE_CONFIG.get(use_case, {}).get("schema")

    def _stream_plain(fmt=None):
        """Stream tokens from Ollama, optionally with a format constraint.

        Includes a running-buffer repetition guard: if the accumulated text
        develops a looping line-block pattern (the same N lines repeating
        more than 3 times) the stream is stopped early and truncated.
        """
        payload: dict = {
            "model":    model,
            "messages": msgs,
            "stream":   True,
            "options":  opts,
        }
        if fmt:
            payload["format"] = fmt
        try:
            with requests.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json=payload,
                stream=True,
                timeout=(10, 300),
            ) as resp:
                if not resp.ok:
                    # Surface the actual Ollama error message, not just the HTTP status
                    try:
                        detail = resp.json().get("error", resp.text)
                    except Exception:
                        detail = resp.text or f"HTTP {resp.status_code}"
                    yield f"data: {json.dumps({'error': detail})}\n\n"
                    return

                accumulated = ""
                sent_chars  = 0  # how many chars of `accumulated` we've already emitted

                for raw in resp.iter_lines():
                    if not raw:
                        continue
                    chunk = json.loads(raw)
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        accumulated += token
                        # Every ~200 accumulated chars, check for repetition
                        if len(accumulated) - sent_chars >= 200:
                            truncated = _truncate_repetition(accumulated)
                            if len(truncated) < len(accumulated):
                                # Repetition detected — emit only the new safe portion
                                new_safe = truncated[sent_chars:]
                                if new_safe:
                                    yield f"data: {json.dumps({'token': new_safe})}\n\n"
                                yield f"data: {json.dumps({'done': True})}\n\n"
                                return
                            # No repetition — emit the unsent portion
                            new_part = accumulated[sent_chars:]
                            yield f"data: {json.dumps({'token': new_part})}\n\n"
                            sent_chars = len(accumulated)

                    if chunk.get("done"):
                        # Emit any buffered remainder, apply final truncation
                        remainder = accumulated[sent_chars:]
                        if remainder:
                            clean = _truncate_repetition(accumulated)[sent_chars:]
                            if clean:
                                yield f"data: {json.dumps({'token': clean})}\n\n"
                        yield f"data: {json.dumps({'done': True})}\n\n"
                        return
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    # Plain-text streaming — no grammar constraint, all chars preserved
    if schema is None:
        yield from _stream_plain()
        return

    # Free-form JSON streaming
    if schema == "json":
        yield from _stream_plain(fmt="json")
        return

    # Structured JSON schema — blocking call, then render and deliver via SSE
    payload = {
        "model":    model,
        "messages": msgs,
        "stream":   False,
        "format":   schema,
        "options":  opts,
    }
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=(10, 300),
        )
        if not resp.ok:
            try:
                detail = resp.json().get("error", resp.text)
            except Exception:
                detail = resp.text or f"HTTP {resp.status_code}"
            yield f"data: {json.dumps({'error': detail})}\n\n"
            return
        raw    = resp.json().get("message", {}).get("content", "")
        result = _parse_structured_response(raw, use_case)
        yield f"data: {json.dumps({'token': result})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


# Set to True while the startup pre-warm is in progress so the UI can
# show a "loading model" state and block premature inference requests.
_model_warming: bool = False
_model_ready:   bool = False


def prewarm_model(model: str, max_attempts: int = 5) -> None:
    """
    Load the model into GPU VRAM by sending a tiny text-only request.

    Retries up to *max_attempts* times with a back-off pause between
    attempts because Ollama sometimes needs a moment after starting before
    CUDA initialises fully (the first request can race the CUDA runtime
    and return a spurious "not enough system memory" error).
    """
    import time as _time

    global _model_warming, _model_ready
    _model_warming = True
    _model_ready   = False

    for attempt in range(1, max_attempts + 1):
        try:
            print(f"[ollama] Pre-warm attempt {attempt}/{max_attempts} for '{model}'…")
            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model":   model,
                    "prompt":  "hi",
                    "stream":  False,
                # num_gpu = 999 forces all model layers onto GPU (safe large number;
                # Ollama caps at the actual layer count automatically).
                "options": {"num_predict": 1, "num_gpu": 999},
                },
                timeout=300,   # large models can take minutes to load into VRAM
            )
            if resp.status_code == 200:
                print(f"[ollama] ✓ Model '{model}' loaded into GPU VRAM on attempt {attempt}.")
                _model_warming = False
                _model_ready   = True
                return
            else:
                body = ""
                try:    body = resp.json().get("error", resp.text)
                except Exception: body = resp.text
                print(f"[ollama] Pre-warm attempt {attempt} returned {resp.status_code}: {body[:200]}")
        except Exception as e:
            print(f"[ollama] Pre-warm attempt {attempt} exception: {e}")

        if attempt < max_attempts:
            wait = attempt * 5   # 5 s, 10 s, 15 s, 20 s …
            print(f"[ollama] Retrying in {wait} s…")
            _time.sleep(wait)

    print(f"[ollama] ✗ Pre-warm failed after {max_attempts} attempts. "
          "Model will load on first inference request.")
    _model_warming = False
    _model_ready   = False


# ---------------------------------------------------------------------------
# Job queue
# ---------------------------------------------------------------------------

_jobs: dict[str, dict] = {}
_job_q: thread_queue.Queue = thread_queue.Queue()
_jobs_lock = threading.Lock()


def _worker():
    while True:
        job_id = _job_q.get()
        if job_id is None:
            break
        with _jobs_lock:
            job = _jobs.get(job_id)
        if not job:
            _job_q.task_done()
            continue

        with _jobs_lock:
            _jobs[job_id]["status"] = "processing"

        try:
            cfg    = load_config()
            folder = job.get("folder", "")

            # Resolve image path: check folder first, then root DATA_DIR, then examples
            if folder:
                image_path = DATA_DIR / folder / job["filename"]
            else:
                image_path = DATA_DIR / job["filename"]
            if not image_path.exists():
                image_path = DATA_DIR / job["filename"]
            if not image_path.exists():
                image_path = EXAMPLES_DIR / job["filename"]

            use_case = job["use_case"]
            # Inject question into config for QA without mutating the global
            effective_cfg = cfg.copy()
            if job.get("question") and use_case == "Complicated Document QA":
                effective_cfg["_qa_question"] = job["question"]

            result = analyze_image(image_path, use_case, effective_cfg)

            # Persist result as a .txt file for later download
            try:
                save_result_txt(job["filename"], job["use_case"], result, folder)
            except Exception:
                pass  # Non-fatal

            with _jobs_lock:
                _jobs[job_id].update({
                    "status":       "complete",
                    "result":       result,
                    "completed_at": datetime.utcnow().isoformat(),
                })
        except Exception as e:
            with _jobs_lock:
                _jobs[job_id].update({
                    "status":       "error",
                    "error":        str(e),
                    "completed_at": datetime.utcnow().isoformat(),
                })
        finally:
            _job_q.task_done()


_worker_thread = threading.Thread(target=_worker, daemon=True)
_worker_thread.start()


def create_job(filename: str, use_case: str, question: str = "", folder: str = "") -> dict:
    job_id = str(uuid.uuid4())
    job = {
        "id":           job_id,
        "filename":     filename,
        "use_case":     use_case,
        "question":     question,
        "folder":       folder,
        "status":       "queued",
        "result":       None,
        "error":        None,
        "created_at":   datetime.utcnow().isoformat(),
        "completed_at": None,
    }
    with _jobs_lock:
        _jobs[job_id] = job
    _job_q.put(job_id)
    return job


# ---------------------------------------------------------------------------
# User info (from CML environment)
# ---------------------------------------------------------------------------

def get_user_info() -> dict:
    """Read personalisation data from the CML runtime environment."""
    full_name = (
        os.environ.get("GIT_AUTHOR_NAME")
        or os.environ.get("GIT_COMMITTER_NAME")
        or ""
    )
    username  = os.environ.get("PROJECT_OWNER", "")
    project   = os.environ.get("CDSW_PROJECT", "image-analysis")
    domain    = os.environ.get("CDSW_DOMAIN", "")
    email     = (
        os.environ.get("GIT_AUTHOR_EMAIL")
        or os.environ.get("GIT_COMMITTER_EMAIL")
        or ""
    )
    # Derive initials for the avatar
    parts    = full_name.split() if full_name else (username[:2].upper() if username else ["?"])
    initials = "".join(p[0].upper() for p in parts[:2]) if full_name else (username[:2].upper() if username else "?")
    return {
        "full_name": full_name or username or "User",
        "username":  username,
        "initials":  initials,
        "project":   project,
        "domain":    domain,
        "email":     email,
        "portal_url": f"https://{domain}" if domain else "",
    }


# ---------------------------------------------------------------------------
# Folder helpers (subdirectories of DATA_DIR, excluding _results)
# ---------------------------------------------------------------------------

_RESERVED_DIRS = {"_results", "examples"}

def list_folders() -> list[dict]:
    folders = []
    for d in sorted(DATA_DIR.iterdir()):
        if d.is_dir() and d.name not in _RESERVED_DIRS and not d.name.startswith("."):
            count = sum(1 for f in d.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS)
            size  = sum(f.stat().st_size for f in d.iterdir() if f.is_file())
            results_dir = RESULTS_DIR / d.name
            result_count = len(list(results_dir.glob("OCR_*.txt"))) if results_dir.exists() else 0
            folders.append({"name": d.name, "count": count, "size": size, "result_count": result_count})
    return folders


def folder_path(name: str) -> Path:
    """Return the resolved path for a named folder (inside DATA_DIR)."""
    # Sanitise: strip slashes and path traversal
    safe = Path(name).name
    if not safe or safe in _RESERVED_DIRS or safe.startswith("."):
        raise HTTPException(status_code=400, detail=f"Invalid folder name: {name!r}")
    return DATA_DIR / safe


# ---------------------------------------------------------------------------
# Result persistence helpers
# ---------------------------------------------------------------------------

def save_result_txt(filename: str, use_case: str, result: str, folder: str = "") -> Path:
    """
    Write a job result as  OCR_<stem>.txt.
    Files are stored under RESULTS_DIR/<folder>/ (or RESULTS_DIR/ for root).
    """
    stem     = Path(filename).stem
    txt_name = f"OCR_{stem}.txt"
    dest_dir = (RESULTS_DIR / folder) if folder else RESULTS_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / txt_name
    out_path.write_text(
        f"File     : {filename}\n"
        f"Folder   : {folder or '(root)'}\n"
        f"Use case : {use_case}\n"
        f"Date     : {datetime.utcnow().isoformat()}Z\n"
        f"{'─'*60}\n\n"
        + result,
        encoding="utf-8",
    )
    return out_path


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ConfigBody(BaseModel):
    local_model: str = LOCAL_MODEL_DEFAULT
    max_tokens:  int = 4096


class PullModelBody(BaseModel):
    model: str


class ProcessBody(BaseModel):
    filename:  str
    use_case:  str
    question:  Optional[str] = ""
    source:    Optional[str] = "uploads"


class BatchBody(BaseModel):
    filenames: list[str]
    use_case:  str
    question:  Optional[str] = ""
    source:    Optional[str] = "uploads"
    folder:    Optional[str] = ""


class FolderBody(BaseModel):
    name: str


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Cloudera Image Analysis")


@app.on_event("startup")
def _startup():
    """
    Always restart Ollama with the correct GPU environment variables, then
    pre-warm the configured model into VRAM.

    We restart unconditionally because a previous Ollama instance (e.g. from
    the setup job or a prior app session) may have been launched without the
    OLLAMA_NUM_GPU / LD_LIBRARY_PATH / OLLAMA_RUNNERS_DIR env vars, causing
    the model to load on CPU RAM instead of GPU VRAM.
    """
    if not ollama_installed():
        print("[startup] Ollama not installed — skipping start.")
        return

    if ollama_running():
        print("[startup] Ollama already running — restarting with GPU env vars…")
        ollama_stop()

    print("[startup] Starting Ollama server with GPU environment…")
    ollama_start()

    # Pre-warm the model in a background thread so startup doesn't block HTTP
    def _warm():
        import time
        # Wait for Ollama to be ready (up to 30 s)
        for _ in range(15):
            if ollama_running():
                break
            time.sleep(2)
        if not ollama_running():
            print("[startup] Ollama did not start in time — skipping pre-warm.")
            return
        model = load_config().get("local_model", LOCAL_MODEL_DEFAULT)
        print(f"[startup] Pre-warming model '{model}' into GPU VRAM…")
        prewarm_model(model)

    threading.Thread(target=_warm, daemon=True).start()


app.mount("/static",   StaticFiles(directory=str(STATIC_DIR)),   name="static")
app.mount("/images",   StaticFiles(directory=str(DATA_DIR)),      name="images")
app.mount("/examples", StaticFiles(directory=str(EXAMPLES_DIR)), name="examples")


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


# ── Config ─────────────────────────────────────────────────────────────────

@app.get("/api/config")
def get_config():
    cfg = load_config()
    return {
        "local_model":         cfg.get("local_model", LOCAL_MODEL_DEFAULT),
        "max_tokens":          cfg.get("max_tokens", 4096),
        "local_model_catalog": LOCAL_MODEL_CATALOG,
        "use_cases":           USE_CASES,
    }


@app.post("/api/config")
def post_config(body: ConfigBody):
    cfg = load_config()
    cfg.update({"local_model": body.local_model, "max_tokens": body.max_tokens})
    save_config(cfg)
    return {"ok": True}


# ── User info ───────────────────────────────────────────────────────────────

@app.get("/api/user-info")
def api_user_info():
    return get_user_info()


# ── Folders ─────────────────────────────────────────────────────────────────

@app.get("/api/folders")
def api_list_folders():
    return {"folders": list_folders()}


@app.post("/api/folders")
def api_create_folder(body: FolderBody):
    p = folder_path(body.name)
    if p.exists():
        raise HTTPException(status_code=409, detail=f"Folder '{body.name}' already exists.")
    p.mkdir(parents=True)
    return {"ok": True, "name": p.name}


@app.delete("/api/folders/{name}")
def api_delete_folder(name: str):
    p = folder_path(name)
    if not p.exists():
        raise HTTPException(status_code=404, detail="Folder not found.")
    shutil.rmtree(p)
    # Also remove the results sub-directory for this folder
    rp = RESULTS_DIR / name
    if rp.exists():
        shutil.rmtree(rp)
    return {"ok": True}


class RenameFolderBody(BaseModel):
    new_name: str


@app.put("/api/folders/{name}/rename")
def api_rename_folder(name: str, body: RenameFolderBody):
    """
    Rename a folder and its associated results directory atomically.
    OCR_*.txt files are preserved — only the containing directory is renamed.
    """
    new_name = body.new_name.strip()
    if not new_name:
        raise HTTPException(status_code=400, detail="New folder name cannot be empty.")
    if "/" in new_name or "\\" in new_name:
        raise HTTPException(status_code=400, detail="Folder name cannot contain path separators.")

    src  = folder_path(name)
    dst  = folder_path(new_name)
    if not src.exists():
        raise HTTPException(status_code=404, detail=f"Folder '{name}' not found.")
    if dst.exists():
        raise HTTPException(status_code=409, detail=f"Folder '{new_name}' already exists.")

    src.rename(dst)

    # Rename the results sub-directory in the same step so OCR files stay linked
    rsrc = RESULTS_DIR / name
    rdst = RESULTS_DIR / new_name
    if rsrc.exists() and not rdst.exists():
        rsrc.rename(rdst)

    return {"ok": True, "old_name": name, "new_name": new_name}


@app.get("/api/folders/{name}/images")
def api_folder_images(name: str):
    p = folder_path(name)
    if not p.exists():
        raise HTTPException(status_code=404, detail="Folder not found.")
    return {"images": list_images(p), "folder": name}


# ── Images ──────────────────────────────────────────────────────────────────

@app.get("/api/images")
def get_images():
    return {
        "uploads":  list_images(DATA_DIR),
        "examples": list_images(EXAMPLES_DIR),
        "folders":  list_folders(),
    }


def _pdf_to_images(pdf_bytes: bytes, stem: str, dest_dir: Path, dpi: int = 200) -> list[str]:
    """
    Render each page of a PDF to a JPEG and save in dest_dir.
    Returns a list of saved filenames (e.g. ["report_page_001.jpg", ...]).

    Uses pypdfium2 (PDFium / BSD-Apache-2.0) — no AGPL dependency.
    PDFium is the same renderer used in Google Chrome.
    """
    try:
        import pypdfium2 as pdfium
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="pypdfium2 is not installed. Run the session install-dependencies step.",
        )

    saved  = []
    doc    = pdfium.PdfDocument(pdf_bytes)
    n      = len(doc)
    pad    = len(str(n))
    scale  = dpi / 72  # PDFium uses 72 DPI as its base unit

    for i in range(n):
        page   = doc[i]
        bitmap = page.render(scale=scale, rotation=0)
        pil_img = bitmap.to_pil()
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        name = f"{stem}_page_{str(i + 1).zfill(pad)}.jpg"
        pil_img.save(dest_dir / name, "JPEG", quality=92)
        saved.append(name)

    return saved


@app.post("/api/images/upload")
async def upload_images(
    files: list[UploadFile] = File(...),
    folder: Optional[str]   = Query(default=""),
):
    """
    Upload images or PDFs.
    PDFs are split into per-page JPEG files and stored alongside normal images.
    Returns a `pdf_pages` dict mapping original PDF name → list of page image names.
    """
    if folder:
        dest_dir = folder_path(folder)
        dest_dir.mkdir(parents=True, exist_ok=True)
    else:
        dest_dir = DATA_DIR

    saved     = []
    pdf_pages = {}   # {original_pdf_name: [page_img_name, ...]}

    for file in files:
        ext  = Path(file.filename).suffix.lower()
        data = await file.read()

        if ext in PDF_EXTENSIONS:
            stem  = Path(file.filename).stem
            pages = _pdf_to_images(data, stem, dest_dir)
            saved.extend(pages)
            pdf_pages[file.filename] = pages

        elif ext in IMAGE_EXTENSIONS:
            dest = dest_dir / Path(file.filename).name
            dest.write_bytes(data)
            saved.append(file.filename)

    return {"saved": saved, "folder": folder, "pdf_pages": pdf_pages}


@app.delete("/api/images/{filename}")
def delete_image(filename: str, folder: Optional[str] = Query(default="")):
    path = (folder_path(folder) / filename) if folder else (DATA_DIR / filename)
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    path.unlink()
    return {"ok": True}


class RenameImageBody(BaseModel):
    new_name: str
    folder:   str = ""


@app.put("/api/images/{filename}/rename")
def rename_image(filename: str, body: RenameImageBody):
    """
    Rename an image file and, if an OCR result exists for it, rename that too.

    The new filename must keep a valid image extension.
    OCR_<old_stem>.txt → OCR_<new_stem>.txt in the same results directory.
    """
    new_name = body.new_name.strip()
    if not new_name:
        raise HTTPException(status_code=400, detail="New filename cannot be empty.")
    if "/" in new_name or "\\" in new_name:
        raise HTTPException(status_code=400, detail="Filename cannot contain path separators.")

    new_ext = Path(new_name).suffix.lower()
    if new_ext not in IMAGE_EXTENSIONS:
        # Preserve original extension if user omitted it
        orig_ext = Path(filename).suffix.lower()
        new_name = Path(new_name).stem + orig_ext

    folder    = body.folder
    img_dir   = folder_path(folder) if folder else DATA_DIR
    src_img   = img_dir / filename
    dst_img   = img_dir / new_name

    if not src_img.exists():
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found.")
    if dst_img.exists():
        raise HTTPException(status_code=409, detail=f"A file named '{new_name}' already exists.")

    src_img.rename(dst_img)

    # Rename the associated OCR result file if it exists
    result_dir  = (RESULTS_DIR / folder) if folder else RESULTS_DIR
    old_txt     = result_dir / f"OCR_{Path(filename).stem}.txt"
    new_txt     = result_dir / f"OCR_{Path(new_name).stem}.txt"
    renamed_txt = False
    if old_txt.exists() and not new_txt.exists():
        # Update the "File:" header inside the txt to reflect the new filename
        try:
            content  = old_txt.read_text(encoding="utf-8")
            content  = content.replace(f"File: {filename}", f"File: {new_name}", 1)
            new_txt.write_text(content, encoding="utf-8")
            old_txt.unlink()
            renamed_txt = True
        except Exception:
            old_txt.rename(new_txt)
            renamed_txt = True

    return {"ok": True, "new_name": new_name, "result_renamed": renamed_txt}


# ── Results download ─────────────────────────────────────────────────────────

@app.get("/api/results/download")
def download_results(folder: Optional[str] = Query(default="")):
    """
    Download result OCR_*.txt files as a zip.
    If `folder` is given, only includes results from RESULTS_DIR/<folder>/.
    Otherwise includes every result across all subdirectories.
    """
    if folder:
        search_dir = RESULTS_DIR / folder
        txt_files  = list(search_dir.glob("OCR_*.txt")) if search_dir.exists() else []
    else:
        # Collect from root + all subdirs
        txt_files = list(RESULTS_DIR.glob("OCR_*.txt")) + list(RESULTS_DIR.glob("*/OCR_*.txt"))

    if not txt_files:
        raise HTTPException(status_code=404, detail="No results available to download.")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in txt_files:
            # Preserve folder structure inside the zip
            arc_name = f.relative_to(RESULTS_DIR)
            zf.write(f, arc_name)
    buf.seek(0)

    zip_name = f"results_{folder}.zip" if folder else "results_all.zip"
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{zip_name}"'},
    )


@app.get("/api/results/export-csv")
def export_results_csv(folder: Optional[str] = Query(default="")):
    """
    Export all OCR results as a CSV file.

    Each row = one document.  Columns:
      file, folder, use_case, date, text / JSON fields (flattened).

    For results whose body is valid JSON, each top-level key becomes a
    column.  For plain-text results, the content goes into a `text` column.
    All files are merged into a single CSV; missing keys are left blank.
    """
    import csv as _csv

    if folder:
        txt_files = list((RESULTS_DIR / folder).glob("OCR_*.txt")) if (RESULTS_DIR / folder).exists() else []
    else:
        txt_files = list(RESULTS_DIR.glob("OCR_*.txt")) + list(RESULTS_DIR.glob("*/OCR_*.txt"))

    if not txt_files:
        raise HTTPException(status_code=404, detail="No results to export.")

    rows = []
    all_json_keys: list[str] = []

    for path in txt_files:
        try:
            raw    = path.read_text(encoding="utf-8", errors="replace")
            lines  = raw.splitlines()
            sep_i  = next((i for i, l in enumerate(lines) if l.startswith("─")), None)
            header = {l.split(":", 1)[0].strip().lower(): l.split(":", 1)[1].strip()
                      for l in lines[:sep_i or 6] if ":" in l}
            body   = "\n".join(lines[sep_i + 1:]).strip() if sep_i is not None else raw.strip()
        except Exception:
            continue

        rel    = path.relative_to(RESULTS_DIR)
        row    = {
            "file":     header.get("file", path.stem[4:]),   # strip OCR_ prefix
            "folder":   rel.parent.name if rel.parent != Path(".") else "",
            "use_case": header.get("use case", ""),
            "date":     header.get("date", ""),
        }

        # Try to parse JSON body — if it succeeds, flatten top-level keys
        try:
            parsed = json.loads(body)
            if isinstance(parsed, dict):
                for k, v in parsed.items():
                    col = str(k).lower().replace(" ", "_")
                    row[col] = json.dumps(v) if isinstance(v, (dict, list)) else str(v)
                    if col not in all_json_keys:
                        all_json_keys.append(col)
            else:
                row["text"] = body
        except (json.JSONDecodeError, TypeError):
            row["text"] = body

        rows.append(row)

    if not rows:
        raise HTTPException(status_code=404, detail="No results to export.")

    # Build a stable, ordered set of columns
    base_cols  = ["file", "folder", "use_case", "date"]
    extra_cols = all_json_keys + (["text"] if any("text" in r for r in rows) else [])
    fieldnames = base_cols + [c for c in extra_cols if c not in base_cols]

    buf = io.StringIO()
    writer = _csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)

    csv_name = f"results_{folder}.csv" if folder else "results_all.csv"
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{csv_name}"'},
    )


@app.get("/api/folders/{name}/results")
def api_folder_results(name: str):
    """List OCR result filenames for a given folder (or 'root')."""
    if name == "root":
        files = [f.name for f in RESULTS_DIR.glob("OCR_*.txt")]
    else:
        d = RESULTS_DIR / name
        files = [f.name for f in d.glob("OCR_*.txt")] if d.exists() else []
    return {"results": files, "folder": name}


@app.get("/api/results/read")
def api_result_read(filename: str, folder: str = ""):
    """
    Return the text content of a single OCR result file.

    Query params:
      filename  – the image filename (e.g. 'photo.png'); resolves to OCR_<stem>.txt
      folder    – optional folder name; empty string means root.
    """
    stem     = Path(filename).stem
    txt_name = f"OCR_{stem}.txt"
    path     = (RESULTS_DIR / folder / txt_name) if folder else (RESULTS_DIR / txt_name)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"No result found for {filename}")
    return {"filename": filename, "folder": folder, "content": path.read_text(encoding="utf-8")}


class SaveResultBody(BaseModel):
    filename: str
    folder:   str = ""
    content:  str


@app.put("/api/results/save")
def api_result_save(body: SaveResultBody):
    """Overwrite the body of an OCR result file (user-edited correction)."""
    stem     = Path(body.filename).stem
    txt_name = f"OCR_{stem}.txt"
    path     = (RESULTS_DIR / body.folder / txt_name) if body.folder else (RESULTS_DIR / txt_name)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"No result found for {body.filename}")
    # Preserve the header block (everything up to and including the separator line),
    # then replace the body with the user's edited text.
    old_lines = path.read_text(encoding="utf-8").splitlines()
    sep_idx   = next((i for i, l in enumerate(old_lines) if l.startswith("─")), None)
    if sep_idx is not None:
        header = "\n".join(old_lines[: sep_idx + 1])
        new_content = header + "\n" + body.content.strip() + "\n"
    else:
        new_content = body.content
    path.write_text(new_content, encoding="utf-8")
    return {"ok": True}


@app.get("/api/results/search")
def api_results_search(q: str = Query(..., min_length=1)):
    """
    Full-text search across all OCR_*.txt result files.
    Returns up to 50 matches with a 160-char snippet around each hit.
    """
    if not q:
        return {"matches": []}

    q_lower  = q.lower()
    matches  = []
    txt_files = list(RESULTS_DIR.glob("OCR_*.txt")) + list(RESULTS_DIR.glob("*/OCR_*.txt"))

    for path in txt_files:
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        # Find the body (after the separator line)
        lines   = content.splitlines()
        sep_idx = next((i for i, l in enumerate(lines) if l.startswith("─")), None)
        body    = "\n".join(lines[sep_idx + 1:]) if sep_idx is not None else content

        idx = body.lower().find(q_lower)
        if idx == -1:
            continue

        # Build a readable snippet around the match
        start   = max(0, idx - 80)
        end     = min(len(body), idx + len(q) + 80)
        snippet = ("…" if start > 0 else "") + body[start:end] + ("…" if end < len(body) else "")

        # Resolve folder from path structure
        rel = path.relative_to(RESULTS_DIR)
        folder = rel.parent.name if rel.parent != Path(".") else ""

        # Extract original image filename from the header if present
        img_filename = path.stem[4:]  # strip "OCR_" prefix, gives stem
        for line in lines[:6]:
            if line.startswith("File:"):
                img_filename = line.split(":", 1)[1].strip()
                break

        matches.append({
            "txt_name":     path.name,
            "img_filename": img_filename,
            "folder":       folder,
            "snippet":      snippet,
            "match_start":  idx - start,   # offset within snippet where match begins
            "match_len":    len(q),
        })
        if len(matches) >= 50:
            break

    return {"matches": matches, "query": q}


# ── Single process ──────────────────────────────────────────────────────────

@app.post("/api/process")
def post_process(body: ProcessBody):
    if not ollama_running():
        raise HTTPException(
            status_code=503,
            detail="Ollama server is not running. Go to Configuration to start it.",
        )

    image_path = (EXAMPLES_DIR if body.source == "examples" else DATA_DIR) / body.filename
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {body.filename}")

    try:
        cfg = load_config()
        if body.question and body.use_case == "Complicated Document QA":
            cfg["_qa_question"] = body.question
        result = analyze_image(image_path, body.use_case, cfg)
        return {"filename": body.filename, "use_case": body.use_case, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/process/stream")
def post_process_stream(body: ProcessBody):
    """
    Like /api/process but streams tokens via Server-Sent Events so the UI
    can display results progressively rather than waiting for the full response.
    """
    if not ollama_running():
        raise HTTPException(
            status_code=503,
            detail="Ollama server is not running. Go to Configuration to start it.",
        )

    image_path = (EXAMPLES_DIR if body.source == "examples" else DATA_DIR) / body.filename
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {body.filename}")

    cfg = load_config()
    if body.question and body.use_case == "Complicated Document QA":
        cfg["_qa_question"] = body.question
    return StreamingResponse(
        stream_analyze_image(image_path, body.use_case, cfg),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering in CML
        },
    )


# ── Batch jobs ──────────────────────────────────────────────────────────────

@app.post("/api/batch")
def post_batch(body: BatchBody):
    if not body.filenames:
        raise HTTPException(status_code=400, detail="No files specified.")
    created = [create_job(f, body.use_case, body.question or "", body.folder or "") for f in body.filenames]
    return {"jobs": created}


@app.get("/api/jobs")
def get_jobs():
    with _jobs_lock:
        return {"jobs": list(_jobs.values())}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.delete("/api/jobs")
def clear_jobs():
    with _jobs_lock:
        completed = [jid for jid, j in _jobs.items() if j["status"] in ("complete", "error")]
        for jid in completed:
            del _jobs[jid]
    return {"cleared": len(completed)}


# ── Ollama ──────────────────────────────────────────────────────────────────

@app.get("/api/ollama/status")
def get_ollama_status():
    installed   = ollama_installed()
    running     = ollama_running() if installed else False
    models      = ollama_list_models() if running else []
    loaded      = ollama_ps() if running else []   # models currently in memory
    gpu         = get_gpu_info()

    # GPU is "allocated" if nvidia-smi reports at least one GPU.
    # We rely on nvidia-smi rather than CUDA_VISIBLE_DEVICES because CML does
    # not always set that variable even when a GPU has been assigned.
    gpu_allocated = gpu["available"] and len(gpu.get("gpus", [])) > 0

    # GPU is "in use" (model loaded into VRAM) when at least one model reports
    # non-zero VRAM. When the model list is empty the GPU is idle but ready.
    gpu_in_use = gpu_allocated and any(m.get("size_vram", 0) > 0 for m in loaded)
    gpu_ready  = gpu_allocated and not gpu_in_use

    with _pull_lock:
        pull = dict(_pull_state)

    return {
        "installed":     installed,
        "running":       running,
        "models":        models,
        "loaded":        loaded,
        "pull":          pull,
        "catalog":       LOCAL_MODEL_CATALOG,
        "gpu":           gpu,
        "gpu_allocated": gpu_allocated,
        "gpu_in_use":    gpu_in_use,
        "gpu_ready":     gpu_ready,
        "model_warming": _model_warming,   # True while startup pre-warm is in progress
        "model_ready":   _model_ready,     # True once pre-warm confirmed success
    }


@app.get("/api/ollama/log")
def get_ollama_log():
    """Return the last 100 lines of the Ollama server log for diagnostics."""
    log_path = Path.home() / ".local" / "share" / "ollama-serve.log"
    if not log_path.exists():
        return {"log": "(log file not found — Ollama may have been started externally)"}
    try:
        lines = log_path.read_text(errors="replace").splitlines()
        return {"log": "\n".join(lines[-100:])}
    except Exception as e:
        return {"log": f"Error reading log: {e}"}


@app.post("/api/ollama/start")
def post_ollama_start():
    if not ollama_installed():
        raise HTTPException(status_code=400, detail="Ollama is not installed. Run the setup job.")
    if ollama_running():
        # Restart to ensure GPU env vars are applied
        ollama_stop()
    ollama_start()
    return {"ok": True, "message": "Ollama (re)started with GPU environment — allow a few seconds to come up"}


@app.post("/api/ollama/pull")
def post_ollama_pull(body: PullModelBody):
    if not ollama_installed():
        raise HTTPException(status_code=400, detail="Ollama is not installed.")
    if not ollama_running():
        raise HTTPException(status_code=400, detail="Ollama server is not running. Start it first.")
    with _pull_lock:
        if _pull_state["running"]:
            return {"ok": False, "message": f"Already pulling {_pull_state['model']}"}
    t = threading.Thread(target=_do_pull, args=(body.model,), daemon=True)
    t.start()
    return {"ok": True, "message": f"Pulling {body.model}…"}
