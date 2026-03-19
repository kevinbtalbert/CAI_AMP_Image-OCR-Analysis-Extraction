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
            "visible character."
        ),
        "user": "Transcribe all printed or typed text visible in this image.",
        "render": None,
        "options": {"temperature": 0.05, "repeat_penalty": 1.4, "repeat_last_n": 128,
                    "top_k": 10, "top_p": 0.5, "num_ctx": 65536},
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
        "options": {"temperature": 0.05, "repeat_penalty": 1.4, "repeat_last_n": 128,
                    "top_k": 10, "top_p": 0.5, "num_ctx": 65536},
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
                    "top_k": 20, "top_p": 0.6, "num_ctx": 65536},
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

    - Prepends ~/.local/lib/ollama (bundled CUDA runners installed by setup)
      and common CUDA library directories to LD_LIBRARY_PATH.
    - Sets OLLAMA_RUNNERS_DIR explicitly so Ollama finds its CUDA runners
      when installed to a non-standard prefix (~/.local/).
    - Sets OLLAMA_NUM_GPU based on the actual GPU count from nvidia-smi.
      Using nvidia-smi is more reliable than CUDA_VISIBLE_DEVICES because CML
      does not always set that variable even when a GPU is allocated.
    """
    env = os.environ.copy()

    # Bundled CUDA runners + system CUDA libraries
    lib_paths = [
        str(_OLLAMA_LIB_DIR),           # bundled runners from setup archive
        str(_OLLAMA_LOCAL / "lib"),
        "/usr/local/cuda/lib64",
        "/usr/lib/x86_64-linux-gnu",
    ]
    existing = env.get("LD_LIBRARY_PATH", "")
    new_paths = ":".join(p for p in lib_paths if p not in existing)
    env["LD_LIBRARY_PATH"] = f"{new_paths}:{existing}" if existing else new_paths

    # Tell Ollama where to find its runners (critical for non-system installs)
    env.setdefault("OLLAMA_RUNNERS_DIR", str(_OLLAMA_LIB_DIR))

    # Tell Ollama explicitly how many GPUs to use
    n_gpu = _count_nvidia_gpus()
    if n_gpu > 0:
        env["OLLAMA_NUM_GPU"] = str(n_gpu)
        print(f"[ollama] nvidia-smi found {n_gpu} GPU(s) — setting OLLAMA_NUM_GPU={n_gpu}")
    else:
        print("[ollama] No GPU detected via nvidia-smi — Ollama will run on CPU")

    # Log key env vars for debugging
    cvd = env.get("CUDA_VISIBLE_DEVICES", "<not set>")
    print(f"[ollama] CUDA_VISIBLE_DEVICES={cvd}")
    print(f"[ollama] OLLAMA_RUNNERS_DIR={env.get('OLLAMA_RUNNERS_DIR')}")
    print(f"[ollama] lib/ollama exists={_OLLAMA_LIB_DIR.is_dir()}")

    return env


def ollama_start() -> None:
    if not ollama_installed():
        return
    env = _ollama_env()
    # Ensure ~/.local/bin is on PATH so the ollama binary is found
    local_bin = str(_OLLAMA_BIN_DIR)
    path = env.get("PATH", "")
    if local_bin not in path.split(os.pathsep):
        env["PATH"] = local_bin + os.pathsep + path
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
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

    base    = {"num_predict": base_predict}
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
    return _parse_structured_response(raw, use_case) if schema else raw


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
        """Stream tokens from Ollama, optionally with a format constraint."""
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
                resp.raise_for_status()
                for raw in resp.iter_lines():
                    if not raw:
                        continue
                    chunk = json.loads(raw)
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        yield f"data: {json.dumps({'token': token})}\n\n"
                    if chunk.get("done"):
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
        resp.raise_for_status()
        raw    = resp.json().get("message", {}).get("content", "")
        result = _parse_structured_response(raw, use_case)
        yield f"data: {json.dumps({'token': result})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


def prewarm_model(model: str) -> None:
    """
    Load the model into VRAM by sending a tiny text-only request.
    Called once on startup so the first real image request doesn't pay
    the cold-load penalty (typically 5-15 s for an 8 B model).
    """
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": model, "prompt": "hi", "stream": False,
                  "options": {"num_predict": 1}},
            timeout=60,
        )
        if resp.status_code == 200:
            print(f"[ollama] Model '{model}' pre-warmed into VRAM.")
        else:
            print(f"[ollama] Pre-warm returned {resp.status_code} — model may load on first request.")
    except Exception as e:
        print(f"[ollama] Pre-warm skipped: {e}")


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
            folders.append({"name": d.name, "count": count, "size": size})
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
    """Write a job result to RESULTS_DIR as a .txt file."""
    stem   = Path(filename).stem
    folder_tag = f"_{folder}" if folder else ""
    txt_name   = f"{stem}{folder_tag}_{use_case.replace(' ', '_')[:30]}.txt"
    out_path   = RESULTS_DIR / txt_name
    out_path.write_text(
        f"File     : {filename}\n"
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
    """Auto-start Ollama and pre-warm the configured model into VRAM."""
    if ollama_installed() and not ollama_running():
        print("[startup] Starting Ollama server…")
        ollama_start()

    # Pre-warm the model in a background thread so startup doesn't block
    def _warm():
        import time
        # Give Ollama a moment to be ready if it just started
        for _ in range(10):
            if ollama_running():
                break
            time.sleep(2)
        if ollama_running():
            model = load_config().get("local_model", LOCAL_MODEL_DEFAULT)
            print(f"[startup] Pre-warming model '{model}'…")
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
    return {"ok": True}


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


@app.post("/api/images/upload")
async def upload_images(
    files: list[UploadFile] = File(...),
    folder: Optional[str]   = Query(default=""),
):
    """Upload images. If `folder` is provided, place files in that sub-folder."""
    if folder:
        dest_dir = folder_path(folder)
        dest_dir.mkdir(parents=True, exist_ok=True)
    else:
        dest_dir = DATA_DIR

    saved = []
    for file in files:
        if Path(file.filename).suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        dest = dest_dir / Path(file.filename).name
        dest.write_bytes(await file.read())
        saved.append(file.filename)
    return {"saved": saved, "folder": folder}


@app.delete("/api/images/{filename}")
def delete_image(filename: str, folder: Optional[str] = Query(default="")):
    path = (folder_path(folder) / filename) if folder else (DATA_DIR / filename)
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    path.unlink()
    return {"ok": True}


# ── Results download ─────────────────────────────────────────────────────────

@app.get("/api/results/download")
def download_results(folder: Optional[str] = Query(default="")):
    """
    Download all saved .txt result files as a zip.
    If `folder` is specified, only include results tagged for that folder.
    """
    txt_files = list(RESULTS_DIR.glob("*.txt"))
    if folder:
        txt_files = [f for f in txt_files if f"_{folder}_" in f.name or f.name.startswith(f"{folder}_")]

    if not txt_files:
        raise HTTPException(status_code=404, detail="No results available to download.")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in txt_files:
            zf.write(f, f.name)
    buf.seek(0)

    zip_name = f"results{'_' + folder if folder else ''}.zip"
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{zip_name}"'},
    )


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
        "installed":    installed,
        "running":      running,
        "models":       models,
        "loaded":       loaded,
        "pull":         pull,
        "catalog":      LOCAL_MODEL_CATALOG,
        "gpu":          gpu,
        "gpu_allocated": gpu_allocated,
        "gpu_in_use":   gpu_in_use,
        "gpu_ready":    gpu_ready,
    }


@app.post("/api/ollama/start")
def post_ollama_start():
    if not ollama_installed():
        raise HTTPException(status_code=400, detail="Ollama is not installed. Run the setup job.")
    if ollama_running():
        return {"ok": True, "message": "Already running"}
    ollama_start()
    return {"ok": True, "message": "Start requested — allow a few seconds for Ollama to come up"}


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
