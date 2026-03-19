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

# Vision-capable Ollama models, ordered by RAM footprint.
# All perform OCR + analysis in a single pass — no separate OCR step needed.
LOCAL_MODEL_CATALOG = {
    "moondream2  (~1.7 GB · fastest)":             "moondream",
    "Llama 3.2 Vision 11B  (~7.9 GB · default)":   "llama3.2-vision:11b",
    "LLaVA 7B  (~4.7 GB · lighter)":               "llava:7b",
    "LLaVA 13B  (~8.0 GB · higher quality)":       "llava:13b",
}
LOCAL_MODEL_DEFAULT = "llama3.2-vision:11b"
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

# System message + user prompt pair per use case.
# system: frames the model's role and hard behavioural constraints.
# user:   the specific task instruction for that image.
# options: Ollama sampling overrides (layered on top of shared defaults).
USE_CASE_CONFIG: dict[str, dict] = {
    "Transcribing Typed Text": {
        "system": (
            "You are a precise OCR engine. Your ONLY job is to reproduce the text "
            "you see in the image, verbatim. Rules you MUST follow:\n"
            "1. Output ONLY the text that physically appears in the image.\n"
            "2. Preserve original line breaks, capitalisation, spacing, and punctuation.\n"
            "3. NEVER paraphrase, summarise, explain, or add any text not in the image.\n"
            "4. NEVER repeat a line or phrase you have already output.\n"
            "5. When you reach the end of the visible text, STOP immediately.\n"
            "6. Do not output any preamble, headers, or closing remarks."
        ),
        "user": "Transcribe every character of printed or typed text visible in this image.",
        "options": {"temperature": 0.05, "repeat_penalty": 1.4, "repeat_last_n": 128,
                    "top_k": 10, "top_p": 0.5},
    },
    "Transcribing Handwritten Text": {
        "system": (
            "You are a precise handwriting OCR engine. Rules you MUST follow:\n"
            "1. Output ONLY the handwritten text visible in the image, verbatim.\n"
            "2. Preserve original line breaks, capitalisation, and punctuation.\n"
            "3. For illegible words, output your best guess surrounded by [brackets].\n"
            "4. NEVER paraphrase, summarise, or add explanations.\n"
            "5. NEVER repeat a line or phrase you have already output.\n"
            "6. When you reach the end of the handwritten text, STOP immediately.\n"
            "7. Do not output any preamble or closing remarks."
        ),
        "user": "Transcribe every word of handwritten text visible in this image.",
        "options": {"temperature": 0.05, "repeat_penalty": 1.4, "repeat_last_n": 128,
                    "top_k": 10, "top_p": 0.5},
    },
    "Transcribing Forms": {
        "system": (
            "You are a form data extractor. Rules you MUST follow:\n"
            "1. Output each field as 'Label: Value', one per line.\n"
            "2. Preserve the exact field labels as printed in the form.\n"
            "3. If a field is blank, output 'Label: (blank)'.\n"
            "4. NEVER add fields that do not appear in the form.\n"
            "5. NEVER repeat a field you have already output.\n"
            "6. Do not add commentary, headings, or preamble."
        ),
        "user": "Extract all fields and their values from this form.",
        "options": {"temperature": 0.05, "repeat_penalty": 1.35, "repeat_last_n": 64,
                    "top_k": 20, "top_p": 0.6},
    },
    "Complicated Document QA": {
        "system": (
            "You are a document analyst. Answer questions using ONLY information "
            "present in the provided document image. If the answer is not in the "
            "document, say so. Be concise and cite the relevant section when possible."
        ),
        "user": "Answer the following question based on this document: {question}",
        "options": {"temperature": 0.2, "repeat_penalty": 1.15, "repeat_last_n": 64,
                    "top_k": 40, "top_p": 0.9},
    },
    "Unstructured Information → JSON": {
        "system": (
            "You are a structured data extractor. Rules you MUST follow:\n"
            "1. Output ONLY valid JSON — no markdown fences, no commentary.\n"
            "2. Identify all logical fields and group related information.\n"
            "3. Use snake_case keys.\n"
            "4. NEVER invent data not present in the image."
        ),
        "user": "Convert all information in this document into structured JSON.",
        "options": {"temperature": 0.1, "repeat_penalty": 1.2, "repeat_last_n": 64,
                    "top_k": 20, "top_p": 0.7},
    },
    "Summarize Image": {
        "system": (
            "You are an expert at analysing and describing images and documents. "
            "Provide clear, informative summaries that capture the purpose, content, "
            "and key takeaways. Write in well-structured prose."
        ),
        "user": "Describe and summarise the content of this image, including its purpose and key information.",
        "options": {"temperature": 0.3, "repeat_penalty": 1.15, "repeat_last_n": 64,
                    "top_k": 40, "top_p": 0.9},
    },
}

# Flat prompt map kept for backward-compat (batch worker uses this for the instruction string)
USE_CASE_PROMPTS = {k: v["user"] for k, v in USE_CASE_CONFIG.items()}

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

# llama3.2-vision processes images as 560×560 tiles (up to 4 tiles + 1 thumbnail).
# Sending anything larger than 1120px on the longest side gives no quality gain
# and significantly increases encoding time and model prompt tokens.
_MAX_IMAGE_PX = 1120
_JPEG_QUALITY = 88


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
    """Return merged Ollama sampling options for a given use case."""
    base = {
        "num_predict": cfg.get("max_tokens", 4096),
    }
    uc_opts = USE_CASE_CONFIG.get(use_case, {}).get("options", {})
    return {**base, **uc_opts}


def _build_ollama_messages(image_path: Path, use_case: str, cfg: dict | None = None) -> tuple[list[dict], list[dict]]:
    """
    Return (openai_messages, native_messages) for a use case.

    openai_messages  — for /v1/chat/completions (image_url content blocks)
    native_messages  — for /api/chat (images list on message object)
    """
    uc      = USE_CASE_CONFIG.get(use_case, {})
    system  = uc.get("system", "")
    user    = uc.get("user", "Describe this image.")
    if cfg and cfg.get("_qa_question") and "{question}" in user:
        user = user.replace("{question}", cfg["_qa_question"])
    b64, mime = prepare_image(image_path)

    openai_msgs = []
    native_msgs = []
    if system:
        openai_msgs.append({"role": "system", "content": system})
        native_msgs.append({"role": "system", "content": system})

    openai_msgs.append({
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            {"type": "text",      "text": user},
        ],
    })
    native_msgs.append({
        "role":    "user",
        "content": user,
        "images":  [b64],
    })
    return openai_msgs, native_msgs


# Minimum phrase length and occurrence count to declare a repetition loop.
_REP_PHRASE_LEN = 40
_REP_MAX_COUNT  = 3


def _detect_repetition(text: str) -> bool:
    """
    Return True if *text* contains a repeated phrase that indicates
    the model is stuck in a generation loop.

    Strategy: check whether any substring of length >= _REP_PHRASE_LEN
    appears _REP_MAX_COUNT or more times consecutively in the tail of the text.
    We only scan the last 600 characters to keep this fast.
    """
    tail = text[-600:]
    n    = len(tail)
    for plen in (60, 45, _REP_PHRASE_LEN):
        if n < plen * _REP_MAX_COUNT:
            continue
        phrase = tail[-(plen * _REP_MAX_COUNT): -(plen * (_REP_MAX_COUNT - 1))]
        if phrase and tail.count(phrase) >= _REP_MAX_COUNT:
            return True
    return False


def analyze_image(image_path: Path, use_case: str, cfg: dict) -> str:
    """
    Send an image to the local Ollama vision model (blocking, for batch jobs).
    Uses the OpenAI-compatible /v1 endpoint with per-use-case system messages
    and sampling options.
    """
    model = cfg.get("local_model", LOCAL_MODEL_DEFAULT)
    client = OpenAI(base_url=f"{OLLAMA_BASE_URL}/v1", api_key="ollama")
    openai_msgs, _ = _build_ollama_messages(image_path, use_case, cfg)
    opts = _use_case_options(use_case, cfg)

    response = client.chat.completions.create(
        model=model,
        messages=openai_msgs,
        max_tokens=opts.get("num_predict", 4096),
        temperature=opts.get("temperature", 0.1),
        extra_body={k: v for k, v in opts.items()
                    if k not in ("num_predict", "temperature")},
    )
    return response.choices[0].message.content


def stream_analyze_image(image_path: Path, use_case: str, cfg: dict):
    """
    Yield SSE lines ('data: {...}\\n\\n') streaming tokens from Ollama.

    Uses Ollama's native /api/chat with stream=true.  Includes:
    - Per-use-case system messages and sampling options.
    - A repetition-loop guard: if the accumulated output shows the model
      repeating itself, generation is terminated and a warning is yielded.
    """
    model = cfg.get("local_model", LOCAL_MODEL_DEFAULT)
    _, native_msgs = _build_ollama_messages(image_path, use_case, cfg)
    opts = _use_case_options(use_case, cfg)

    payload = {
        "model":    model,
        "messages": native_msgs,
        "stream":   True,
        "options":  opts,
    }

    accumulated = ""
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
                    accumulated += token
                    yield f"data: {json.dumps({'token': token})}\n\n"

                    # Repetition guard — stop early and warn
                    if _detect_repetition(accumulated):
                        yield (f"data: {json.dumps({'token': '\n\n⚠ [Generation stopped: repetition loop detected]'})}\n\n")
                        yield f"data: {json.dumps({'done': True})}\n\n"
                        return

                if chunk.get("done"):
                    yield f"data: {json.dumps({'done': True})}\n\n"
                    return
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
