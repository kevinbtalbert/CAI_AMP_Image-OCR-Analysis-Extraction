import os
import json
import base64
import uuid
import threading
import queue as thread_queue
from datetime import datetime
from pathlib import Path
from typing import Optional

import shutil
import subprocess

import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from openai import OpenAI
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

DATA_DIR.mkdir(parents=True, exist_ok=True)

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
]

USE_CASE_PROMPTS = {
    "Transcribing Typed Text":
        "Transcribe all typed text from this document exactly as it appears.",
    "Transcribing Handwritten Text":
        "Transcribe all handwritten text from this document exactly as it appears.",
    "Transcribing Forms":
        "Transcribe this form exactly, preserving all field labels and their values.",
    "Complicated Document QA":
        "Answer the following question based on the document: {question}",
    "Unstructured Information → JSON": (
        "Convert the content of this document into well-structured JSON. "
        "Identify logical fields and group related information."
    ),
}

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


def _ollama_env() -> dict:
    """
    Build the environment for the Ollama subprocess.

    - Appends common CUDA library paths to LD_LIBRARY_PATH so the CUDA runtime
      is discoverable (CML may not include these by default).
    - Sets OLLAMA_NUM_GPU based on the actual GPU count from nvidia-smi.
      Using nvidia-smi is more reliable than CUDA_VISIBLE_DEVICES because CML
      does not always set that variable even when a GPU is allocated.
    """
    env = os.environ.copy()

    # Ensure CUDA runtime libraries are findable
    cuda_lib_dirs = [
        "/usr/local/cuda/lib64",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/local/lib",
    ]
    existing = env.get("LD_LIBRARY_PATH", "")
    additions = ":".join(d for d in cuda_lib_dirs if d not in existing)
    env["LD_LIBRARY_PATH"] = f"{additions}:{existing}" if existing else additions

    # Tell Ollama explicitly how many GPUs to use
    n_gpu = _count_nvidia_gpus()
    if n_gpu > 0:
        env["OLLAMA_NUM_GPU"] = str(n_gpu)
        print(f"[ollama] nvidia-smi found {n_gpu} GPU(s) — setting OLLAMA_NUM_GPU={n_gpu}")
    else:
        print("[ollama] No GPU detected via nvidia-smi — Ollama will run on CPU")

    # Log CUDA_VISIBLE_DEVICES for debugging (informational only)
    cvd = env.get("CUDA_VISIBLE_DEVICES", "<not set>")
    print(f"[ollama] CUDA_VISIBLE_DEVICES={cvd}")

    return env


def ollama_start() -> None:
    if not ollama_installed():
        return
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        env=_ollama_env(),
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

def get_base64_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def get_media_type(path: Path) -> str:
    return {
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png":  "image/png",
        ".gif":  "image/gif",
        ".webp": "image/webp",
    }.get(path.suffix.lower(), "image/jpeg")


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

def analyze_image(image_path: Path, instruction: str, cfg: dict) -> str:
    """Send an image + instruction to the local Ollama vision model."""
    model  = cfg.get("local_model", LOCAL_MODEL_DEFAULT)
    client = OpenAI(base_url=f"{OLLAMA_BASE_URL}/v1", api_key="ollama")
    b64    = get_base64_image(image_path)
    mime   = get_media_type(image_path)

    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": instruction},
            ],
        }],
        max_tokens=cfg.get("max_tokens", 4096),
        temperature=0.2,
    )
    return response.choices[0].message.content


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
            cfg        = load_config()
            image_path = DATA_DIR / job["filename"]
            if not image_path.exists():
                image_path = EXAMPLES_DIR / job["filename"]

            instruction = USE_CASE_PROMPTS.get(
                job["use_case"], "Describe the content of this document."
            )
            if job.get("question") and "{question}" in instruction:
                instruction = instruction.replace("{question}", job["question"])

            result = analyze_image(image_path, instruction, cfg)

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


def create_job(filename: str, use_case: str, question: str = "") -> dict:
    job_id = str(uuid.uuid4())
    job = {
        "id":           job_id,
        "filename":     filename,
        "use_case":     use_case,
        "question":     question,
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


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Cloudera Image Analysis")


@app.on_event("startup")
def _startup():
    """Auto-start Ollama server if it's installed but not yet running."""
    if ollama_installed() and not ollama_running():
        print("[startup] Starting Ollama server…")
        ollama_start()


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


# ── Images ──────────────────────────────────────────────────────────────────

@app.get("/api/images")
def get_images():
    return {
        "uploads":  list_images(DATA_DIR),
        "examples": list_images(EXAMPLES_DIR),
    }


@app.post("/api/images/upload")
async def upload_images(files: list[UploadFile] = File(...)):
    saved = []
    for file in files:
        if Path(file.filename).suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        dest = DATA_DIR / file.filename
        dest.write_bytes(await file.read())
        saved.append(file.filename)
    return {"saved": saved}


@app.delete("/api/images/{filename}")
def delete_image(filename: str):
    path = DATA_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    path.unlink()
    return {"ok": True}


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

    instruction = USE_CASE_PROMPTS.get(body.use_case, "Describe the content of this document.")
    if body.question and "{question}" in instruction:
        instruction = instruction.replace("{question}", body.question)

    try:
        cfg    = load_config()
        result = analyze_image(image_path, instruction, cfg)
        return {"filename": body.filename, "use_case": body.use_case, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Batch jobs ──────────────────────────────────────────────────────────────

@app.post("/api/batch")
def post_batch(body: BatchBody):
    if not body.filenames:
        raise HTTPException(status_code=400, detail="No files specified.")
    created = [create_job(f, body.use_case, body.question or "") for f in body.filenames]
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
