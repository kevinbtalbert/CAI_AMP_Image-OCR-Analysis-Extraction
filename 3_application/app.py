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
from fastapi.responses import FileResponse, JSONResponse
from openai import OpenAI
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

try:
    BASE_DIR = Path(__file__).parent
except NameError:
    BASE_DIR = Path(os.getcwd()) / "3_application"

STATIC_DIR = BASE_DIR / "static"
EXAMPLES_DIR = BASE_DIR.parent / "data" / "examples"
DATA_DIR = Path(os.getenv("CDSW_HOME", "/home/cdsw")) / "data"
CONFIG_PATH = Path(os.getenv("CDSW_HOME", "/home/cdsw")) / ".cai_inference_config.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------

LLM_MODEL_LABEL = "Llama 3.3 70B Instruct"
LLM_MODEL_ID = "meta/llama-3.3-70b-instruct"
OCR_MODEL_LABEL = "NeMo Retriever-Parse"
OCR_MODEL_ID = "nemoretriever-parse"

# Local Ollama model catalog — all vision-capable, ordered by RAM footprint
# Recommended for 4 CPU / 16 GB RAM: llava:7b (~4.7 GB Q4)
LOCAL_MODEL_CATALOG = {
    "moondream2  (~1.7 GB · fastest)":           "moondream",
    "Llama 3.2 Vision 11B  (~7.9 GB · default)":  "llama3.2-vision:11b",
    "LLaVA 7B  (~4.7 GB · lighter)":             "llava:7b",
    "LLaVA 13B  (~8.0 GB · higher quality)":     "llava:13b",
}
LOCAL_MODEL_DEFAULT = "llama3.2-vision:11b"
OLLAMA_BASE_URL = "http://localhost:11434"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}

USE_CASES = [
    "Transcribing Typed Text",
    "Transcribing Handwritten Text",
    "Transcribing Forms",
    "Complicated Document QA",
    "Unstructured Information → JSON",
]

USE_CASE_PROMPTS = {
    "Transcribing Typed Text": "Transcribe all typed text from this document exactly as it appears.",
    "Transcribing Handwritten Text": "Transcribe all handwritten text from this document exactly as it appears.",
    "Transcribing Forms": "Transcribe this form exactly, preserving all field labels and their values.",
    "Complicated Document QA": "Answer the following question based on the document: {question}",
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
        # Cloudera AI Inference
        "inference_token": os.getenv("CAI_INFERENCE_TOKEN", ""),
        "ocr_endpoint_url": os.getenv("CAI_OCR_ENDPOINT_URL", ""),
        "llm_endpoint_url": os.getenv("CAI_LLM_ENDPOINT_URL", ""),
        "llm_model_id": LLM_MODEL_ID,
        "ocr_model": OCR_MODEL_ID,
        "max_tokens": 4096,
        # Service mode
        "service_mode": os.getenv("SERVICE_MODE", "cloudera"),  # "cloudera" | "local"
        "local_model": os.getenv("LOCAL_MODEL", LOCAL_MODEL_DEFAULT),
    }


def load_config() -> dict:
    cfg = default_config()
    if CONFIG_PATH.exists():
        try:
            saved = json.loads(CONFIG_PATH.read_text())
            cfg.update(saved)
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


def ollama_start() -> None:
    """Attempt to start the Ollama server in the background."""
    if not ollama_installed():
        return
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def ollama_list_models() -> list[str]:
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


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
                "Ollama binary not found. Run the '2_setup_models' setup job first, "
                "or install Ollama manually from https://ollama.com/download"
            )
        # Let stdout/stderr flow to the app log; avoid buffering a multi-GB download
        result = subprocess.run(
            [ollama_bin, "pull", model],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=7200,   # 2-hour hard limit
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ollama pull exited with code {result.returncode}. "
                + result.stdout[-300:] if result.stdout else ""
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

def get_base64_encoded_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def get_media_type(path: Path) -> str:
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }.get(path.suffix.lower(), "image/jpeg")


def list_images(directory: Path) -> list[dict]:
    if not directory.exists():
        return []
    files = []
    for f in sorted(directory.iterdir()):
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
            stat = f.stat()
            files.append({
                "name": f.name,
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })
    return files


# ---------------------------------------------------------------------------
# OCR — NeMo Retriever-Parse
# ---------------------------------------------------------------------------

def call_nemoretriever_parse(image_path: Path, cfg: dict) -> str:
    base_url = cfg["ocr_endpoint_url"].rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    url = f"{base_url}/parse"
    headers = {"Authorization": f"Bearer {cfg['inference_token']}"}

    with open(image_path, "rb") as f:
        files = {"file": (image_path.name, f, get_media_type(image_path))}
        resp = requests.post(url, headers=headers, files=files, timeout=120)

    resp.raise_for_status()
    result = resp.json()

    blocks = result.get("data", result.get("elements", result.get("content", [])))
    parts = [b.get("content", b.get("text", "")).strip() for b in blocks if b.get("content") or b.get("text")]
    return "\n\n".join(p for p in parts if p) or json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# LLM — OpenAI-compatible
# ---------------------------------------------------------------------------

def call_llm(prompt: str, cfg: dict, system_prompt: str = None) -> str:
    client = OpenAI(base_url=cfg["llm_endpoint_url"], api_key=cfg["inference_token"])
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model=cfg["llm_model_id"],
        messages=messages,
        max_tokens=cfg.get("max_tokens", 4096),
        temperature=0.2,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Local vision LLM — Ollama
# ---------------------------------------------------------------------------

def call_local_vision_llm(image_path: Path, instruction: str, cfg: dict) -> str:
    """Send image + instruction directly to a local Ollama vision model."""
    model = cfg.get("local_model", LOCAL_MODEL_DEFAULT)
    client = OpenAI(
        base_url=f"{OLLAMA_BASE_URL}/v1",
        api_key="ollama",           # Ollama ignores the key but the SDK requires one
    )
    b64 = get_base64_encoded_image(image_path)
    media_type = get_media_type(image_path)
    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}},
                {"type": "text", "text": instruction},
            ],
        }],
        max_tokens=cfg.get("max_tokens", 4096),
        temperature=0.2,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Process pipeline
# ---------------------------------------------------------------------------

def run_pipeline(image_path: Path, instruction: str, cfg: dict) -> dict:
    """Route to local Ollama or Cloudera AI Inference depending on config."""
    if cfg.get("service_mode") == "local":
        # Local: single-step vision LLM — no separate OCR needed
        result = call_local_vision_llm(image_path, instruction, cfg)
        return {"ocr_text": None, "result": result}

    # Cloudera: OCR → LLM pipeline
    ocr_text = call_nemoretriever_parse(image_path, cfg)
    system_prompt = (
        "You are a document analysis assistant. "
        "The user has provided text extracted from an image via OCR. "
        "Use that extracted text to fulfil the user's instruction accurately."
    )
    prompt = (
        f"The following text was extracted from a document image:\n\n"
        f"---\n{ocr_text}\n---\n\n"
        f"Instruction: {instruction}"
    )
    result = call_llm(prompt, cfg, system_prompt=system_prompt)
    return {"ocr_text": ocr_text, "result": result}


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
            cfg = load_config()
            image_path = DATA_DIR / job["filename"]
            if not image_path.exists():
                # Try examples dir
                image_path = EXAMPLES_DIR / job["filename"]

            instruction = USE_CASE_PROMPTS.get(job["use_case"], "Describe the content of this document.")
            if job.get("question") and "{question}" in instruction:
                instruction = instruction.replace("{question}", job["question"])

            output = run_pipeline(image_path, instruction, cfg)

            with _jobs_lock:
                _jobs[job_id].update({
                    "status": "complete",
                    "ocr_text": output["ocr_text"],
                    "result": output["result"],
                    "completed_at": datetime.utcnow().isoformat(),
                })
        except Exception as e:
            with _jobs_lock:
                _jobs[job_id].update({
                    "status": "error",
                    "error": str(e),
                    "completed_at": datetime.utcnow().isoformat(),
                })
        finally:
            _job_q.task_done()


_worker_thread = threading.Thread(target=_worker, daemon=True)
_worker_thread.start()


def create_job(filename: str, use_case: str, question: str = "") -> dict:
    job_id = str(uuid.uuid4())
    job = {
        "id": job_id,
        "filename": filename,
        "use_case": use_case,
        "question": question,
        "status": "queued",
        "ocr_text": None,
        "result": None,
        "error": None,
        "created_at": datetime.utcnow().isoformat(),
        "completed_at": None,
    }
    with _jobs_lock:
        _jobs[job_id] = job
    _job_q.put(job_id)
    return job


# ---------------------------------------------------------------------------
# Connection test helpers
# ---------------------------------------------------------------------------

def test_llm_connection(cfg: dict) -> dict:
    try:
        result = call_llm("Reply with: OK", cfg)
        return {"ok": True, "message": f"Connected — model replied: {result[:60]}"}
    except Exception as e:
        return {"ok": False, "message": str(e)}


def test_ocr_connection(cfg: dict) -> dict:
    base_url = cfg.get("ocr_endpoint_url", "").rstrip("/")
    if not base_url:
        return {"ok": False, "message": "OCR endpoint URL not set."}
    token = cfg.get("inference_token", "")
    if not token:
        return {"ok": False, "message": "Token not set."}
    check = base_url.rstrip("/v1").rstrip("/") + "/v1/metrics"
    try:
        r = requests.get(check, headers={"Authorization": f"Bearer {token}"}, timeout=10)
        if r.status_code < 400:
            return {"ok": True, "message": f"OCR endpoint reachable (HTTP {r.status_code})"}
        return {"ok": False, "message": f"HTTP {r.status_code}"}
    except Exception as e:
        return {"ok": False, "message": str(e)}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ConfigBody(BaseModel):
    inference_token: str = ""
    ocr_endpoint_url: str = ""
    llm_endpoint_url: str = ""
    max_tokens: int = 4096
    service_mode: str = "cloudera"   # "cloudera" | "local"
    local_model: str = LOCAL_MODEL_DEFAULT


class PullModelBody(BaseModel):
    model: str


class ProcessBody(BaseModel):
    filename: str
    use_case: str
    question: Optional[str] = ""
    source: Optional[str] = "uploads"  # "uploads" | "examples"


class BatchBody(BaseModel):
    filenames: list[str]
    use_case: str
    question: Optional[str] = ""
    source: Optional[str] = "uploads"


class TestConnectionBody(BaseModel):
    service: str  # "ocr" | "llm"
    inference_token: str = ""
    ocr_endpoint_url: str = ""
    llm_endpoint_url: str = ""


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Image OCR & Analysis — Cloudera AI Inference")


@app.on_event("startup")
def _startup():
    """If local mode is configured and Ollama isn't running yet, start it."""
    cfg = load_config()
    if cfg.get("service_mode") == "local" and ollama_installed() and not ollama_running():
        print("[startup] Local mode detected — starting Ollama server…")
        ollama_start()

# Serve static assets
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Serve uploaded images
app.mount("/images", StaticFiles(directory=str(DATA_DIR)), name="images")

# Serve example images
app.mount("/examples", StaticFiles(directory=str(EXAMPLES_DIR)), name="examples")


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


# ── Config ─────────────────────────────────────────────────────────────────

@app.get("/api/config")
def get_config():
    cfg = load_config()
    safe = {k: v for k, v in cfg.items() if k != "inference_token"}
    safe["has_token"] = bool(cfg.get("inference_token"))
    safe["models"] = {
        "ocr_label": OCR_MODEL_LABEL,
        "llm_label": LLM_MODEL_LABEL,
        "llm_model_id": LLM_MODEL_ID,
    }
    safe["local_model_catalog"] = LOCAL_MODEL_CATALOG
    safe["use_cases"] = USE_CASES
    return safe


@app.post("/api/config")
def post_config(body: ConfigBody):
    existing = load_config()
    existing.update({
        "inference_token": body.inference_token,
        "ocr_endpoint_url": body.ocr_endpoint_url,
        "llm_endpoint_url": body.llm_endpoint_url,
        "max_tokens": body.max_tokens,
        "llm_model_id": LLM_MODEL_ID,
        "ocr_model": OCR_MODEL_ID,
        "service_mode": body.service_mode,
        "local_model": body.local_model,
    })
    save_config(existing)
    return {"ok": True}


@app.post("/api/test-connection")
def post_test_connection(body: TestConnectionBody):
    cfg = {
        "inference_token": body.inference_token,
        "ocr_endpoint_url": body.ocr_endpoint_url,
        "llm_endpoint_url": body.llm_endpoint_url,
        "llm_model_id": LLM_MODEL_ID,
        "max_tokens": 256,
    }
    if body.service == "llm":
        return test_llm_connection(cfg)
    return test_ocr_connection(cfg)


# ── Images ──────────────────────────────────────────────────────────────────

@app.get("/api/images")
def get_images():
    return {
        "uploads": list_images(DATA_DIR),
        "examples": list_images(EXAMPLES_DIR),
    }


@app.post("/api/images/upload")
async def upload_images(files: list[UploadFile] = File(...)):
    saved = []
    for file in files:
        suffix = Path(file.filename).suffix.lower()
        if suffix not in IMAGE_EXTENSIONS:
            continue
        dest = DATA_DIR / file.filename
        content = await file.read()
        dest.write_bytes(content)
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
    cfg = load_config()
    if not cfg.get("inference_token") or not cfg.get("llm_endpoint_url"):
        raise HTTPException(status_code=400, detail="Service not configured. Please set up your endpoints in Configuration.")

    if body.source == "examples":
        image_path = EXAMPLES_DIR / body.filename
    else:
        image_path = DATA_DIR / body.filename

    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {body.filename}")

    instruction = USE_CASE_PROMPTS.get(body.use_case, "Describe the content of this document.")
    if body.question and "{question}" in instruction:
        instruction = instruction.replace("{question}", body.question)

    try:
        output = run_pipeline(image_path, instruction, cfg)
        return {
            "filename": body.filename,
            "use_case": body.use_case,
            "ocr_text": output["ocr_text"],
            "result": output["result"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Batch jobs ──────────────────────────────────────────────────────────────

@app.post("/api/batch")
def post_batch(body: BatchBody):
    if not body.filenames:
        raise HTTPException(status_code=400, detail="No files specified.")
    created = []
    for filename in body.filenames:
        job = create_job(filename, body.use_case, body.question or "")
        created.append(job)
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
    """Return Ollama install/running state, available models, and pull progress."""
    installed = ollama_installed()
    running = ollama_running() if installed else False
    models = ollama_list_models() if running else []
    with _pull_lock:
        pull = dict(_pull_state)
    return {
        "installed": installed,
        "running": running,
        "models": models,
        "pull": pull,
        "catalog": LOCAL_MODEL_CATALOG,
    }


@app.post("/api/ollama/start")
def post_ollama_start():
    if not ollama_installed():
        raise HTTPException(status_code=400, detail="Ollama is not installed. See https://ollama.com/download")
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
