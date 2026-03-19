"""
Setup script — installs Ollama and pulls the default vision model.

Run order:
  1_session-install-dependencies  →  pip install requirements (includes zstandard)
  2_setup_models/setup_models.py  →  install Ollama + pull model  (this script)
  3_application/start-app.py     →  launch the FastAPI app

Ollama is installed to ~/.local/bin/ollama so *no superuser / root access is
required*. The binary is downloaded directly from the Ollama GitHub Releases.

The script queries the GitHub API for the *latest* release tag at runtime, so
it always installs the newest version without needing code changes when Ollama
releases a new version.

Environment variables (optional):
  LOCAL_MODEL   Override the model to pull  (default: llama3.2-vision:11b)
"""

import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "llama3.2-vision:11b"
MODEL         = os.environ.get("LOCAL_MODEL", DEFAULT_MODEL)

# User-writable install target — no root required.
# The Ollama archive has the layout:
#   bin/ollama          → the executable
#   lib/ollama/         → CUDA runners (essential for GPU support)
# We extract everything to ~/.local/ so the relative paths work:
#   ~/.local/bin/ollama
#   ~/.local/lib/ollama/
OLLAMA_LOCAL_DIR   = os.path.expanduser("~/.local")
OLLAMA_INSTALL_DIR = os.path.join(OLLAMA_LOCAL_DIR, "bin")
OLLAMA_LIB_DIR     = os.path.join(OLLAMA_LOCAL_DIR, "lib", "ollama")
OLLAMA_BIN         = os.path.join(OLLAMA_INSTALL_DIR, "ollama")

# GitHub API endpoint — returns latest release metadata including the tag
_OLLAMA_API        = "https://api.github.com/repos/ollama/ollama/releases/latest"
_OLLAMA_DL_BASE    = "https://github.com/ollama/ollama/releases/download"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def banner(msg: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}\n")


def _ensure_on_path(directory: str) -> None:
    """Prepend *directory* to PATH for this process (and children)."""
    current = os.environ.get("PATH", "")
    if directory not in current.split(os.pathsep):
        os.environ["PATH"] = directory + os.pathsep + current
        print(f"  Added {directory} to PATH")


def _get_latest_ollama_tag() -> str | None:
    """
    Query the GitHub Releases API for the latest Ollama tag (e.g. 'v0.18.2').
    Returns the tag string, or None on failure.
    """
    try:
        req = urllib.request.Request(
            _OLLAMA_API,
            headers={
                "Accept":     "application/vnd.github.v3+json",
                "User-Agent": "CAI-AMP-Image-Analysis/1.0",
            },
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            tag = json.loads(resp.read())["tag_name"]
            print(f"  Latest Ollama release: {tag}")
            return tag
    except Exception as e:
        print(f"  GitHub API unavailable: {e}")
        return None


def _download(url: str, dest: str, *, label: str = "") -> bool:
    """
    Download *url* to *dest* with periodic progress output.
    Returns True on success.
    """
    last: list[float] = [0.0]

    def _progress(count: int, block: int, total: int) -> None:
        now = time.monotonic()
        if now - last[0] < 10:   # print at most every 10 s
            return
        last[0] = now
        if total > 0:
            pct = min(100, count * block * 100 / total)
            done_mb  = count * block / 1_048_576
            total_mb = total / 1_048_576
            print(f"  {pct:5.1f}%  {done_mb:6.0f} / {total_mb:.0f} MB", flush=True)
        else:
            print(f"  Downloaded {count * block / 1_048_576:.0f} MB…", flush=True)

    print(f"  Downloading {label or url}")
    try:
        urllib.request.urlretrieve(url, dest, _progress)
        print("  Download complete.", flush=True)
        return True
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code}: {url}")
        return False
    except Exception as e:
        print(f"  Download error: {e}")
        return False


def _extract_zst_tar(archive: str, prefix: str) -> bool:
    """
    Extract the full Ollama .tar.zst archive into *prefix* (e.g. ~/.local/).

    The archive layout is:
        bin/ollama          → <prefix>/bin/ollama   (executable)
        lib/ollama/         → <prefix>/lib/ollama/  (CUDA runners — GPU required!)

    Extracting ONLY the binary is not enough: without lib/ollama/ Ollama
    silently falls back to CPU mode even when a GPU is present.

    Primary method: Python zstandard + tarfile (pure-Python, no system deps).
    Fallback: system 'tar' command (works if zstd tool is installed).
    """
    os.makedirs(prefix, exist_ok=True)

    # ── Primary: zstandard Python package ─────────────────────────────────
    try:
        import zstandard  # installed by requirements.txt

        print(f"  Extracting full archive via zstandard → {prefix}")
        extracted = []
        with open(archive, "rb") as fh:
            dctx = zstandard.ZstdDecompressor()
            with dctx.stream_reader(fh) as reader:
                with tarfile.open(fileobj=reader, mode="r|") as tar:
                    for member in tar:
                        # Skip unsafe paths (absolute paths, .. traversal)
                        if member.name.startswith("/") or ".." in member.name:
                            continue
                        tar.extract(member, prefix)
                        extracted.append(member.name)

        if not extracted:
            print("  Archive was empty — extraction failed.")
            return False

        bin_path = os.path.join(prefix, "bin", "ollama")
        lib_path = os.path.join(prefix, "lib", "ollama")
        print(f"  Extracted {len(extracted)} entries.")
        print(f"  Binary : {bin_path}  (exists={os.path.isfile(bin_path)})")
        print(f"  Lib dir: {lib_path}  (exists={os.path.isdir(lib_path)})")
        return os.path.isfile(bin_path)

    except ImportError:
        print("  zstandard package not found — trying system tar…")
    except Exception as e:
        print(f"  zstandard extraction failed: {e} — trying system tar…")

    # ── Fallback: system tar ───────────────────────────────────────────────
    print(f"  Extracting via system tar → {prefix}")
    result = subprocess.run(
        ["tar", "-xf", archive, "-C", prefix],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  tar failed: {result.stderr.strip()}")
        return False

    bin_path = os.path.join(prefix, "bin", "ollama")
    print(f"  Binary : {bin_path}  (exists={os.path.isfile(bin_path)})")
    return os.path.isfile(bin_path)


# ---------------------------------------------------------------------------
# 1. Install Ollama (no root)
# ---------------------------------------------------------------------------

def install_ollama() -> bool:
    """
    Install Ollama to ~/.local/ without root/sudo.

    The full archive is extracted to ~/.local/ preserving paths:
      ~/.local/bin/ollama        — executable
      ~/.local/lib/ollama/       — CUDA runners (required for GPU)

    Steps:
      1. Query GitHub Releases API for the latest tag.
      2. Download ollama-linux-{arch}.tar.zst (current release format).
      3. Extract the FULL archive (not just the binary) to ~/.local/.
    """
    # Already installed with CUDA runners? Skip download.
    lib_ok = os.path.isdir(OLLAMA_LIB_DIR) and bool(os.listdir(OLLAMA_LIB_DIR))
    bin_ok = bool(shutil.which("ollama")) or (
        os.path.isfile(OLLAMA_BIN) and os.access(OLLAMA_BIN, os.X_OK)
    )
    if bin_ok and lib_ok:
        print("✓ Ollama binary and CUDA runners already present.")
        _ensure_on_path(OLLAMA_INSTALL_DIR)
        _verify_lib_dir()
        return True
    if bin_ok and not lib_ok:
        print("⚠ Ollama binary found but lib/ollama/ is missing — reinstalling full archive.")
        print("  (Previous install may have extracted only the binary; GPU requires lib/ollama/)")
    elif not bin_ok:
        print("  Ollama not found — installing.")

    system  = platform.system().lower()
    machine = platform.machine().lower()

    banner(f"Installing Ollama (no root) — {system}/{machine}")

    # ── Linux ─────────────────────────────────────────────────────────────
    if system == "linux":
        arch = "arm64" if machine in ("aarch64", "arm64") else "amd64"

        tag = _get_latest_ollama_tag()
        if tag:
            url = f"{_OLLAMA_DL_BASE}/{tag}/ollama-linux-{arch}.tar.zst"
        else:
            url = (f"https://github.com/ollama/ollama/releases/latest"
                   f"/download/ollama-linux-{arch}.tar.zst")

        with tempfile.NamedTemporaryFile(suffix=".tar.zst", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            label = f"ollama-linux-{arch}.tar.zst ({tag or 'latest'})"
            if not _download(url, tmp_path, label=label):
                print("ERROR: Could not download Ollama archive. Check network access.")
                return False

            if not _extract_zst_tar(tmp_path, OLLAMA_LOCAL_DIR):
                return False

            os.chmod(OLLAMA_BIN, 0o755)
            _ensure_on_path(OLLAMA_INSTALL_DIR)
            _verify_lib_dir()
            print(f"✓ Ollama {tag or ''} installed → {OLLAMA_BIN}")
            return True

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    # ── macOS ─────────────────────────────────────────────────────────────
    elif system == "darwin":
        if shutil.which("brew"):
            result = subprocess.run(["brew", "install", "ollama"], capture_output=False)
            if result.returncode == 0:
                print("✓ Ollama installed via Homebrew.")
                return True
            print("  Homebrew install failed, falling back to binary download.")

        arch = "arm64" if machine in ("arm64", "aarch64") else "amd64"
        tag  = _get_latest_ollama_tag()
        if tag:
            url = f"{_OLLAMA_DL_BASE}/{tag}/ollama-darwin-{arch}.tar.zst"
        else:
            url = (f"https://github.com/ollama/ollama/releases/latest"
                   f"/download/ollama-darwin-{arch}.tar.zst")

        with tempfile.NamedTemporaryFile(suffix=".tar.zst", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            if not _download(url, tmp_path, label=f"ollama-darwin-{arch}.tar.zst"):
                print("ERROR: Could not download Ollama for macOS.")
                print("  Please install manually: https://ollama.com/download")
                return False
            if not _extract_zst_tar(tmp_path, OLLAMA_LOCAL_DIR):
                return False
            os.chmod(OLLAMA_BIN, 0o755)
            _ensure_on_path(OLLAMA_INSTALL_DIR)
            print(f"✓ Ollama {tag or ''} installed → {OLLAMA_BIN}")
            return True
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    else:
        print(f"ERROR: Unsupported OS '{system}'. "
              "Install Ollama manually: https://ollama.com/download")
        return False


def _verify_lib_dir() -> None:
    """Warn if the CUDA runner library directory is missing."""
    if os.path.isdir(OLLAMA_LIB_DIR):
        runners = os.listdir(OLLAMA_LIB_DIR)
        cuda_runners = [r for r in runners if "cuda" in r.lower()]
        print(f"  lib/ollama entries : {len(runners)}")
        print(f"  CUDA runners found : {cuda_runners or 'none'}")
        if not cuda_runners:
            print("  WARNING: No CUDA runners found — GPU will not be available.")
    else:
        print(f"  WARNING: {OLLAMA_LIB_DIR} missing — GPU will not be available.")
        print("  Re-run this setup script to reinstall the full Ollama archive.")


# ---------------------------------------------------------------------------
# 2. Start Ollama server
# ---------------------------------------------------------------------------

def _build_ollama_env() -> dict:
    """
    Build environment for the Ollama subprocess with GPU support.

    - Adds ~/.local/lib/ollama to LD_LIBRARY_PATH so Ollama finds its
      bundled CUDA runners even if they weren't on the system path.
    - Sets OLLAMA_NUM_GPU based on nvidia-smi GPU count.
    - Sets OLLAMA_RUNNERS_DIR explicitly so Ollama doesn't search for runners
      in the wrong place when installed to a non-standard prefix.
    """
    env = os.environ.copy()

    # Bundled CUDA runners live in ~/.local/lib/ollama/
    lib_paths = [
        OLLAMA_LIB_DIR,
        os.path.join(OLLAMA_LOCAL_DIR, "lib"),
        "/usr/local/cuda/lib64",
        "/usr/lib/x86_64-linux-gnu",
    ]
    existing = env.get("LD_LIBRARY_PATH", "")
    new_paths = ":".join(p for p in lib_paths if p not in existing)
    env["LD_LIBRARY_PATH"] = f"{new_paths}:{existing}" if existing else new_paths

    # Point Ollama at its runners directory explicitly
    env.setdefault("OLLAMA_RUNNERS_DIR", OLLAMA_LIB_DIR)

    # Count GPUs via nvidia-smi and tell Ollama how many to use
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            n_gpu = len([l for l in r.stdout.strip().splitlines() if l.strip()])
            if n_gpu > 0:
                env["OLLAMA_NUM_GPU"] = str(n_gpu)
                print(f"  nvidia-smi: {n_gpu} GPU(s) — setting OLLAMA_NUM_GPU={n_gpu}")
            else:
                print("  nvidia-smi: no GPUs found — CPU mode")
        else:
            print("  nvidia-smi not available — CPU mode")
    except Exception as e:
        print(f"  nvidia-smi check failed: {e} — CPU mode")

    cvd = env.get("CUDA_VISIBLE_DEVICES", "<not set>")
    print(f"  CUDA_VISIBLE_DEVICES={cvd}")

    return env


def ensure_ollama_running() -> bool:
    """Start the Ollama server (with GPU env) if it isn't already responding."""
    def is_up() -> bool:
        try:
            urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3)
            return True
        except Exception:
            return False

    if is_up():
        print("✓ Ollama server already running.")
        return True

    ollama_bin = shutil.which("ollama") or OLLAMA_BIN
    print(f"  Starting Ollama server ({ollama_bin})…")
    env = _build_ollama_env()
    subprocess.Popen(
        [ollama_bin, "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        env=env,
    )

    for i in range(15):
        time.sleep(2)
        if is_up():
            print(f"✓ Ollama server ready (after {(i+1)*2}s).")
            return True
        print(f"  Waiting… ({(i+1)*2}s)")

    print("ERROR: Ollama server did not respond within 30 seconds.")
    return False


# ---------------------------------------------------------------------------
# 3. Pull model
# ---------------------------------------------------------------------------

def pull_model(model: str) -> bool:
    """Pull a model via `ollama pull`. Streams progress to stdout."""
    # Check if already available
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        pulled = [m["name"] for m in data.get("models", [])]
        if any(m == model or m.startswith(model.split(":")[0]) for m in pulled):
            print(f"✓ Model '{model}' already pulled.")
            return True
    except Exception:
        pass

    banner(f"Pulling model: {model}")
    print("  This may take several minutes for large models.")
    print(f"  Storage: ~/.ollama/models/\n")

    ollama_bin = shutil.which("ollama") or OLLAMA_BIN
    result = subprocess.run(
        [ollama_bin, "pull", model],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    if result.returncode != 0:
        print(f"\nERROR: `ollama pull {model}` failed (exit {result.returncode})")
        return False

    print(f"\n✓ Model '{model}' pulled successfully.")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    banner("Cloudera Image Analysis — Local Model Setup")
    print(f"  Target model : {MODEL}")
    print(f"  Platform     : {platform.system()} {platform.machine()}")
    print(f"  Install dir  : {OLLAMA_INSTALL_DIR}")

    if not install_ollama():
        print("\nWARNING: Ollama installation failed. Local mode will not be available.")
        print("  Set SERVICE_MODE=cloudera to use Cloudera AI Inference instead.")
        sys.exit(1)

    if not ensure_ollama_running():
        print("\nWARNING: Could not start Ollama server. Try manually: `ollama serve`")
        sys.exit(1)

    if not pull_model(MODEL):
        print(f"\nWARNING: Could not pull model '{MODEL}'.")
        sys.exit(1)

    banner("Setup complete — ready to start the application")
    print(f"  Ollama is running with model: {MODEL}\n")


if __name__ == "__main__":
    main()
