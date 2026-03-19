"""
Setup script — installs Ollama and pulls the default vision model.

Run order:
  1_session-install-dependencies  →  pip install requirements
  2_setup_models/setup_models.py  →  install Ollama + pull model  (this script)
  3_application/start-app.py     →  launch the FastAPI app

Ollama is installed to ~/.local/bin/ollama so *no superuser / root access is
required*. The binary is downloaded directly from GitHub Releases.

Environment variables (optional):
  LOCAL_MODEL   Override the model to pull  (default: llama3.2-vision:11b)
  SKIP_OLLAMA   Set to '1' to skip this entire script
"""

import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import urllib.error

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "llama3.2-vision:11b"
MODEL = os.environ.get("LOCAL_MODEL", DEFAULT_MODEL)
SKIP  = os.environ.get("SKIP_OLLAMA", "0").strip() == "1"

# Install Ollama here — user-writable, no root needed.
OLLAMA_INSTALL_DIR = os.path.expanduser("~/.local/bin")
OLLAMA_BIN         = os.path.join(OLLAMA_INSTALL_DIR, "ollama")

# GitHub Releases base URL (redirects to latest)
_GH_BASE = "https://github.com/ollama/ollama/releases/latest/download"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def banner(msg: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}\n")


def _ensure_on_path(directory: str) -> None:
    """Prepend *directory* to PATH for this process and child processes."""
    current = os.environ.get("PATH", "")
    dirs = current.split(os.pathsep)
    if directory not in dirs:
        os.environ["PATH"] = directory + os.pathsep + current
        print(f"  Added {directory} to PATH")


def _download(url: str, dest: str, *, label: str = "") -> bool:
    """Stream-download *url* to *dest*, return True on success."""
    try:
        print(f"  Downloading {label or url}")
        urllib.request.urlretrieve(url, dest)
        return True
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code}: {url}")
        return False
    except Exception as e:
        print(f"  Download error: {e}")
        return False


def _extract_ollama_from_tgz(tgz_path: str, dest_path: str) -> bool:
    """
    Extract the 'ollama' binary from a .tgz archive.
    Handles both 'ollama' and 'bin/ollama' archive layouts.
    """
    try:
        with tarfile.open(tgz_path, "r:gz") as tar:
            member = None
            for m in tar.getmembers():
                if m.isfile() and (m.name == "ollama" or m.name.endswith("/ollama")):
                    member = m
                    break
            if member is None:
                print("  Warning: 'ollama' binary not found in tarball")
                return False
            member.name = "ollama"  # flatten any sub-directory
            tar.extract(member, os.path.dirname(dest_path))
        return True
    except Exception as e:
        print(f"  tgz extraction failed: {e}")
        return False


# ---------------------------------------------------------------------------
# 1. Install Ollama (no root)
# ---------------------------------------------------------------------------

def install_ollama() -> bool:
    """
    Install Ollama to ~/.local/bin without requiring root or sudo.
    Downloads the pre-built binary directly from GitHub Releases.
    """
    # Check if already available on PATH
    if shutil.which("ollama"):
        print("✓ Ollama already on PATH.")
        return True

    # Check our install location directly
    if os.path.isfile(OLLAMA_BIN) and os.access(OLLAMA_BIN, os.X_OK):
        print(f"✓ Ollama binary found at {OLLAMA_BIN}.")
        _ensure_on_path(OLLAMA_INSTALL_DIR)
        return True

    system = platform.system().lower()
    machine = platform.machine().lower()

    banner(f"Installing Ollama (no root) — {system}/{machine}")
    os.makedirs(OLLAMA_INSTALL_DIR, exist_ok=True)

    # ── Linux ─────────────────────────────────────────────────────────────
    if system == "linux":
        arch = "arm64" if machine in ("aarch64", "arm64") else "amd64"

        # Strategy 1: download the tgz release
        tgz_url = f"{_GH_BASE}/ollama-linux-{arch}.tgz"
        with tempfile.NamedTemporaryFile(suffix=".tgz", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            if _download(tgz_url, tmp_path, label=f"ollama-linux-{arch}.tgz"):
                if _extract_ollama_from_tgz(tmp_path, OLLAMA_BIN):
                    os.chmod(OLLAMA_BIN, 0o755)
                    _ensure_on_path(OLLAMA_INSTALL_DIR)
                    print(f"✓ Ollama installed to {OLLAMA_BIN} (from tgz).")
                    return True
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        # Strategy 2: fall back to direct binary (older release format)
        bin_url = f"{_GH_BASE}/ollama-linux-{arch}"
        if _download(bin_url, OLLAMA_BIN, label=f"ollama-linux-{arch} binary"):
            os.chmod(OLLAMA_BIN, 0o755)
            _ensure_on_path(OLLAMA_INSTALL_DIR)
            print(f"✓ Ollama installed to {OLLAMA_BIN} (direct binary).")
            return True

        print("ERROR: Could not download Ollama binary. Check network access.")
        return False

    # ── macOS ─────────────────────────────────────────────────────────────
    elif system == "darwin":
        # Try Homebrew first (doesn't need root if Homebrew itself was installed in user space)
        if shutil.which("brew"):
            result = subprocess.run(["brew", "install", "ollama"],
                                    capture_output=False)
            if result.returncode == 0:
                print("✓ Ollama installed via Homebrew.")
                return True
            print("  Homebrew install failed, falling back to binary download.")

        arch = "arm64" if machine in ("arm64", "aarch64") else "amd64"

        # Try tgz
        tgz_url = f"{_GH_BASE}/ollama-darwin-{arch}.tgz"
        with tempfile.NamedTemporaryFile(suffix=".tgz", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            if _download(tgz_url, tmp_path, label=f"ollama-darwin-{arch}.tgz"):
                if _extract_ollama_from_tgz(tmp_path, OLLAMA_BIN):
                    os.chmod(OLLAMA_BIN, 0o755)
                    _ensure_on_path(OLLAMA_INSTALL_DIR)
                    print(f"✓ Ollama installed to {OLLAMA_BIN}.")
                    return True
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        # Direct binary
        bin_url = f"{_GH_BASE}/ollama-darwin"
        if _download(bin_url, OLLAMA_BIN, label="ollama-darwin binary"):
            os.chmod(OLLAMA_BIN, 0o755)
            _ensure_on_path(OLLAMA_INSTALL_DIR)
            print(f"✓ Ollama installed to {OLLAMA_BIN}.")
            return True

        print("ERROR: Could not install Ollama on macOS. "
              "Please install manually: https://ollama.com/download")
        return False

    else:
        print(f"ERROR: Unsupported OS '{system}'. "
              "Install Ollama manually: https://ollama.com/download")
        return False


# ---------------------------------------------------------------------------
# 2. Start Ollama server
# ---------------------------------------------------------------------------

def ensure_ollama_running() -> bool:
    """Start the Ollama server if it isn't already responding."""
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
    subprocess.Popen(
        [ollama_bin, "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    for i in range(15):
        time.sleep(2)
        if is_up():
            print(f"✓ Ollama server ready (after {(i+1)*2}s).")
            return True
        print(f"  Waiting for Ollama to start… ({(i+1)*2}s)")

    print("ERROR: Ollama server did not respond within 30 seconds.")
    return False


# ---------------------------------------------------------------------------
# 3. Pull model
# ---------------------------------------------------------------------------

def pull_model(model: str) -> bool:
    """Pull a model via `ollama pull`. Shows live progress."""
    import json

    # Check if already pulled
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

def main():
    banner("Cloudera Image Analysis — Local Model Setup")
    print(f"  Target model : {MODEL}")
    print(f"  Platform     : {platform.system()} {platform.machine()}")
    print(f"  Install dir  : {OLLAMA_INSTALL_DIR}")

    if SKIP:
        print("\nSKIP_OLLAMA=1 — skipping setup.")
        sys.exit(0)

    # Step 1: install
    if not install_ollama():
        print("\nWARNING: Ollama installation failed. Local mode will not be available.")
        print("  Set SERVICE_MODE=cloudera to use Cloudera AI Inference instead.")
        sys.exit(1)

    # Step 2: start server
    if not ensure_ollama_running():
        print("\nWARNING: Could not start Ollama server. Try manually: `ollama serve`")
        sys.exit(1)

    # Step 3: pull model
    if not pull_model(MODEL):
        print(f"\nWARNING: Could not pull model '{MODEL}'.")
        sys.exit(1)

    banner("Setup complete — ready to start the application")
    print(f"  Ollama is running with model: {MODEL}\n")


if __name__ == "__main__":
    main()
