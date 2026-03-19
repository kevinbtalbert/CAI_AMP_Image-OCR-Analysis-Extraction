"""
Setup script — installs Ollama and pulls the default vision model.

Run order:
  1_session-install-dependencies  →  pip install requirements
  2_setup_models                  →  install Ollama + pull model  (this script)
  3_application (start-app.py)   →  launch the FastAPI app

Environment variables (optional):
  LOCAL_MODEL   Override the model to pull  (default: llama3.2-vision:11b)
  SKIP_OLLAMA   Set to '1' to skip this entire script
"""

import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "llama3.2-vision:11b"
OLLAMA_LINUX_INSTALL_URL = "https://ollama.com/install.sh"
MODEL = os.environ.get("LOCAL_MODEL", DEFAULT_MODEL)
SKIP = os.environ.get("SKIP_OLLAMA", "0").strip() == "1"


def banner(msg: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}\n")


def run(cmd: list, **kwargs) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, **kwargs)


# ---------------------------------------------------------------------------
# 1. Install Ollama
# ---------------------------------------------------------------------------

def install_ollama() -> bool:
    if shutil.which("ollama"):
        print("✓ Ollama already installed.")
        return True

    system = platform.system().lower()
    banner(f"Installing Ollama ({system})")

    if system == "linux":
        # Official one-liner install script
        print("  Downloading and running Ollama install script…")
        result = subprocess.run(
            "curl -fsSL https://ollama.com/install.sh | sh",
            shell=True,
        )
        if result.returncode != 0:
            print("  curl not available, trying wget…")
            result = subprocess.run(
                "wget -qO- https://ollama.com/install.sh | sh",
                shell=True,
            )
        if result.returncode != 0:
            print("ERROR: Could not install Ollama. Check network access.")
            return False
        print("✓ Ollama installed.")
        return True

    elif system == "darwin":
        # Try Homebrew first, then direct download
        if shutil.which("brew"):
            result = run(["brew", "install", "ollama"])
            if result.returncode == 0:
                print("✓ Ollama installed via Homebrew.")
                return True
        # Fallback: download the macOS binary directly
        print("  Attempting direct binary download for macOS…")
        try:
            dest = "/usr/local/bin/ollama"
            url = "https://github.com/ollama/ollama/releases/latest/download/ollama-darwin"
            urllib.request.urlretrieve(url, dest)
            os.chmod(dest, 0o755)
            print("✓ Ollama binary installed to /usr/local/bin/ollama.")
            return True
        except Exception as e:
            print(f"ERROR: Could not download Ollama binary: {e}")
            print("  Please install Ollama manually: https://ollama.com/download")
            return False

    else:
        print(f"ERROR: Unsupported OS '{system}'. Install Ollama manually: https://ollama.com/download")
        return False


# ---------------------------------------------------------------------------
# 2. Start Ollama server
# ---------------------------------------------------------------------------

def ensure_ollama_running() -> bool:
    """Start the Ollama server if it isn't already responding."""
    import urllib.error

    def is_up() -> bool:
        try:
            urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3)
            return True
        except Exception:
            return False

    if is_up():
        print("✓ Ollama server already running.")
        return True

    print("  Starting Ollama server…")
    subprocess.Popen(
        ["ollama", "serve"],
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
    import urllib.error

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
    print(f"  This may take several minutes for large models.")
    print(f"  Model storage: ~/.ollama/models/\n")

    # Run ollama pull — streams progress to stdout so CML logs show it
    result = run(
        ["ollama", "pull", model],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    if result.returncode != 0:
        print(f"\nERROR: `ollama pull {model}` failed with exit code {result.returncode}")
        return False

    print(f"\n✓ Model '{model}' pulled successfully.")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    banner("Cloudera AI Inference — Local Model Setup")
    print(f"  Target model : {MODEL}")
    print(f"  Platform     : {platform.system()} {platform.machine()}")

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
