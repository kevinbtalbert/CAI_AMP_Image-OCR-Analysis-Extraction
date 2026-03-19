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

# User-writable install target — no root required
OLLAMA_INSTALL_DIR = os.path.expanduser("~/.local/bin")
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


def _extract_zst_tar(archive: str, dest_bin: str) -> bool:
    """
    Extract the 'ollama' binary from a .tar.zst archive.

    Primary method: Python zstandard + tarfile (pure-Python, no system deps).
    Fallback: system 'tar' command (works if zstd tool is installed).
    """
    # ── Primary: zstandard Python package ─────────────────────────────────
    try:
        import zstandard  # installed by requirements.txt

        print("  Extracting via zstandard…")
        with open(archive, "rb") as fh:
            dctx = zstandard.ZstdDecompressor()
            with dctx.stream_reader(fh) as reader:
                with tarfile.open(fileobj=reader, mode="r|") as tar:
                    for member in tar:
                        name = member.name
                        # Match 'ollama' or 'bin/ollama' inside the archive
                        if member.isfile() and (name == "ollama" or name.endswith("/ollama")):
                            # Extract just this member, flattening any sub-dir
                            member.name = "ollama"
                            tar.extract(member, os.path.dirname(dest_bin))
                            print(f"  Extracted '{name}' → {dest_bin}")
                            return True
        print("  'ollama' binary not found inside archive.")
        return False

    except ImportError:
        print("  zstandard package not found — trying system tar…")
    except Exception as e:
        print(f"  zstandard extraction failed: {e} — trying system tar…")

    # ── Fallback: system tar ───────────────────────────────────────────────
    print("  Extracting via system tar…")
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            ["tar", "-xf", archive, "-C", tmpdir],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"  tar failed: {result.stderr.strip()}")
            return False

        # Walk the extracted tree to find the binary
        for root, _, files in os.walk(tmpdir):
            if "ollama" in files:
                found = os.path.join(root, "ollama")
                shutil.copy2(found, dest_bin)
                print(f"  Copied {found} → {dest_bin}")
                return True

    print("  'ollama' binary not found after system tar extraction.")
    return False


# ---------------------------------------------------------------------------
# 1. Install Ollama (no root)
# ---------------------------------------------------------------------------

def install_ollama() -> bool:
    """
    Install Ollama to ~/.local/bin without root/sudo.

    Steps:
      1. Query GitHub Releases API for the latest tag.
      2. Download ollama-linux-{arch}.tar.zst (current release format).
      3. Extract the binary using Python zstandard (or system tar as fallback).
    """
    # Already installed?
    if shutil.which("ollama"):
        print("✓ Ollama already on PATH.")
        return True
    if os.path.isfile(OLLAMA_BIN) and os.access(OLLAMA_BIN, os.X_OK):
        print(f"✓ Ollama found at {OLLAMA_BIN}.")
        _ensure_on_path(OLLAMA_INSTALL_DIR)
        return True

    system  = platform.system().lower()
    machine = platform.machine().lower()

    banner(f"Installing Ollama (no root) — {system}/{machine}")
    os.makedirs(OLLAMA_INSTALL_DIR, exist_ok=True)

    # ── Linux ─────────────────────────────────────────────────────────────
    if system == "linux":
        arch = "arm64" if machine in ("aarch64", "arm64") else "amd64"

        # Resolve latest version dynamically — no need to update code on new releases
        tag = _get_latest_ollama_tag()
        if tag:
            url = f"{_OLLAMA_DL_BASE}/{tag}/ollama-linux-{arch}.tar.zst"
        else:
            # GitHub API unavailable — try the /releases/latest/download/ redirect
            url = (f"https://github.com/ollama/ollama/releases/latest"
                   f"/download/ollama-linux-{arch}.tar.zst")

        with tempfile.NamedTemporaryFile(suffix=".tar.zst", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            label = f"ollama-linux-{arch}.tar.zst ({tag or 'latest'})"
            if not _download(url, tmp_path, label=label):
                print("ERROR: Could not download Ollama binary. Check network access.")
                return False

            if not _extract_zst_tar(tmp_path, OLLAMA_BIN):
                return False

            os.chmod(OLLAMA_BIN, 0o755)
            _ensure_on_path(OLLAMA_INSTALL_DIR)
            print(f"✓ Ollama {tag or ''} installed → {OLLAMA_BIN}")
            return True

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    # ── macOS ─────────────────────────────────────────────────────────────
    elif system == "darwin":
        # Try Homebrew first (doesn't need root for user-space Homebrew installs)
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
            if not _extract_zst_tar(tmp_path, OLLAMA_BIN):
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
