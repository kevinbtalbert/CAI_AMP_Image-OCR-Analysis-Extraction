import subprocess
import os
import sys

port = os.environ.get("CDSW_APP_PORT", "8501")

try:
    # Works when executed as a script
    app_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Fallback for interactive CML sessions / notebooks
    app_dir = os.path.join(os.getcwd(), "2_application")

print(f"Starting app from: {app_dir} on port {port}")

print(subprocess.run(
    [sys.executable, "-m", "uvicorn", "app:app",
     "--host", "127.0.0.1",
     "--port", str(port),
     "--workers", "1"],
    cwd=app_dir
))
