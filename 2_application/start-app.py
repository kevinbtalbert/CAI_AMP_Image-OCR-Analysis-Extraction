import subprocess
import os
import sys

port = os.environ.get("CDSW_APP_PORT", "8501")
app_dir = os.path.join(os.path.dirname(__file__))

print(subprocess.run(
    [sys.executable, "-m", "uvicorn", "app:app",
     "--host", "127.0.0.1",
     "--port", str(port),
     "--workers", "1"],
    cwd=app_dir
))
