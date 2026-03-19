import subprocess
import os

print(subprocess.run(
    ["streamlit", "run", "./2_application/app.py",
     "--server.port", os.environ.get("CDSW_APP_PORT", "8501"),
     "--server.address", "127.0.0.1"],
    shell=False
))
