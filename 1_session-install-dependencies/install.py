import subprocess
print(subprocess.run(
    ["pip", "install", "-r", "1_session-install-dependencies/requirements.txt"],
    capture_output=False
))
