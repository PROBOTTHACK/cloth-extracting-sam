import os
import sys
from pyngrok import ngrok
import uvicorn

# Absolute path to repo
REPO_PATH = "/kaggle/working/cloth-extracting-sam"

# Add repo to PYTHONPATH
sys.path.insert(0, REPO_PATH)

# Change working directory away from sam2 repo (important)
os.chdir("/kaggle/working")

print("📁 CWD:", os.getcwd(), flush=True)
print("📦 PYTHONPATH:", sys.path[:2], flush=True)

# Start ngrok
public_url = ngrok.connect(8000)
print("🚀 PUBLIC API URL:", public_url, flush=True)

# Start FastAPI
uvicorn.run(
    "backend.main:app",
    host="0.0.0.0",
    port=8000,
    log_level="info",
)
