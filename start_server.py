import sys
print("🔵 Starting server file", flush=True)

from pyngrok import ngrok
print("🟢 Imported ngrok", flush=True)

import uvicorn
print("🟢 Imported uvicorn", flush=True)

# Open public tunnel
print("🟡 Opening ngrok tunnel...", flush=True)
public_url = ngrok.connect(8000)
print("🚀 PUBLIC API URL:", public_url, flush=True)

# Start FastAPI server
print("🟣 Starting FastAPI...", flush=True)
uvicorn.run(
    "backend.main:app",
    host="0.0.0.0",
    port=8000,
    log_level="info",
)
