# main.py
import uvicorn
import os
import logging
import warnings

# Suppress warnings from third-party libraries
warnings.filterwarnings("ignore", category=DeprecationWarning, module="uvicorn")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="websockets")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="duckduckgo_search")

from Backend.api.gemini_api import app

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "Backend.api.gemini_api:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )