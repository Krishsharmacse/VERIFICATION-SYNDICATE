# main.py
import uvicorn
import logging
import warnings

# Suppress warnings from third-party libraries
warnings.filterwarnings("ignore", category=DeprecationWarning, module="uvicorn")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="websockets")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="duckduckgo_search")

from Backend.api.gemini_api import app

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        "Backend.api.gemini_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )