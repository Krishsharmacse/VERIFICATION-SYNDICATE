from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import logging
from pathlib import Path
from typing import Optional

# Import the core engine
from Backend.graph.verification_syndicate import VerificationSyndicate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Verification Syndicate API")
syndicate = VerificationSyndicate()

# Serve static files (frontend)
static_dir = Path(__file__).parent.parent.parent / "static"
logger.info(f"Checking for static directory at: {static_dir}")
if static_dir.exists():
    logger.info(f"Static directory found: {static_dir}")
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
else:
    logger.error(f"Static directory NOT found at: {static_dir}")

@app.get("/")
async def root():
    """Serve the frontend HTML"""
    frontend_file = static_dir / "index.html"
    if frontend_file.exists():
        return FileResponse(frontend_file, media_type="text/html")
    return {
        "status": "healthy",
        "service": "Verification Syndicate",
        "version": "2.0.0"
    }

@app.post("/api/verify")
async def verify_multimodal(
    text: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None)
):
    """Verify multimodal input using the agentic pipeline"""
    try:
        image_bytes = await image.read() if image else None
        audio_bytes = await audio.read() if audio else None
        
        result = await syndicate.process_message(
            text=text,
            image=image_bytes,
            audio=audio_bytes,
            sender="web_user",
            message_sid="web_session"
        )
        
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error in multimodal verification: {str(e)}")
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

@app.get("/api/verify/text")
async def verify_text(text: str):
    """Legacy text-only verification (now redirected to syndicate)"""
    try:
        result = await syndicate.process_message(text=text, sender="web_user")
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "ready": True,
        "frontend": "served" if (static_dir / "index.html").exists() else "not found"
    }
