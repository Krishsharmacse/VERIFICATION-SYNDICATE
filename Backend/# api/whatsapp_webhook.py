# api/whatsapp_webhook.py
from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
import logging
from typing import Optional
import io
from PIL import Image
import base64

from config import Config
from graph.verification_syndicate import VerificationSyndicate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Verification Syndicate API")

# Initialize Twilio client
twilio_client = Client(Config.TWILIO_ACCOUNT_SID, Config.TWILIO_AUTH_TOKEN)

# Initialize verification syndicate
verification_syndicate = VerificationSyndicate()


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Verification Syndicate",
        "version": "1.0.0"
    }


@app.post("/webhook/whatsapp")
async def whatsapp_webhook(
    request: Request,
    Body: Optional[str] = Form(None),
    From: Optional[str] = Form(None),
    MessageSid: Optional[str] = Form(None),
    NumMedia: Optional[int] = Form(0),
    MediaUrl0: Optional[str] = Form(None),
    MediaContentType0: Optional[str] = Form(None)
):
    """
    Webhook for incoming WhatsApp messages from Twilio
    """
    logger.info(f"Received message from {From}: {Body}")
    
    try:
        # Process media if present
        image_data = None
        audio_data = None
        
        if NumMedia and int(NumMedia) > 0:
            # Download media from Twilio URL
            if MediaContentType0 and MediaContentType0.startswith('image/'):
                # Download image
                media = twilio_client.api.accounts(Config.TWILIO_ACCOUNT_SID) \
                    .messages(MessageSid) \
                    .media(MediaUrl0.split('/')[-1]) \
                    .fetch()
                
                image_data = base64.b64decode(media)
                
            elif MediaContentType0 and MediaContentType0.startswith('audio/'):
                # Download audio
                media = twilio_client.api.accounts(Config.TWILIO_ACCOUNT_SID) \
                    .messages(MessageSid) \
                    .media(MediaUrl0.split('/')[-1]) \
                    .fetch()
                
                audio_data = base64.b64decode(media)
        
        # Process message through verification syndicate
        result = await verification_syndicate.process_message(
            text=Body,
            image=image_data,
            audio=audio_data,
            sender=From,
            message_sid=MessageSid
        )
        
        # Generate Twilio response
        twilio_response = MessagingResponse()
        
        if result["success"]:
            # Add main message
            msg = twilio_response.message()
            msg.body(result["response"])
            
            # Add educational content as separate messages if needed
            if result.get("educational_content"):
                for content in result["educational_content"][:2]:  # Limit to 2 items
                    educational_msg = f"📚 *{content['tactic']}*\n\n"
                    educational_msg += f"{content['description']}\n\n"
                    educational_msg += f"🔍 *How to spot:* {content['how_to_spot']}"
                    
                    msg = twilio_response.message()
                    msg.body(educational_msg)
            
            # Add verification tips
            if result.get("verification_tips"):
                tips_msg = "✅ *Verification Tips:*\n"
                for i, tip in enumerate(result["verification_tips"][:3], 1):
                    tips_msg += f"{i}. {tip}\n"
                
                msg = twilio_response.message()
                msg.body(tips_msg)
            
            # Add counter-narrative if available
            if result.get("counter_narrative"):
                msg = twilio_response.message()
                msg.body(f"ℹ️ *Fact Check:*\n{result['counter_narrative']}")
        else:
            # Error response
            msg = twilio_response.message()
            msg.body("Sorry, we encountered an error processing your message. Please try again.")
        
        return PlainTextResponse(str(twilio_response), media_type="application/xml")
        
    except Exception as e:
        logger.error(f"Webhook error: {str(e)}", exc_info=True)
        
        # Return error response
        twilio_response = MessagingResponse()
        msg = twilio_response.message()
        msg.body("An error occurred while processing your message. Please try again later.")
        
        return PlainTextResponse(str(twilio_response), media_type="application/xml")


@app.post("/api/verify")
async def verify_message(
    request: Request,
    text: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None)
):
    """
    API endpoint for direct verification (for testing/integration)
    """
    try:
        # Process image if provided
        image_data = None
        if image:
            contents = await image.read()
            image_data = contents
        
        # Process message
        result = await verification_syndicate.process_message(
            text=text,
            image=image_data
        )
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"API error: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )


@app.get("/health")
async def health_check():
    """Detailed health check endpoint"""
    return {
        "status": "healthy",
        "components": {
            "twilio": "configured" if Config.TWILIO_ACCOUNT_SID else "missing",
            "vertex_ai": "configured" if Config.PROJECT_ID else "missing",
            "verification_syndicate": "initialized"
        }
    }