# models/multimodal_processor.py
import base64
import io
from typing import Union, Dict, Any, List
from PIL import Image
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
import logging
from Backend.config import Config

logger = logging.getLogger(__name__)

class MultimodalInputProcessor:
    """Process multimodal inputs using LangChain & Gemini 1.5 Flash"""
    
    def __init__(self):
        # PROTOTYPE OPTIMIZATION: Make API Key optional for demo
        self.llm_available = False
        if not Config.GEMINI_API_KEY or "your_key" in Config.GEMINI_API_KEY:
            import logging
            logging.getLogger(__name__).warning("GEMINI_API_KEY not found. Running in LITE/MOCK mode.")
            self.llm = None
        else:
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    google_api_key=Config.GEMINI_API_KEY,
                    temperature=0.1
                )
                self.llm_available = True
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(f"Gemini Init Error: {e}")
                self.llm = None
    
    async def process_text(self, text: str) -> Dict[str, Any]:
        """Process text input with deep claim extraction using Gemini"""
        if not self.llm:
            logger.info("Bypassing Text AI (No LLM). Using linguistic fallback.")
            return {
                "type": "text",
                "content": text,
                "analysis": "Heuristic analysis complete. No Gemini key detected.",
                "claims": [text[:100]] if len(text) > 15 else [],
                "fake_probability": await self._calculate_heuristic_score(text)
            }

        messages = [
            SystemMessage(content="You are a cognitive security analyst specializing in misinformation detection. Extract facts and analyze tone."),
            HumanMessage(content=f"""Analyze this message for potential misinformation indicators:
"{text}"

Extract and analyze:
1. MAIN CLAIMS: List exactly the specific factual assertions made.
2. EMOTIONAL TONE: Identify anger, fear, urgency, or empathy tactics.
3. SCARCITY TACTICS: Does it claim limited time or exclusive info?
4. LINGUISTIC PATTERNS: Is it clickbait, sensational, or manipulative?

Provide a detailed summary and then list the claims clearly.""")
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            output = response.content
            
            return {
                "type": "text",
                "content": text,
                "analysis": output,
                "claims": self._extract_claims(output),
                "fake_probability": self._calculate_manipulation_score(output)
            }
        except Exception as e:
            logger.error(f"Gemini Text Error: {str(e)}")
            return {"type": "text", "content": text, "analysis": "Analysis failed", "claims": [text[:50]]}

    async def _calculate_heuristic_score(self, text: str) -> float:
        """Helper for mock text analysis"""
        sensational = ['urgent', 'alert', 'breaking', 'shocking']
        words = text.lower().split()
        score = sum(1 for w in words if w in sensational) / 4
        return min(0.6, 0.3 + score)

    async def process_image(self, image_data: bytes) -> Dict[str, Any]:
        """Process image input with OCR and forensics analysis using Gemini Vision"""
        if not self.llm:
             return {
                "type": "image",
                "analysis": "Image analysis bypassed (No Gemini Key). Forensic markers only.",
                "manipulation_score": 0.5,
                "claims": []
            }

        b64_image = base64.b64encode(image_data).decode('utf-8')
        
        messages = [
            HumanMessage(
                content=[
                    {"type": "text", "text": """ACT AS A DIGITAL FORENSICS AND MISINFORMATION ANALYST.
Analyze this image for potential misinformation indicators:
1. IMAGE MANIPULATION: Look for signs of AI generation, photoshopping, or out-of-context usage.
2. OCR TEXT EXTRACTION: Extract ALL text found in the image.
3. CONTEXTUAL ANALYSIS: What story is this image trying to tell?
4. EMOTIONAL TRIGGER: Does it use shock, fear, or outrage?"""},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{b64_image}"}
                ]
            )
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            output = response.content
            
            return {
                "type": "image",
                "analysis": output,
                "manipulation_score": self._calculate_manipulation_score(output),
                "claims": self._extract_claims(output)
            }
        except Exception as e:
            logger.error(f"Gemini Image Error: {str(e)}")
            return {"type": "image", "analysis": "Image analysis failed", "manipulation_score": 0.5, "claims": []}
    
    async def process_audio(self, audio_data: bytes) -> Dict[str, Any]:
        """Process audio input with transcription and tone analysis using Gemini 1.5"""
        if not self.llm:
            return {
                "type": "audio",
                "transcription": "Audio processing disabled (No Gemini Key).",
                "analysis": "Audio analysis bypassed.",
                "claims": []
            }

        b64_audio = base64.b64encode(audio_data).decode('utf-8')
        
        # Note: Gemini 1.5 supports native audio processing
        messages = [
            HumanMessage(
                content=[
                    {"type": "text", "text": """Transcribe and analyze this audio for misinformation indicators:
1. FULL TRANSCRIPTION: Provide the exact spoken words.
2. EMOTIONAL SPECTRUM: Analyze the speaker's tone for urgency, fear, or manipulation.
3. KEY ASSERTIONS: What specific claims are being made in the speech?
4. AUDIO AUTHENTICITY: Does it sound synthetic (deepfake) or edited?"""},
                    {
                        "type": "media",
                        "mime_type": "audio/mp3",
                        "data": b64_audio
                    }
                ]
            )
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            output = response.content
            
            return {
                "type": "audio",
                "transcription": output,
                "analysis": output,
                "claims": self._extract_claims(output)
            }
        except Exception as e:
            logger.error(f"Gemini Audio Error: {str(e)}")
            return {"type": "audio", "analysis": "Audio analysis failed", "claims": []}
    
    def _extract_claims(self, analysis: str) -> List[str]:
        """Extract individual claims from analysis"""
        claims = []
        lines = analysis.split('\n')
        for line in lines:
            line = line.strip()
            if not line: continue
            if line.startswith(('-', '*', '•', '1.', '2.', '3.')):
                clean_claim = line.lstrip('-*• 123456789.').strip()
                if len(clean_claim) > 15:
                    claims.append(clean_claim)
        
        return list(set(claims))[:5]
    
    def _calculate_manipulation_score(self, analysis: str) -> float:
        """Calculate manipulation score based on forensic indicators"""
        manipulation_indicators = {
            'manipulated': 0.2, 'fake': 0.2, 'altered': 0.15,
            'photoshopped': 0.2, 'ai-generated': 0.3, 'generated': 0.1,
            'inconsistent': 0.1, 'unnatural': 0.1, 'misleading': 0.1
        }
        score = 0.0
        analysis_lower = analysis.lower()
        for indicator, weight in manipulation_indicators.items():
            if indicator in analysis_lower:
                score += weight
        return min(1.0, score)
