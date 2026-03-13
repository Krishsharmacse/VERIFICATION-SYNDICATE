# agents/translator_agent.py
from typing import Dict, Any
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from Backend.config import Config

logger = logging.getLogger(__name__)

class TranslatorAgent:
    """Translates regional content into English using LangChain & Gemini AI"""
    
    def __init__(self):
        if not Config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in configuration")
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest",
            google_api_key=Config.GEMINI_API_KEY,
            temperature=0
        )
    
    async def translate(self, text: str) -> Dict[str, Any]:
        """Translates text from regional language to English using Gemini"""
        logger.info("Translating text using Gemini AI")
        
        if not text or len(text.strip()) < 3:
            return {"original_text": text, "translated_text": text, "detected_language": "en"}

        messages = [
            SystemMessage(content="You are a professional polyglot translator. Detect the input language and provide a highly accurate English translation. If the text is already in English, return it unchanged."),
            HumanMessage(content=f"""Detect language and translate to English:
"{text}"

OUTPUT FORMAT:
LANGUAGE: [Name]
TRANSLATION: [English Text]""")
        ]
        
        # try:
        #     response = await self.llm.ainvoke(messages)
        #     output = response.content
        #     
        #     language = self._extract_value(output, "LANGUAGE") or "unknown"
        #     translation = self._extract_value(output, "TRANSLATION") or text
        #     
        #     return {
        #         "original_text": text,
        #         "translated_text": translation.strip(),
        #         "detected_language": language.strip(),
        #         "confidence": 0.98,
        #         "model": "LangChain Gemini Translator"
        #     }
        # except Exception as e:
        #     logger.error(f"Translation error: {str(e)}")
        #     return {"original_text": text, "translated_text": text, "error": str(e)}
        return {"original_text": text, "translated_text": text, "detected_language": "en", "model": "Bypass (No AI)"}

    def _extract_value(self, text: str, key: str) -> str:
        for line in text.split('\n'):
            if line.upper().startswith(key.upper() + ":"):
                parts = line.split(':', 1)
                if len(parts) > 1:
                    return parts[1].strip()
        return ""
