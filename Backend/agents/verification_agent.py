# agents/verification_agent.py
from duckduckgo_search import DDGS
from typing import List, Dict, Any
import asyncio
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from Backend.config import Config

logger = logging.getLogger(__name__)

class DuckDuckGoVerificationAgent:
    """Live search verification agent using DuckDuckGo (Gemini Bypassed)"""
    
    def __init__(self):
        self.ddgs = DDGS()
        if not Config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in configuration")
        
        # LLM initialized but not used in current bypass mode
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest",
            google_api_key=Config.GEMINI_API_KEY,
            temperature=0.1
        )
    
    async def verify_claims(self, claims: List[str]) -> Dict[str, Any]:
        """Verify claims using live search and AI analysis"""
        
        verification_results = []
        
        for claim in claims:
            result = await self._search_and_analyze_claim(claim)
            verification_results.append(result)
        
        # Aggregate results
        avg_fake_prob = sum(r.get('fake_probability', 0.5) for r in verification_results) / max(1, len(verification_results))
        
        overall_veracity = "UNCERTAIN"
        if avg_fake_prob > 0.7: overall_veracity = "FAKE"
        elif avg_fake_prob > 0.55: overall_veracity = "LIKELY_FAKE"
        elif avg_fake_prob < 0.3: overall_veracity = "TRUE"
        elif avg_fake_prob < 0.45: overall_veracity = "LIKELY_TRUE"
        
        overall_confidence = sum(r.get('confidence', 0.5) for r in verification_results) / max(1, len(verification_results))
        
        return {
            "model": "DuckDuckGo + Gemini Analysis",
            "verification_results": verification_results,
            "overall_confidence": overall_confidence,
            "overall_veracity": overall_veracity,
            "fake_probability": avg_fake_prob,
            "evidence": [r.get('reasoning') for r in verification_results if r.get('reasoning')]
        }
    
    async def _search_and_analyze_claim(self, claim: str) -> Dict[str, Any]:
        """Search and use Gemini to analyze the results"""
        if not claim or len(claim.strip()) < 5:
            return {"claim": claim, "verdict": "UNCERTAIN", "confidence": 0}

        try:
            # Perform search (Sync call in executor)
            search_results = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: list(self.ddgs.text(claim + " fact check", max_results=5))
            )
            
            if not search_results:
                return {"claim": claim, "verdict": "UNCERTAIN", "confidence": 0.3, "reasoning": "No search results found to verify this."}

            snippets = "\n".join([f"SOURCE: {r.get('title')}\nCONTENT: {r.get('body')}" for r in search_results])
            
            # Use Gemini to cross-reference search results with the claim
            prompt = f"""Cross-reference this CLAIM with the SEARCH RESULTS and provide a verdict.

CLAIM: "{claim}"

SEARCH RESULTS:
{snippets}

TASK:
1. Is the claim supported or contradicted by the results?
2. Determine VERDICT: TRUE, FALSE, or UNCERTAIN.
3. Provide a FAKE_PROBABILITY (0.0 to 1.0).
4. Provide REASONING (max 2 sentences).

OUTPUT FORMAT:
VERDICT: [Verdict]
FAKE_PROBABILITY: [Score]
REASONING: [Text]"""

            messages = [
                SystemMessage(content="You are a precise fact-checking assistant."),
                HumanMessage(content=prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            output = response.content
            
            verdict = self._extract_value(output, "VERDICT") or "UNCERTAIN"
            fake_prob = float(self._extract_value(output, "FAKE_PROBABILITY") or 0.5)
            reasoning = self._extract_value(output, "REASONING") or "Search analysis inconclusive."
            
            return {
                "claim": claim,
                "verdict": verdict,
                "fake_probability": fake_prob,
                "confidence": 0.8,
                "reasoning": reasoning,
                "found_contradicting": verdict == "FALSE"
            }
            
        except Exception as e:
            logger.error(f"Search/Analysis error: {str(e)}")
            return {"claim": claim, "verdict": "ERROR", "fake_probability": 0.5, "error": str(e)}

    def _extract_value(self, text: str, key: str) -> str:
        for line in text.split('\n'):
            if line.upper().startswith(key.upper() + ":"):
                parts = line.split(':', 1)
                if len(parts) > 1:
                    return parts[1].strip()
        return ""