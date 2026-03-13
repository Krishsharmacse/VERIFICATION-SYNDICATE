# agents/apitube_agent.py
import aiohttp
import logging
from typing import List, Dict, Any
from Backend.config import Config

logger = logging.getLogger(__name__)

class APITubeVerificationAgent:
    """News verification agent using APITube.io"""
    
    def __init__(self):
        self.api_key = Config.APITUBE_API_KEY
        self.base_url = "https://api.apitube.io/v1"
        self.headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json"
        }

    async def verify_claims(self, claims: List[str]) -> Dict[str, Any]:
        """Verify claims by searching news and checking publisher rankings"""
        if not self.api_key:
            return {"error": "APITUBE_API_KEY not configured"}
            
        results = []
        for claim in claims:
            analysis = await self._analyze_claim_via_news(claim)
            results.append(analysis)
            
        # Aggregate results
        avg_fake_prob = sum(r.get('fake_probability', 0.5) for r in results) / max(1, len(results))
        avg_confidence = sum(r.get('confidence', 0.5) for r in results) / max(1, len(results))
        
        return {
            "model": "APITube News Analysis",
            "verification_results": results,
            "fake_probability": avg_fake_prob,
            "overall_confidence": avg_confidence,
            "overall_veracity": "UNCERTAIN" if 0.4 < avg_fake_prob < 0.6 else ("FAKE" if avg_fake_prob > 0.6 else "TRUE")
        }

    async def _analyze_claim_via_news(self, claim: str) -> Dict[str, Any]:
        """Search news and analyze results quality"""
        try:
            async with aiohttp.ClientSession() as session:
                params = {
                    "q": claim,
                    "limit": 5,
                    "language": "en"
                }
                async with session.get(f"{self.base_url}/news/everything", headers=self.headers, params=params) as resp:
                    if resp.status != 200:
                        logger.error(f"APITube error: {resp.status}")
                        return {"claim": claim, "fake_probability": 0.5, "confidence": 0}
                    
                    data = await resp.json()
                    articles = data.get('results', [])
                    
                    if not articles:
                        return {
                            "claim": claim, 
                            "fake_probability": 0.6, # Slight bias towards warning if no news found for a specific claim
                            "confidence": 0.3, 
                            "reasoning": "No verified news articles found matching this claim in the APITube index."
                        }
                    
                    # Heuristic: Check Online Publisher Rank (OPR)
                    # OPR is 1-10. High OPR = trustworthy.
                    total_opr = 0
                    count = 0
                    high_trust_sources = 0
                    
                    for art in articles:
                        source = art.get('source', {})
                        rank = source.get('rank', 5) # Default middle
                        total_opr += rank
                        count += 1
                        if rank >= 7:
                            high_trust_sources += 1
                    
                    avg_rank = total_opr / count if count > 0 else 5
                    
                    # If articles exist but sources are low rank, or sentiment is highly polarized (optional)
                    # For now, if we have high trust sources mentioning it, it's more likely true.
                    # Misinformation often lacks "high trust" coverage or has "debunk" coverage.
                    # We'll check titles for "fake" or "debunk" keywords too.
                    
                    fake_keywords = ['fake', 'false', 'hoax', 'misleading', 'untrue', 'rumor', 'debunked']
                    has_debunk = any(any(kw in art.get('title', '').lower() for kw in fake_keywords) for art in articles)
                    
                    fake_prob = 0.5
                    if high_trust_sources >= 2:
                        fake_prob = 0.2 # Likely true if covered by multiple high trust sources
                    elif has_debunk:
                        fake_prob = 0.9 # High probability fake if debunk markers found
                    elif avg_rank < 4:
                        fake_prob = 0.7 # Likely fake if only low trust sources cover it
                        
                    return {
                        "claim": claim,
                        "fake_probability": fake_prob,
                        "confidence": 0.7,
                        "reasoning": f"Found {len(articles)} news articles. Avg Source Trust (OPR): {avg_rank:.1f}/10. High trust coverage: {high_trust_sources}.",
                        "publisher_stats": {"avg_opr": avg_rank, "high_trust_count": high_trust_sources}
                    }
                    
        except Exception as e:
            logger.error(f"APITube logic error: {str(e)}")
            return {"claim": claim, "fake_probability": 0.5, "confidence": 0}
