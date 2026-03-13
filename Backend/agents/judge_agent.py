# agents/judge_agent.py
from typing import Dict, Any, List
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from Backend.config import Config

logger = logging.getLogger(__name__)

class JudgeAgent:
    """Final judge that synthesizes all analyses and makes a final verdict using LangChain & Gemini AI"""
    
    def __init__(self):
        # PROTOTYPE OPTIMIZATION: Make AI optional
        self.llm_available = False
        if not Config.GEMINI_API_KEY or "your_key" in Config.GEMINI_API_KEY:
            logger.warning("JudgeAgent running in Weighted Heuristic mode (AI disabled).")
            self.llm = None
        else:
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-pro", 
                    google_api_key=Config.GEMINI_API_KEY,
                    temperature=0.1
                )
                self.llm_available = True
            except Exception as e:
                logger.error(f"JudgeAgent Gemini Init Error: {e}")
                self.llm = None
    
    async def judge(self, analyses: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make final judgment based on all analyses using AI synthesis or Heuristic Fallback
        """
        
        try:
            # Prepare context for the AI Judge
            context_summary = "RESULTS FROM SPECIALIZED AGENTS:\n\n"
            
            for agent, result in analyses.items():
                if result and not result.get('error'):
                    context_summary += f"### {agent.upper()} AGENT OUTPUT:\n"
                    context_summary += f"- Probability of fake content: {result.get('fake_probability', 'N/A')}\n"
                    # Include analysis text but truncate if too long
                    analysis_text = str(result.get('analysis', result.get('response', 'N/A')))
                    if len(analysis_text) > 500:
                        analysis_text = analysis_text[:500] + "..."
                    context_summary += f"- Analysis Summary: {analysis_text}\n"
                    
                    if result.get('manipulation_tactics'):
                        context_summary += f"- Detected Tactics: {result.get('manipulation_tactics')}\n"
                    if result.get('verification_results'):
                        context_summary += f"- Search Results Summary: {result.get('verification_results')}\n"
                    context_summary += "\n"

            if self.llm:
                messages = [
                    SystemMessage(content="You are an expert fact-checker and cognitive security analyst. Your task is to synthesize multiple agent reports into a final verdict. You must be precise, objective, and highlight specific red flags."),
                    HumanMessage(content=f"""Synthesize the following reports from our verification syndicate and provide a final determination:

{context_summary}

TASK: 
1. Determine the FINAL VERDICT: Choose exactly one: TRUE, LIKELY_TRUE, UNCERTAIN, LIKELY_FAKE, FAKE.
2. Provide a CONFIDENCE score (0.0 to 1.0) based on how consistent the reports are.
3. Provide a FAKE_PROBABILITY estimate (0.0 to 1.0) - higher means more likely to be misinformation.
4. Identify the specific MANIPULATION TACTICS found across ALL reports.
5. Provide a clear, actionable RECOMMENDATION for the user.
6. List the 3 most critical pieces of EVIDENCE from the reports.

OUTPUT FORMAT (Follow strictly):
VERDICT: [Verdict]
CONFIDENCE: [Score]
FAKE_PROBABILITY: [Score]
TACTICS: [Tactic 1, Tactic 2, etc.]
RECOMMENDATION: [Text]
EVIDENCE: [Bullet points]""")
                ]
                try:
                    response = await self.llm.ainvoke(messages)
                    output = response.content
                    
                    # Parse response
                    final_verdict = self._extract_value(output, "VERDICT") or "UNCERTAIN"
                    confidence = float(self._extract_value(output, "CONFIDENCE") or 0.5)
                    fake_prob = float(self._extract_value(output, "FAKE_PROBABILITY") or 0.5)
                    recommendation = self._extract_value(output, "RECOMMENDATION") or "Verify with reliable sources."
                    all_tactics = self._collect_tactics(output)
                    
                    return {
                        "final_verdict": final_verdict.strip().upper(),
                        "fake_probability": fake_prob,
                        "confidence": confidence,
                        "manipulation_tactics": all_tactics,
                        "supporting_evidence": self._collect_evidence(output),
                        "recommendation": recommendation.strip()
                    }
                except Exception as e:
                    logger.error(f"AI synthesis failed: {e}")
            
            # Weighted Heuristic Fallback
            weighted_probs = []
            for agent, result in analyses.items():
                if isinstance(result, dict) and 'fake_probability' in result:
                    prob = float(result['fake_probability'])
                    weight = 1.0
                    if agent == 'duckduckgo': weight = 2.0
                    if agent == 'apitube': weight = 2.5
                    weighted_probs.append((prob, weight))
            
            avg_prob = sum(p * w for p, w in weighted_probs) / sum(w for p, w in weighted_probs) if weighted_probs else 0.5
            verdict = "FAKE" if avg_prob > 0.7 else ("TRUE" if avg_prob < 0.3 else "UNCERTAIN")
            
            all_tactics = []
            for agent, result in analyses.items():
                if isinstance(result, dict) and 'manipulation_tactics' in result:
                    all_tactics.extend(result['manipulation_tactics'])

            return {
                "final_verdict": verdict,
                "fake_probability": avg_prob,
                "confidence": 0.5,
                "manipulation_tactics": all_tactics,
                "supporting_evidence": ["Fallback heuristic analysis used."],
                "recommendation": "Manual verification recommended."
            }

            
        except Exception as e:
            logger.error(f"Judge error: {str(e)}")
            return {
                "final_verdict": "ERROR",
                "fake_probability": 0.5,
                "confidence": 0,
                "error": str(e),
                "manipulation_tactics": [],
                "supporting_evidence": []
            }

    def _extract_value(self, text: str, key: str) -> str:
        for line in text.split('\n'):
            if line.upper().startswith(key.upper() + ":"):
                parts = line.split(':', 1)
                if len(parts) > 1:
                    return parts[1].strip()
        return ""
    
    def _collect_tactics(self, output: str) -> List[Dict]:
        tactics_str = self._extract_value(output, "TACTICS")
        if not tactics_str: return []
        # Remove brackets if present
        tactics_str = tactics_str.replace('[', '').replace(']', '')
        items = [t.strip() for t in tactics_str.split(',')]
        return [{"tactic": t, "severity": "high" if "high" in output.lower() else "medium"} for t in items if t]

    def _collect_evidence(self, output: str) -> List[str]:
        evidence_str = self._extract_value(output, "EVIDENCE")
        lines = output.split('\n')
        evidence = []
        found_header = False
        
        for line in lines:
            if 'EVIDENCE:' in line.upper():
                found_header = True
                val = line.split(':', 1)[1].strip()
                if val: evidence.append(val)
                continue
            
            if found_header:
                if line.strip().startswith(('-', '*', '•', '1.', '2.', '3.')):
                    evidence.append(line.strip().lstrip('-*• 123.').strip())
                elif not line.strip() and evidence:
                    break
        
        return (evidence if evidence else ["Evidence synthesis pending further verification."])[:5]
