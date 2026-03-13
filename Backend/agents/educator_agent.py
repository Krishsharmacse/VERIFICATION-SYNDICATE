import logging
from typing import Dict, Any, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from Backend.config import Config

logger = logging.getLogger(__name__)

class EducatorAgent:
    """Educational agent that generates explanations and counter-narratives using Gemini"""
    
    def __init__(self):
        # PROTOTYPE OPTIMIZATION: AI optional for demo
        self.llm_available = False
        if not Config.GEMINI_API_KEY or "your_key" in Config.GEMINI_API_KEY:
            logger.warning("EducatorAgent running in Template mode (AI disabled).")
            self.llm = None
        else:
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash", 
                    google_api_key=Config.GEMINI_API_KEY,
                    temperature=0.7
                )
                self.llm_available = True
            except Exception as e:
                logger.error(f"EducatorAgent Gemini Init Error: {e}")
                self.llm = None
        self.tactic_explanations = {
            "Fear mongering": {
                "description": "This message uses fear to make you react emotionally rather than logically.",
                "how_to_spot": "Look for words that create panic or extreme concern without evidence.",
                "why_it_works": "Fear triggers our survival instincts, making us less likely to verify facts."
            },
            "False urgency": {
                "description": "Creates artificial deadlines to pressure you into sharing without thinking.",
                "how_to_spot": "Be suspicious of messages demanding immediate action or sharing.",
                "why_it_works": "Urgency bypasses our normal fact-checking processes."
            },
            "Emotional overload": {
                "description": "Uses multiple strong emotions to overwhelm your critical thinking.",
                "how_to_spot": "Notice if the message makes you feel many strong emotions at once.",
                "why_it_works": "Overwhelmed brains struggle to process information logically."
            },
            "Empathy exploitation": {
                "description": "Uses emotional stories about vulnerable people to manipulate you.",
                "how_to_spot": "Check if the message uses emotional stories instead of facts.",
                "why_it_works": "We're naturally protective of vulnerable groups, making us less critical."
            },
            "Scarcity manipulation": {
                "description": "Creates false scarcity to make information seem more valuable.",
                "how_to_spot": "Look for claims about limited time or availability without reason.",
                "why_it_works": "We value things more when we think they're scarce."
            },
            "Factual falsehood": {
                "description": "Contains claims that contradict verified facts.",
                "how_to_spot": "Cross-check claims with reliable sources before believing.",
                "why_it_works": "False information often contains elements of truth to seem credible."
            }
        }
    
    async def generate_response(self, 
                              judgment: Dict[str, Any], 
                              original_message: str,
                              analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Generate educational response based on judgment using Gemini"""
        
        verdict = judgment.get("final_verdict", "UNCERTAIN")
        tactics = judgment.get("manipulation_tactics", [])
        evidence = judgment.get("supporting_evidence", [])
        
        if not self.llm:
            logger.info("Bypassing Educator AI (No LLM). Using templates.")
            # Fallback to templates if LLM is disabled
            main_message = self._generate_fake_message(judgment, tactics) if verdict in ["FAKE", "LIKELY_FAKE"] else self._generate_true_message(judgment)
            return {
                "main_message": main_message,
                "educational_content": self._generate_educational_content(tactics),
                "counter_narrative": self._generate_counter_narrative(original_message, tactics) if verdict in ["FAKE", "LIKELY_FAKE"] else None,
                "verification_tips": self._generate_verification_tips(tactics),
                "response_type": "template_fallback"
            }

        try:
            prompt = f"""You are a digital literacy educator. Based on a fact-checking report, create an educational response.

ORIGINAL MESSAGE: "{original_message}"
VERDICT: {verdict}
RED FLAGS: {tactics}
EVIDENCE: {evidence}

TASK:
1. Generate a MAIN MESSAGE: A clear summary of the verdict and why.
2. Generate EDUCATIONAL CONTENT: Explain the specific manipulation tactics used (if any).
3. Generate a COUNTER-NARRATIVE: Provide the factual alternative or context.
4. Generate 3 VERIFICATION TIPS: Specific advice for this type of content.

OUTPUT FORMAT (JSON):
{{
  "main_message": "...",
  "educational_content": [{{ "tactic": "...", "description": "...", "how_to_spot": "..." }}],
  "counter_narrative": "...",
  "verification_tips": ["...", "...", "..."]
}}"""

            messages = [
                SystemMessage(content="You are a helpful and clear digital literacy expert. Output only valid JSON."),
                HumanMessage(content=prompt)
            ]
            
            import json
            response = await self.llm.ainvoke(messages)
            content = response.content
            # Basic JSON extraction in case of markdown wrapping
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            data = json.loads(content)
            return data
            
        except Exception as e:
            logger.error(f"Educator LLM failed: {str(e)}. Falling back to templates.")
            # Fallback to templates if LLM fails
            main_message = self._generate_fake_message(judgment, tactics) if verdict in ["FAKE", "LIKELY_FAKE"] else self._generate_true_message(judgment)
            return {
                "main_message": main_message,
                "educational_content": self._generate_educational_content(tactics),
                "counter_narrative": self._generate_counter_narrative(original_message, tactics) if verdict in ["FAKE", "LIKELY_FAKE"] else None,
                "verification_tips": self._generate_verification_tips(tactics),
                "response_type": "fallback"
            }
    
    def _generate_fake_message(self, judgment: Dict[str, Any], tactics: List[Dict]) -> str:
        """Generate message for fake content"""
        
        probability = judgment.get("fake_probability", 0.7)
        
        if probability > 0.8:
            confidence_phrase = "highly likely"
        elif probability > 0.6:
            confidence_phrase = "likely"
        else:
            confidence_phrase = "possibly"
        
        message = f"⚠️ *MISINFORMATION ALERT*\n\n"
        message += f"This message is {confidence_phrase} to contain misinformation.\n\n"
        
        if tactics:
            message += "🚩 *Red Flags Detected:*\n"
            for tactic in tactics[:3]:  # Show top 3 tactics
                t_name = tactic.get('tactic', 'Unknown')
                message += f"• {t_name}\n"
        
        message += "\nPlease read the educational content below to understand why this is misleading."
        
        return message
    
    def _generate_true_message(self, judgment: Dict[str, Any]) -> str:
        """Generate message for likely true content"""
        
        probability = judgment.get("fake_probability", 0.2)
        
        if probability < 0.3:
            confidence_phrase = "appears to be"
        else:
            confidence_phrase = "might be"
        
        message = f"✅ *MESSAGE VERIFICATION*\n\n"
        message += f"This message {confidence_phrase} accurate based on our analysis.\n\n"
        message += "However, always verify important information from multiple reliable sources."
        
        return message
    
    def _generate_educational_content(self, tactics: List[Dict]) -> List[Dict]:
        """Generate educational content about manipulation tactics"""
        
        educational_items = []
        
        for tactic in tactics[:3]:  # Limit to top 3 tactics
            t_name = tactic.get("tactic")
            if t_name in self.tactic_explanations:
                explanation = self.tactic_explanations[t_name]
                educational_items.append({
                    "tactic": t_name,
                    "severity": tactic.get("severity", "medium"),
                    "description": explanation["description"],
                    "how_to_spot": explanation["how_to_spot"],
                    "why_it_works": explanation["why_it_works"]
                })
        
        # Add general media literacy if no specific tactics
        if not educational_items:
            educational_items.append({
                "tactic": "General misinformation awareness",
                "severity": "low",
                "description": "Always verify information from multiple reliable sources.",
                "how_to_spot": "Check the source, date, and look for supporting evidence.",
                "why_it_works": "False information often spreads faster than truth."
            })
        
        return educational_items
    
    def _generate_counter_narrative(self, original_message: str, tactics: List[Dict]) -> str:
        """Generate counter-narrative to combat misinformation"""
        
        counter = "Here's what you should know:\n\n"
        
        t_names = [t.get("tactic") for t in tactics]
        if "Fear mongering" in t_names:
            counter += "• While this message uses fear, the actual risks are often exaggerated.\n"
        
        if "False urgency" in t_names:
            counter += "• There's no real deadline - take time to verify before sharing.\n"
        
        if "Scarcity manipulation" in t_names:
            counter += "• Claims of scarcity are often exaggerated to create pressure.\n"
        
        if "Factual falsehood" in t_names:
            counter += "• The facts in this message don't match verified information.\n"
        
        counter += "\nAlways verify information from trusted sources before acting on it."
        
        return counter
    
    def _generate_verification_tips(self, tactics: List[Dict]) -> List[str]:
        """Generate specific verification tips based on tactics"""
        
        tips = []
        
        # General tips
        tips.append("Check the source - is it a reputable news organization?")
        tips.append("Look for the same information from multiple reliable sources")
        tips.append("Check the date - old news often resurfaces as current")
        
        t_names = [t.get("tactic") for t in tactics]
        # Tactic-specific tips
        if "Fear mongering" in t_names:
            tips.append("Ask yourself: Is the fear justified by evidence?")
        if "False urgency" in t_names:
            tips.append("Ignore urgency claims - real emergencies don't need messaging apps")
        if "Scarcity manipulation" in t_names:
            tips.append("Verify scarcity claims with official sources")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_tips = []
        for tip in tips:
            if tip not in seen:
                unique_tips.append(tip)
                seen.add(tip)
        
        return unique_tips[:5]  # Limit to 5 tips
