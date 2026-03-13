# graph/verification_syndicate.py
from typing import Dict, Any, TypedDict, Optional, Annotated
import asyncio
import operator
from langgraph.graph import StateGraph, END
import logging

# Import agents
# Import agents and models using full package paths
from Backend.models.multimodal_processor import MultimodalInputProcessor
from Backend.classifiers import (
    DLClassifierAgent, 
    CLBClassifierAgent, 
    BharatFakeNewsKosh, 
    NovEmoFake
)
from Backend.agents.translator_agent import TranslatorAgent
from Backend.agents.rag_agent import RAGAgent
from Backend.agents.verification_agent import DuckDuckGoVerificationAgent
from Backend.agents.judge_agent import JudgeAgent
from Backend.agents.educator_agent import EducatorAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define state schema
class VerificationState(TypedDict):
    """State schema for the verification graph"""
    input_text: Optional[str]
    input_image: Optional[bytes]
    input_audio: Optional[bytes]
    sender: str
    message_sid: str
    
    # Processing results
    multimodal_analysis: Optional[Dict]
    dl_classification: Optional[Dict]
    clb_classification: Optional[Dict]
    bharat_analysis: Optional[Dict]
    novemofake_analysis: Optional[Dict]
    translation: Optional[Dict]
    rag_results: Optional[Dict]
    apitube_results: Optional[Dict]
    verification_results: Optional[Dict]
    
    # Final outputs
    judgment: Optional[Dict]
    educational_response: Optional[Dict]
    
    # Metadata
    error: Optional[str]
    processing_stage: Annotated[str, lambda x, y: y]


class VerificationSyndicate:
    """Multi-agent system for misinformation combat"""
    
    def __init__(self):
        # Initialize all agents
        self.multimodal_processor = MultimodalInputProcessor()
        self.dl_classifier = DLClassifierAgent()
        self.clb_classifier = CLBClassifierAgent()
        self.bharat_model = BharatFakeNewsKosh()
        self.novemofake_model = NovEmoFake()
        self.translator_agent = TranslatorAgent()
        self.rag_agent = RAGAgent()
        self.verification_agent = DuckDuckGoVerificationAgent()
        from Backend.agents.apitube_agent import APITubeVerificationAgent
        self.apitube_agent = APITubeVerificationAgent()
        self.judge_agent = JudgeAgent()
        self.educator_agent = EducatorAgent()
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        # Initialize graph
        workflow = StateGraph(VerificationState)
        
        # Add nodes
        workflow.add_node("orchestrator", self.orchestrator_node)
        workflow.add_node("multimodal_processor", self.multimodal_node)
        workflow.add_node("dl_classifier", self.dl_classifier_node)
        workflow.add_node("clb_classifier", self.clb_classifier_node)
        workflow.add_node("bharat_analyzer", self.bharat_analyzer_node)
        workflow.add_node("novemofake_analyzer", self.novemofake_node)
        workflow.add_node("translator", self.translator_node)
        workflow.add_node("rag_agent", self.rag_node)
        workflow.add_node("apitube_analyzer", self.apitube_node)
        workflow.add_node("verification", self.verification_node)
        workflow.add_node("judge", self.judge_node)
        workflow.add_node("educator", self.educator_node)
        
        # Define edges
        workflow.set_entry_point("orchestrator")
        
        # Branch after multimodal processing
        workflow.add_edge("multimodal_processor", "dl_classifier")
        workflow.add_edge("multimodal_processor", "clb_classifier")
        workflow.add_edge("multimodal_processor", "bharat_analyzer")
        workflow.add_edge("multimodal_processor", "novemofake_analyzer")
        workflow.add_edge("multimodal_processor", "translator")
        workflow.add_edge("multimodal_processor", "apitube_analyzer")
        
        # Translator to RAG Agent
        workflow.add_edge("translator", "rag_agent")
        
        # Join parallel nodes into verification
        for node in ["dl_classifier", "clb_classifier", "bharat_analyzer", "novemofake_analyzer", "rag_agent", "apitube_analyzer"]:
            workflow.add_edge(node, "verification")
        
        # Final steps
        workflow.add_edge("verification", "judge")
        workflow.add_edge("judge", "educator")
        workflow.add_edge("educator", END)
        
        # Add conditional edges for error handling
        workflow.add_conditional_edges(
            "orchestrator",
            self.check_error,
            {
                "error": END,
                "continue": "multimodal_processor"
            }
        )
        
        return workflow.compile()
    
    async def orchestrator_node(self, state: VerificationState) -> VerificationState:
        """Orchestrator node - manages state and routing"""
        logger.info(f"Processing message from {state.get('sender')}")
        
        try:
            # Validate input
            text = state.get('input_text', '')
            if not any([text, state.get('input_image'), state.get('input_audio')]):
                return {"error": "No input provided", "processing_stage": "error"}
            
            # PROTOTYPE OPTIMIZATION: Check if message is obviously safe/short
            # If it's just a greeting or very short, we can lower search priority
            if text and len(text.split()) < 3 and not any([state.get('input_image'), state.get('input_audio')]):
                logger.info("Short message detected, using Lite Analysis mode.")
                return {"processing_stage": "orchestrated", "is_lite_mode": True}
            
            return {"processing_stage": "orchestrated", "is_lite_mode": False}
            
        except Exception as e:
            logger.error(f"Orchestrator error: {str(e)}")
            return {"error": str(e), "processing_stage": "error"}
    
    async def multimodal_node(self, state: VerificationState) -> VerificationState:
        """Process multimodal input and prepare text for downstream agents"""
        logger.info("Processing multimodal input")
        
        try:
            multimodal_data = {}
            new_text = state.get('input_text')
            
            # Process image if present (OCR / Content Analysis)
            if state.get('input_image') and self.multimodal_processor:
                image_analysis = await self.multimodal_processor.process_image(state['input_image'])
                if image_analysis:
                    multimodal_data.update(image_analysis)
                    # If there's OCR text, use it as input for other agents if they lack text
                    if image_analysis.get('analysis') and not new_text:
                        new_text = image_analysis['analysis']
            
            # Process audio if present (Transcription)
            if state.get('input_audio') and self.multimodal_processor:
                audio_analysis = await self.multimodal_processor.process_audio(state['input_audio'])
                if audio_analysis:
                    if not multimodal_data:
                        multimodal_data = audio_analysis
                    else:
                        multimodal_data['audio'] = audio_analysis
                    
                    # If there's transcription, use it as input for other agents
                    if audio_analysis.get('transcription') and not new_text:
                        new_text = audio_analysis['transcription']

            # Process text if present
            if new_text and not multimodal_data and self.multimodal_processor:
                multimodal_data = await self.multimodal_processor.process_text(new_text)
            
            return {
                "multimodal_analysis": multimodal_data,
                "input_text": new_text,
                "processing_stage": "multimodal_processed"
            }
            
        except Exception as e:
            logger.error(f"Multimodal processing error: {str(e)}")
            return {"error": f"Multimodal Layer Error: {str(e)}", "processing_stage": "error"}
    
    async def dl_classifier_node(self, state: VerificationState) -> VerificationState:
        """Deep Learning Classifier node"""
        logger.info("Running DL classifier")
        
        try:
            text = state.get('input_text', '')
            image = None  
            
            result = await self.dl_classifier.classify(text, image)
            return {"dl_classification": result}
            
        except Exception as e:
            logger.error(f"DL classifier error: {str(e)}")
            return {"dl_classification": {"error": str(e), "fake_probability": 0.5, "confidence": 0}}
    
    async def clb_classifier_node(self, state: VerificationState) -> VerificationState:
        """Cross-Lingual BERT Classifier node"""
        logger.info("Running CLB classifier")
        
        try:
            text = state.get('input_text', '')
            language = "en"
            
            result = await self.clb_classifier.classify(text, language)
            return {"clb_classification": result}
            
        except Exception as e:
            logger.error(f"CLB classifier error: {str(e)}")
            return {"clb_classification": {"error": str(e), "fake_probability": 0.5, "confidence": 0}}
    
    async def bharat_analyzer_node(self, state: VerificationState) -> VerificationState:
        """Bharat Fake News Kosh analyzer node"""
        logger.info("Running Bharat Fake News analysis")
        
        try:
            text = state.get('input_text', '')
            result = await self.bharat_model.analyze(text)
            return {"bharat_analysis": result}
            
        except Exception as e:
            logger.error(f"Bharat analyzer error: {str(e)}")
            return {"bharat_analysis": {"error": str(e), "fake_probability": 0.5, "confidence": 0}}
    
    async def novemofake_node(self, state: VerificationState) -> VerificationState:
        """NovEmoFake emotional analysis node"""
        logger.info("Running NovEmoFake analysis")
        
        try:
            text = state.get('input_text', '')
            result = await self.novemofake_model.analyze(text)
            return {"novemofake_analysis": result}
            
        except Exception as e:
            logger.error(f"NovEmoFake error: {str(e)}")
            return {"novemofake_analysis": {"error": str(e), "fake_probability": 0.5, "confidence": 0}}
    
    async def translator_node(self, state: VerificationState) -> VerificationState:
        """Translator node for handling regional languages"""
        logger.info("Running NLP translation")
        try:
            text = state.get('input_text', '')
            result = await self.translator_agent.translate(text)
            return {"translation": result}
        except Exception as e:
            logger.error(f"Translator error: {str(e)}")
            return {"translation": {"error": str(e)}}

    async def rag_node(self, state: VerificationState) -> VerificationState:
        """RAG Agent for checking known fact-checks database (Vector Search)"""
        logger.info("Running RAG Vector Search")
        try:
            text_to_search = state.get('translation', {}).get('translated_text', state.get('input_text', ''))
            result = await self.rag_agent.retrieve([text_to_search])
            return {"rag_results": result}
        except Exception as e:
            logger.error(f"RAG error: {str(e)}")
            return {"rag_results": {"error": str(e), "confidence": 0.0}}

    async def apitube_node(self, state: VerificationState) -> VerificationState:
        """News analysis node using APITube.io"""
        logger.info("Running APITube News Analysis")
        try:
            claims = []
            mm_analysis = state.get('multimodal_analysis')
            if mm_analysis:
                claims = mm_analysis.get('claims', [])
            
            if not claims and state.get('input_text'):
                claims = [state['input_text']]
                
            if not claims:
                return {"apitube_results": {"fake_probability": 0.5, "confidence": 0}}
                
            result = await self.apitube_agent.verify_claims(claims)
            return {"apitube_results": result}
        except Exception as e:
            logger.error(f"APITube node error: {str(e)}")
            return {"apitube_results": {"error": str(e), "fake_probability": 0.5}}

    async def verification_node(self, state: VerificationState) -> VerificationState:
        """Live verification node (Fallback if RAG confidence < 70%)"""
        logger.info("Running verification node")
        
        try:
            # Extract claims from multimodal analysis
            claims = []
            mm_analysis = state.get('multimodal_analysis')
            if mm_analysis:
                claims = mm_analysis.get('claims', [])
            
            # If no claims extracted, use the whole text
            if not claims and state.get('input_text'):
                claims = [state['input_text']]
            
            verification_results = {}
            if claims:
                rag_confidence = state.get('rag_results', {}).get('confidence', 0.0)
                if rag_confidence >= 0.7:
                    logger.info("RAG search found strong match (>70%), skipping DuckDuckGo fallback")
                    verification_results = {
                        "model": "RAG Vector Search",
                        "overall_veracity": "fact_checked",
                        "overall_confidence": rag_confidence,
                        "message": "Matched with known fact-checks database",
                        "sources": state.get('rag_results', {}).get('matches', [])
                    }
                else:
                    logger.info("RAG confidence below 70%, falling back to live DuckDuckGo search")
                    verification_results = await self.verification_agent.verify_claims(claims)
            else:
                verification_results = {
                    "model": "DuckDuckGo Verification",
                    "overall_veracity": "uncertain",
                    "overall_confidence": 0,
                    "message": "No claims to verify"
                }
            
            return {"verification_results": verification_results, "processing_stage": "verified"}
            
        except Exception as e:
            logger.error(f"Verification error: {str(e)}")
            return {"verification_results": {"error": str(e), "overall_veracity": "error"}, "processing_stage": "verified"}
    
    async def judge_node(self, state: VerificationState) -> VerificationState:
        """Final judge node"""
        logger.info("Making final judgment")
        
        try:
            # Collect all analyses
            analyses = {
                "dl_classifier": state.get('dl_classification', {}),
                "clb_classifier": state.get('clb_classification', {}),
                "bharat_fake_news": state.get('bharat_analysis', {}),
                "novemofake": state.get('novemofake_analysis', {}),
                "rag": state.get('rag_results', {}),
                "apitube": state.get('apitube_results', {}),
                "duckduckgo": state.get('verification_results', {})
            }
            
            judgment = await self.judge_agent.judge(analyses)
            return {"judgment": judgment, "processing_stage": "judged"}
            
        except Exception as e:
            logger.error(f"Judge error: {str(e)}")
            return {
                "judgment": {"final_verdict": "ERROR", "error": str(e), "fake_probability": 0.5},
                "processing_stage": "error"
            }
    
    async def educator_node(self, state: VerificationState) -> VerificationState:
        """Educational response node"""
        logger.info("Generating educational response")
        
        try:
            # Collect analyses
            analyses = {
                "dl_classifier": state.get('dl_classification', {}),
                "clb_classifier": state.get('clb_classification', {}),
                "bharat_fake_news": state.get('bharat_analysis', {}),
                "novemofake": state.get('novemofake_analysis', {}),
                "duckduckgo": state.get('verification_results', {})
            }
            
            educational_response = await self.educator_agent.generate_response(
                state.get('judgment', {}),
                state.get('input_text', ''),
                analyses
            )
            
            return {"educational_response": educational_response, "processing_stage": "completed"}
            
        except Exception as e:
            logger.error(f"Educator error: {str(e)}")
            return {
                "educational_response": {"main_message": "Unable to generate educational response due to an error.", "error": str(e)},
                "processing_stage": "completed"
            }
    
    def check_error(self, state: VerificationState) -> str:
        """Check if error occurred"""
        if state.get('error'):
            return "error"
        return "continue"
    
    async def process_message(self, 
                            text: str = None, 
                            image: bytes = None, 
                            audio: bytes = None,
                            sender: str = None,
                            message_sid: str = None) -> Dict[str, Any]:
        """Process a message through the entire pipeline"""
        
        # Initialize state
        initial_state: VerificationState = {
            "input_text": text,
            "input_image": image,
            "input_audio": audio,
            "sender": sender,
            "message_sid": message_sid,
            "processing_stage": "initialized",
            "multimodal_analysis": None,
            "dl_classification": None,
            "clb_classification": None,
            "bharat_analysis": None,
            "novemofake_analysis": None,
            "translation": None,
            "rag_results": None,
            "verification_results": None,
            "judgment": None,
            "educational_response": None,
            "error": None
        }
        
        # Run the graph
        try:
            final_state = await self.graph.ainvoke(initial_state)
            return {
                "success": not final_state.get('error'),
                "judgment": final_state.get('judgment'),
                "response": final_state.get('educational_response', {}).get('main_message'),
                "educational_content": final_state.get('educational_response', {}).get('educational_content'),
                "counter_narrative": final_state.get('educational_response', {}).get('counter_narrative'),
                "verification_tips": final_state.get('educational_response', {}).get('verification_tips'),
                "error": final_state.get('error'),
                "processing_stage": final_state.get('processing_stage')
            }
        except Exception as e:
            logger.error(f"Pipeline error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "response": "Sorry, we encountered an error processing your message."
            }