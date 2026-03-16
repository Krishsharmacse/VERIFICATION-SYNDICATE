import os
import json
import re
import operator
from typing import List, Dict, Any, TypedDict, Annotated
from datetime import datetime
import pickle

import torch
import torch.nn as nn
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

import google.generativeai as genai
from langgraph.graph import StateGraph, END
from gnews import GNews
from dotenv import load_dotenv

# Optional imports with graceful degradation
try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False

try:
    import wikipediaapi
    WIKI_AVAILABLE = True
except ImportError:
    WIKI_AVAILABLE = False

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# ==================== LOAD ENVIRONMENT VARIABLES ====================
load_dotenv() # This loads the keys from your .env file

class Config:
    # Fetch keys securely from the .env file
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    WIKI_USER_AGENT = "FactChecker/1.0 (contact@example.com)"
    MAX_SEARCH_RESULTS = 3
    MAX_CLAIMS = 3

config = Config()

if not config.GEMINI_API_KEY:
    print("❌ ERROR: GEMINI_API_KEY not found! Please check your .env file.")
    exit(1)

# ==================== NLTK SETUP ====================
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)

# ==================== GEMINI SETUP ====================
genai.configure(api_key=config.GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.5-pro')

# ==================== DEVICE SETUP ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== BILSTM ARCHITECTURE ====================
class ImprovedBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_layers=2, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = self.embedding_dropout(embedded)
        lstm_out, _ = self.lstm(embedded)
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        attended = torch.sum(attention_weights * lstm_out, dim=1)
        output = self.classifier(attended)
        return output.squeeze()

# Load Vocab and Model
try:
    with open("vocab_nltk.pkl", "rb") as f:
        vocab = pickle.load(f)
    VOCAB_SIZE = len(vocab)
    bilstm_model = ImprovedBiLSTM(VOCAB_SIZE).to(device)
    bilstm_model.load_state_dict(torch.load("fake_news_bilstm_nltk.pth", map_location=device))
    bilstm_model.eval()
    print("✅ BiLSTM Model loaded successfully!")
except Exception as e:
    print(f"⚠️ BiLSTM Model failed to load: {e}")
    bilstm_model = None

# NLP Cleaners
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def advanced_nltk_clean(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    pos_tags = pos_tag(tokens)
    lemmatized_tokens = []
    for word, tag in pos_tags:
        if tag.startswith('V'): pos = 'v'
        elif tag.startswith('J'): pos = 'a'
        elif tag.startswith('R'): pos = 'r'
        else: pos = 'n'
        lemmatized_tokens.append(lemmatizer.lemmatize(word, pos=pos))
    return ' '.join(lemmatized_tokens)

def text_to_seq(text, max_len=150):
    tokens = text.split()
    seq = [vocab.get(w, 1) for w in tokens]
    if len(seq) < max_len:
        seq += [0] * (max_len - len(seq))
    else:
        seq = seq[:max_len]
    return seq

# ==================== LANGGRAPH STATE ====================

# FIXED REDUCER: Safely merges dictionaries from parallel agents
def merge_dicts(dict1: dict, dict2: dict) -> dict:
    if not dict1: dict1 = {}
    if not dict2: dict2 = {}
    return {**dict1, **dict2}

class FactCheckState(TypedDict):
    # Standard fields automatically overwrite in LangGraph (No operator needed)
    article_text: str
    bilstm_analysis: dict
    claims: list
    final_verdict: dict
    
    # Evidence dictionaries updated concurrently by different agents (Needs custom reducer)
    wikipedia_results: Annotated[Dict[int, Dict], merge_dicts]
    web_search_results: Annotated[Dict[int, List[Dict]], merge_dicts]
    gnews_results: Annotated[Dict[int, List[Dict]], merge_dicts]
    gemini_analysis: Annotated[Dict[int, Dict], merge_dicts]
    
    # Lists that agents can append to (Needs operator.add)
    verification_results: Annotated[List[Dict[str, Any]], operator.add]
    errors: Annotated[List[str], operator.add]

# ==================== AGENT NODES ====================

def bilstm_node(state: FactCheckState):
    print("🤖 AGENT 0: Running BiLSTM Structural Analysis...")
    if not bilstm_model:
        return {"bilstm_analysis": {"error": "Model not loaded"}}
    
    cleaned = advanced_nltk_clean(state["article_text"])
    seq = text_to_seq(cleaned)
    tensor = torch.tensor(seq).unsqueeze(0).to(device)
    
    with torch.no_grad():
        out = bilstm_model(tensor)
        prob = torch.sigmoid(out).item()
    
    label = "Real" if prob >= 0.5 else "Fake"
    confidence = prob if prob >= 0.5 else 1 - prob
    
    analysis = {
        "structural_label": label,
        "structural_confidence": confidence,
        "raw_prob": prob
    }
    return {"bilstm_analysis": analysis}

def claim_extractor_node(state: FactCheckState):
    print("🔍 AGENT 1: Extracting claims...")
    sentences = sent_tokenize(state['article_text'])
    claims = []
    
    for i, sent in enumerate(sentences[:config.MAX_CLAIMS]):
        if len(sent.split()) > 4:
            claims.append({
                "claim_text": sent,
                "importance": "high" if i == 0 else "medium"
            })
            
    return {"claims": claims}

def wikipedia_node(state: FactCheckState):
    print("📚 AGENT 2: Checking Wikipedia...")
    if not WIKI_AVAILABLE: return {"wikipedia_results": {}}
    
    wiki = wikipediaapi.Wikipedia(language='en', extract_format=wikipediaapi.ExtractFormat.WIKI, user_agent=config.WIKI_USER_AGENT)
    results_dict = {}
    
    for i, claim in enumerate(state["claims"]):
        words = claim["claim_text"].split()
        key_terms = [w for w in words if len(w) > 4 and w[0].isupper()][:2]
        if not key_terms: key_terms = words[:2]
        
        for term in key_terms:
            page = wiki.page(term)
            if page.exists():
                results_dict[i] = {"title": page.title, "summary": page.summary[:500]}
                break
    return {"wikipedia_results": results_dict}

def gnews_node(state: FactCheckState):
    print("📰 AGENT 3: Checking Google News...")
    google_news = GNews(language='en', period='30d', max_results=2)
    results_dict = {}
    
    for i, claim in enumerate(state["claims"]):
        try:
            query = " ".join(claim["claim_text"].split()[:6]) 
            news = google_news.get_news(query)
            if news:
                results_dict[i] = [{"title": n['title'], "publisher": n.get('publisher',{}).get('title', 'Unknown')} for n in news]
        except Exception as e:
            pass
            
    return {"gnews_results": results_dict}

def duckduckgo_node(state: FactCheckState):
    print("🌐 AGENT 4: Searching DuckDuckGo...")
    if not DDGS_AVAILABLE: return {"web_search_results": {}}
    
    results_dict = {}
    try:
        with DDGS() as ddgs:
            for i, claim in enumerate(state["claims"]):
                query = " ".join(claim["claim_text"].split()[:8])
                results = list(ddgs.text(query, max_results=2))
                if results:
                    results_dict[i] = [{"title": r['title'], "snippet": r['body'][:200]} for r in results]
    except Exception as e:
        return {"errors": [f"DDG Error: {e}"]}
        
    return {"web_search_results": results_dict}

def gemini_analyzer_node(state: FactCheckState):
    print("🧠 AGENT 5: Gemini synthesizing evidence...")
    results_dict = {}
    
    bilstm_data = state.get("bilstm_analysis", {})
    b_label = bilstm_data.get("structural_label", "Unknown")
    b_conf = bilstm_data.get("structural_confidence", 0.0)
    
    for i, claim in enumerate(state["claims"]):
        claim_text = claim["claim_text"]
        
        evidence = []
        if i in state.get("wikipedia_results", {}):
            evidence.append(f"WIKI: {state['wikipedia_results'][i]['title']} - {state['wikipedia_results'][i]['summary']}")
        if i in state.get("gnews_results", {}):
            for n in state["gnews_results"][i]:
                evidence.append(f"NEWS: {n['publisher']} reported '{n['title']}'")
        if i in state.get("web_search_results", {}):
            for w in state["web_search_results"][i]:
                evidence.append(f"WEB: {w['title']} - {w['snippet']}")
                
        evidence_text = "\n".join(evidence) if evidence else "No external evidence found."
        
        prompt = f"""
        You are an elite fact-checker. Analyze this claim.
        
        CLAIM: "{claim_text}"
        
        AI STRUCTURAL ANALYSIS (BiLSTM Model):
        The text structure was flagged as: {b_label} (Confidence: {b_conf:.2f})
        
        EXTERNAL EVIDENCE:
        {evidence_text}
        
        Based on the structural analysis AND the external evidence, provide a final verdict.
        Return ONLY valid JSON:
        {{
            "verdict": "TRUE/FALSE/MIXED/UNVERIFIABLE",
            "confidence": 0.0-1.0,
            "explanation": "Why you chose this verdict"
        }}
        """
        
        try:
            resp = gemini_model.generate_content(prompt).text
            json_match = re.search(r'\{.*\}', resp, re.DOTALL)
            results_dict[i] = json.loads(json_match.group()) if json_match else {"verdict": "UNVERIFIABLE", "explanation": "JSON Parse Error"}
        except Exception as e:
            results_dict[i] = {"verdict": "UNVERIFIABLE", "explanation": str(e)}
            
    return {"gemini_analysis": results_dict}

def judge_node(state: FactCheckState):
    print("👨‍⚖️ AGENT 6: Final Judgment...")
    
    claims_out = []
    total_score = 0
    
    for i, claim in enumerate(state["claims"]):
        analysis = state["gemini_analysis"].get(i, {})
        v = analysis.get("verdict", "UNVERIFIABLE")
        
        if v == "TRUE": total_score += 100
        elif v == "MIXED": total_score += 50
        
        claims_out.append({
            "claim": claim["claim_text"],
            "verdict": v,
            "explanation": analysis.get("explanation", "")
        })
        
    truth_score = total_score // len(state["claims"]) if state["claims"] else 0
    
    final = {
        "overall_verdict": "TRUE" if truth_score > 75 else "FALSE" if truth_score < 25 else "MIXED",
        "truth_score": truth_score,
        "bilstm_structural_flag": state.get("bilstm_analysis", {}).get("structural_label", "Unknown")
    }
    
    return {"verification_results": claims_out, "final_verdict": final}

# ==================== BUILD GRAPH ====================
def build_graph():
    workflow = StateGraph(FactCheckState)
    
    workflow.add_node("bilstm", bilstm_node)
    workflow.add_node("extract", claim_extractor_node)
    workflow.add_node("wiki", wikipedia_node)
    workflow.add_node("gnews", gnews_node)
    workflow.add_node("ddg", duckduckgo_node)
    workflow.add_node("gemini", gemini_analyzer_node)
    workflow.add_node("judge", judge_node)
    
    workflow.set_entry_point("bilstm")
    workflow.add_edge("bilstm", "extract")
    
    # FAN OUT
    workflow.add_edge("extract", "wiki")
    workflow.add_edge("extract", "gnews")
    workflow.add_edge("extract", "ddg")
    
    # FAN IN
    workflow.add_edge("wiki", "gemini")
    workflow.add_edge("gnews", "gemini")
    workflow.add_edge("ddg", "gemini")
    
    workflow.add_edge("gemini", "judge")
    workflow.add_edge("judge", END)
    
    return workflow.compile()

fact_check_app = build_graph()

# ==================== FASTAPI APP ====================
app = FastAPI(title="India AI Fact Checker API")

class NewsRequest(BaseModel):
    text: str

@app.post("/verify")
async def verify_news(request: NewsRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
        
    initial_state = {
        "article_text": request.text,
        "bilstm_analysis": {},
        "claims": [],
        "wikipedia_results": {},
        "web_search_results": {},
        "gnews_results": {},
        "gemini_analysis": {},
        "verification_results": [],
        "final_verdict": {},
        "errors": []
    }
    
    try:
        # Run LangGraph
        final_state = fact_check_app.invoke(initial_state)
        
        return {
            "success": True,
            "bilstm_structural_analysis": final_state["bilstm_analysis"],
            "agent_verdict": final_state["final_verdict"],
            "detailed_claims": final_state["verification_results"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("="*60)
    print("🚀 Starting FastAPI Server for Fact Checker")
    print("="*60)
    # Run the API on localhost port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)