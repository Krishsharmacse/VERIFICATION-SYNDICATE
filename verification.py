import torch
import torch.nn as nn
import pickle
import re
import requests
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from datetime import datetime
import time
from gnews import GNews

# ================= CONFIGURATION =================
class Config:
    # ⚠️ YOUR API KEYS (Keep these secret!) ⚠️
    GOOGLE_FACT_CHECK_API_KEY = "YOUR_GOOGLE_KEY"
    NEWS_API_KEY = "YOUR_NEWSAPI_KEY"
    OPENROUTER_API_KEY = "YOUR_OPENROUTER_KEY"
    
    # Weights for final decision (Total = 1.0)
    MODEL_WEIGHT = 0.20        # Your BiLSTM model
    FACTCHECK_WEIGHT = 0.35    # Google Fact Check (highest authority)
    NEWS_WEIGHT = 0.15         # NewsAPI.org
    GNEWS_WEIGHT = 0.15        # Google News
    LLM_WEIGHT = 0.15          # OpenRouter

config = Config()

# ================= DEVICE =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ================= NLTK SETUP =================
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ================= LOAD VOCAB =================
try:
    with open("vocab_nltk.pkl", "rb") as f:
        vocab = pickle.load(f)
    print("✅ Vocabulary loaded")
except:
    print("❌ vocab_nltk.pkl not found! Ensure it is in the same directory.")
    exit()

MAX_LEN = 150
VOCAB_SIZE = len(vocab)

# ================= MODEL DEFINITION =================
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

# ================= LOAD MODEL =================
try:
    model = ImprovedBiLSTM(VOCAB_SIZE).to(device)
    model.load_state_dict(torch.load("fake_news_bilstm_nltk.pth", map_location=device))
    model.eval()
    print("✅ Model loaded successfully!")
except:
    print("❌ fake_news_bilstm_nltk.pth not found!")
    exit()

# ================= TEXT CLEANING =================
source_words = set([
    'reuters', 'cnn', 'fox', 'breitbart', 'nytimes', 'washingtonpost',
    'bbc', 'abc', 'nbc', 'cbs', 'ap', 'associated', 'press', 'apnews',
    'guardian', 'wsj', 'usatoday', 'huffington', 'huffpost', 'buzzfeed',
])

def advanced_nltk_clean(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and t not in source_words]

    pos_tags = pos_tag(tokens)
    lemmatized_tokens = []
    for word, tag in pos_tags:
        if tag.startswith('V'): pos = 'v'
        elif tag.startswith('J'): pos = 'a'
        elif tag.startswith('R'): pos = 'r'
        else: pos = 'n'
        lemmatized_tokens.append(lemmatizer.lemmatize(word, pos=pos))

    return ' '.join(lemmatized_tokens)

def text_to_seq(text):
    tokens = text.split()
    seq = [vocab.get(w, 1) for w in tokens]  # 1 = <UNK>
    if len(seq) < MAX_LEN:
        seq += [0] * (MAX_LEN - len(seq))
    else:
        seq = seq[:MAX_LEN]
    return seq

# ================= AGENT 1: GOOGLE FACT CHECK =================
class GoogleFactCheckAgent:
    def __init__(self):
        self.api_key = config.GOOGLE_FACT_CHECK_API_KEY
        self.base_url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    
    def check(self, text):
        try:
            params = {"query": text[:200], "key": self.api_key, "languageCode": "en"}
            response = requests.get(self.base_url, params=params, timeout=5)
            if response.status_code == 200:
                claims = response.json().get("claims", [])
                if claims and claims[0].get("claimReview"):
                    review = claims[0]["claimReview"][0]
                    rating = review.get("textualRating", "").lower()
                    publisher = review.get("publisher", {}).get("name", "Unknown")
                    
                    if "false" in rating or "pants" in rating: return (0.10, 0.9, publisher)
                    elif "true" in rating: return (0.90, 0.9, publisher)
                    elif "mostly true" in rating: return (0.80, 0.8, publisher)
                    elif "mostly false" in rating: return (0.30, 0.8, publisher)
                    elif "mixture" in rating or "half" in rating: return (0.50, 0.7, publisher)
            return (None, 0.0, None)
        except: return (None, 0.0, None)

# ================= AGENT 2: NEWS API =================
class NewsAPIAgent:
    def __init__(self):
        self.api_key = config.NEWS_API_KEY
        self.base_url = "https://newsapi.org/v2/everything"
    
    def check(self, text):
        try:
            words = text.split()[:5]
            query = " ".join([w for w in words if len(w) > 3][:3])
            params = {"q": query, "apiKey": self.api_key, "language": "en", "pageSize": 5, "sortBy": "relevancy"}
            response = requests.get(self.base_url, params=params, timeout=5)
            
            if response.status_code == 200:
                articles = response.json().get("articles", [])
                if articles:
                    reputable_sources = ['reuters', 'ap', 'bbc', 'cnn', 'nytimes']
                    source_score = 0
                    
                    for a in articles[:5]:
                        title = a.get('title', '').lower()
                        source = a.get('source', {}).get('name', '').lower()
                        
                        # RED FLAG: If the news is actually a fact-check debunking the claim
                        if any(bad_word in title for bad_word in ['fact check', 'hoax', 'debunk', 'false']):
                            return (0.05, 0.9, len(articles), True) # 5% true = Fake News
                            
                        if any(rs in source for rs in reputable_sources):
                            source_score += 0.2
                            
                    # Cap the heuristic probability so it doesn't blindly trust article counts
                    prob = min(0.75, max(0.25, 0.5 + source_score))
                    return (prob, 0.6, len(articles), False)
            return (None, 0.0, 0, False)
        except: return (None, 0.0, 0, False)

# ================= AGENT 3: GNEWS API =================
class GNewsAgent:
    def __init__(self):
        self.google_news = GNews(language='en', country='US', max_results=5)
        
    def check(self, text):
        try:
            words = text.split()
            query = " ".join([w for w in words if len(w) > 3][:4])
            if not query: query = text[:30]
                
            news = self.google_news.get_news(query)
            
            if news:
                reputable_sources = ['reuters', 'associated press', 'bbc', 'cnn', 'new york times', 'guardian', 'washington post']
                source_score = 0
                
                for article in news[:5]:
                    title = article.get('title', '').lower()
                    source = article.get('publisher', {}).get('title', '').lower()
                    
                    # RED FLAG: If the search results are fact-checkers debunking it
                    if any(bad_word in title for bad_word in ['fact check', 'hoax', 'debunk', 'false', 'conspiracy']):
                        return (0.05, 0.95, len(news), True) 
                        
                    if any(rs in source for rs in reputable_sources):
                        source_score += 0.2
                
                # Cap heuristic probability
                prob = min(0.75, max(0.25, 0.5 + source_score))
                return (prob, 0.65, len(news), False)
                
            return (None, 0.0, 0, False)
        except Exception as e:
            print(f"   ⚠️ GNews error: {e}")
            return (None, 0.0, 0, False)

# ================= AGENT 4: OPENROUTER LLM =================
class OpenRouterAgent:
    def __init__(self):
        self.api_key = config.OPENROUTER_API_KEY
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
    
    def verify(self, text):
        prompt = f"""You are a fact-checker. Analyze this claim: "{text}"
Based on your knowledge, rate the truthfulness from 0.0 (completely false) to 1.0 (completely true).
Respond with ONLY a number between 0 and 1."""
        try:
            models = [
                "google/gemini-2.0-flash-lite-preview-02-05:free", 
                "meta-llama/llama-3.2-3b-instruct:free", 
                "nvidia/nemotron-nano-12b-v2-vl:free"
            ]
            for model in models:
                try:
                    data = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.1, "max_tokens": 10}
                    response = requests.post(self.base_url, headers=self.headers, json=data, timeout=10)
                    if response.status_code == 200:
                        content = response.json()['choices'][0]['message']['content'].strip()
                        # Extract the float or int from the response
                        numbers = re.findall(r"0\.\d+|\d\.\d+|0|1", content)
                        if numbers: return (max(0.0, min(1.0, float(numbers[0]))), 0.8)
                except: continue
            return (None, 0.0)
        except: return (None, 0.0)

# ================= MAIN HYBRID DETECTOR =================
class HybridNewsDetector:
    def __init__(self):
        self.factcheck = GoogleFactCheckAgent()
        self.news = NewsAPIAgent()
        self.gnews = GNewsAgent()
        self.llm = OpenRouterAgent()
        
        self.weights = {
            'model': config.MODEL_WEIGHT,
            'factcheck': config.FACTCHECK_WEIGHT,
            'news': config.NEWS_WEIGHT,
            'gnews': config.GNEWS_WEIGHT,
            'llm': config.LLM_WEIGHT
        }
    
    def predict(self, text, threshold=0.6):
        print("\n" + "="*70)
        print(f"📰 Analyzing: {text[:100]}...")
        print("="*70)
        
        results = {}
        total_weight = 0
        weighted_sum = 0
        
        # STEP 1: BiLSTM Model
        print("\n📊 1. BiLSTM Model...")
        cleaned = advanced_nltk_clean(text)
        seq = text_to_seq(cleaned)
        tensor = torch.tensor(seq).unsqueeze(0).to(device)
        with torch.no_grad():
            model_prob = torch.sigmoid(model(tensor)).item()
        results['model'] = {'probability': model_prob, 'confidence': model_prob if model_prob >= 0.5 else 1-model_prob, 'weight': self.weights['model']}
        print(f"   → Real: {model_prob:.1%}, Fake: {1-model_prob:.1%}")
        weighted_sum += model_prob * self.weights['model']
        total_weight += self.weights['model']
        
        # STEP 2: Google Fact Check
        print("\n🔍 2. Google Fact Check...")
        fc_prob, fc_conf, fc_source = self.factcheck.check(text)
        if fc_prob is not None:
            results['factcheck'] = {'probability': fc_prob, 'confidence': fc_conf, 'source': fc_source, 'weight': self.weights['factcheck']}
            print(f"   → Found: {fc_source} rates it {fc_prob:.1%} true")
            weighted_sum += fc_prob * self.weights['factcheck']
            total_weight += self.weights['factcheck']
        else:
            print("   → No fact-checks found")
        
        # STEP 3: News API
        print("\n📰 3. NewsAPI Search...")
        news_prob, news_conf, news_count, news_debunk = self.news.check(text)
        if news_prob is not None:
            results['news'] = {'probability': news_prob, 'confidence': news_conf, 'article_count': news_count, 'weight': self.weights['news']}
            if news_debunk:
                print(f"   🚨 DEBUNK DETECTED: Articles found are fact-checks refuting this claim.")
            else:
                print(f"   → Found {news_count} articles, suggests {news_prob:.1%} true")
            weighted_sum += news_prob * self.weights['news']
            total_weight += self.weights['news']
        else:
            print("   → No news articles found via NewsAPI")
            
        # STEP 4: GNews API
        print("\n🌍 4. Google News (GNews) Search...")
        gnews_prob, gnews_conf, gnews_count, gnews_debunk = self.gnews.check(text)
        if gnews_prob is not None:
            results['gnews'] = {'probability': gnews_prob, 'confidence': gnews_conf, 'article_count': gnews_count, 'weight': self.weights['gnews']}
            if gnews_debunk:
                print(f"   🚨 DEBUNK DETECTED: Google News articles are refuting this claim.")
            else:
                print(f"   → Found {gnews_count} articles, suggests {gnews_prob:.1%} true")
            weighted_sum += gnews_prob * self.weights['gnews']
            total_weight += self.weights['gnews']
        else:
            print("   → No news articles found via Google News")
        
        # STEP 5: OpenRouter LLM
        print("\n🤖 5. OpenRouter LLM...")
        llm_prob, llm_conf = self.llm.verify(text)
        if llm_prob is not None:
            results['llm'] = {'probability': llm_prob, 'confidence': llm_conf, 'weight': self.weights['llm']}
            print(f"   → LLM rates it {llm_prob:.1%} true")
            weighted_sum += llm_prob * self.weights['llm']
            total_weight += self.weights['llm']
        else:
            print("   → LLM unavailable")
        
        # FINAL: Weighted Decision
        final_prob = (weighted_sum / total_weight) if total_weight > 0 else 0.5
        final_label = "REAL NEWS" if final_prob >= 0.5 else "FAKE NEWS"
        final_conf = final_prob if final_prob >= 0.5 else 1 - final_prob
        
        results['final'] = {
            'probability': final_prob,
            'label': final_label,
            'confidence': final_conf,
            'is_certain': final_conf >= threshold,
            'weights_used': total_weight
        }
        return results
    
    def print_result(self, results):
        f = results['final']
        print("\n" + "="*70)
        print(f"{'✅✅✅' if f['label'] == 'REAL NEWS' else '❌❌❌'} FINAL VERDICT: {f['label']}")
        print(f"   Truth Score: {f['probability']:.1%} real, {1-f['probability']:.1%} fake")
        print(f"   Confidence: {f['confidence']:.1%} ({'✓ HIGH' if f['is_certain'] else '⚠️ LOW'})")
        
        print("\n📊 Source Breakdown:")
        if 'model' in results: print(f"   • BiLSTM Model: {results['model']['probability']:.1%} (weight: {results['model']['weight']:.2f})")
        if 'factcheck' in results: print(f"   • Google Fact Check: {results['factcheck']['probability']:.1%} by {results['factcheck']['source']} (weight: {results['factcheck']['weight']:.2f})")
        if 'news' in results: print(f"   • NewsAPI: {results['news']['probability']:.1%} (weight: {results['news']['weight']:.2f})")
        if 'gnews' in results: print(f"   • Google News: {results['gnews']['probability']:.1%} (weight: {results['gnews']['weight']:.2f})")
        if 'llm' in results: print(f"   • OpenRouter LLM: {results['llm']['probability']:.1%} (weight: {results['llm']['weight']:.2f})")
        print("="*70)

# ================= MAIN EXECUTION =================
if __name__ == "__main__":
    print("="*70)
    print("🧠 HYBRID NEWS DETECTOR")
    print("   BiLSTM + Fact Check + NewsAPI + GNews + OpenRouter")
    print("="*70)
    
    detector = HybridNewsDetector()
    
    test_samples = [
        "Scientists discovered water on Mars in new NASA study",
        "Aliens met the president yesterday in secret White House meeting",
        "Government passes new economic bill to help small businesses",
        "FDA approves new cancer treatment with 90% success rate",
        "BREAKING: SHOCKING video exposes government cover-up"
    ]
    
    for i, text in enumerate(test_samples, 1):
        results = detector.predict(text)
        detector.print_result(results)
        if i < len(test_samples): input("\nPress Enter for next sample...")