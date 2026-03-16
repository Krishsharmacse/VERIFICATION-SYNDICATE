import torch
import torch.nn as nn
import pickle
import re
import requests
import csv
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from dotenv import load_dotenv

# ================= CONFIG =================
load_dotenv()
GOOGLE_FACT_CHECK_API_KEY = os.getenv("GOOGLE_FACT_CHECK_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 150
CSV_FILENAME = "agent_training_data.csv"

# Download NLTK data
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
    print(f"✅ Vocabulary loaded: {len(vocab)} words")
except FileNotFoundError:
    print("❌ vocab_nltk.pkl not found!")
    exit(1)

VOCAB_SIZE = len(vocab)

# ================= MODEL =================
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

# Load model
model_path = r"C:\Users\ASUS\Desktop\Fake News\fake_news_bilstm_nltk.pth"
try:
    model = ImprovedBiLSTM(VOCAB_SIZE).to(DEVICE)
    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# ================= TEXT CLEAN & EXTRACT =================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    pos_tags = pos_tag(tokens)
    lemmatized = []
    for word, tag in pos_tags:
        if tag.startswith('V'): pos='v'
        elif tag.startswith('J'): pos='a'
        elif tag.startswith('R'): pos='r'
        else: pos='n'
        lemmatized.append(lemmatizer.lemmatize(word, pos=pos))
    return ' '.join(lemmatized)

def text_to_seq(text):
    seq = [vocab.get(w, 1) for w in text.split()]
    if len(seq) < MAX_LEN:
        seq += [0] * (MAX_LEN - len(seq))
    else:
        seq = seq[:MAX_LEN]
    return seq

def extract_smart_query(text):
    tokens = word_tokenize(text)
    tags = pos_tag(tokens)
    
    proper_nouns = [w for w, t in tags if t in ('NNP', 'NNPS') and len(w) > 2]
    nouns = [w for w, t in tags if t in ('NN', 'NNS') and len(w) > 3]
    
    combined = []
    for word in proper_nouns + nouns:
        if word not in combined:
            combined.append(word)
            
    if not combined:
        words = re.sub(r"[^a-zA-Z\s]", "", text).split()
        combined = [w for w in words if len(w) > 4]
        
    return " ".join(combined[:5])

# ================= AGENTS =================
def google_factcheck(text):
    try:
        params = {"query": text[:200], "key": GOOGLE_FACT_CHECK_API_KEY, "languageCode": "en"}
        r = requests.get("https://factchecktools.googleapis.com/v1alpha1/claims:search", params=params, timeout=5)
        if r.status_code == 200:
            data = r.json()
            claims = data.get("claims", [])
            if claims:
                reviews = claims[0].get("claimReview", [])
                if reviews:
                    rating = reviews[0].get("textualRating", "").lower()
                    if "false" in rating or "pants" in rating: return 0.15, 0.9, True
                    elif "true" in rating: return 0.90, 0.9, True
                    elif "mostly true" in rating: return 0.80, 0.8, True
                    elif "mostly false" in rating: return 0.30, 0.8, True
                    elif "mixture" in rating or "half" in rating: return 0.50, 0.7, True
            # Searched successfully, found nothing = Fake
            return 0.20, 0.5, False 
        # Server error = Neutral
        return 0.50, 0.5, False
    except:
        return 0.50, 0.5, False

def newsapi_check(text):
    try:
        query = extract_smart_query(text)
        if not query: return 0.20, 0.5, False
        print(f"   [NewsAPI Searching for:] '{query}'")
        
        params = {"q": query, "apiKey": NEWS_API_KEY, "language": "en", "pageSize": 5, "sortBy": "relevancy"}
        r = requests.get("https://newsapi.org/v2/everything", params=params, timeout=5)
        
        if r.status_code == 200:
            data = r.json()
            articles = data.get("articles", [])
            if articles:
                reputable = ['reuters', 'bbc', 'cnn', 'ap', 'nytimes', 'wsj', 'guardian', 'ndtv']
                reputable_count = sum(1 for a in articles[:5] if any(rs in a.get('source', {}).get('name', '').lower() for rs in reputable))
                
                if reputable_count == 0: return 0.30, 0.7, False
                
                # Boosted formula! 1 reputable source = 75% confident it's real.
                prob = min(0.65 + (reputable_count * 0.10), 0.95)
                return prob, 0.8, True
            return 0.20, 0.5, False
        return 0.50, 0.5, False # Neutral on API error
    except:
        return 0.50, 0.5, False

def gnews_check(text):
    try:
        query = extract_smart_query(text)
        if not query: return 0.20, 0.5, False
        print(f"   [GNews Searching for:] '{query}'")
        
        url = f"https://gnews.io/api/v4/search?q={query}&lang=en&max=5&apikey={GNEWS_API_KEY}"
        r = requests.get(url, timeout=5)
        
        if r.status_code == 200:
            data = r.json()
            articles = data.get("articles", [])
            if articles:
                reputable = ['reuters', 'bbc', 'cnn', 'ap', 'nytimes', 'wsj', 'guardian', 'ndtv', 'hindustan times', 'times of india']
                reputable_count = sum(1 for a in articles if any(rs in a.get('source', {}).get('name', '').lower() for rs in reputable))
                
                if reputable_count == 0: return 0.30, 0.7, False
                
                # Boosted formula! 
                prob = min(0.65 + (reputable_count * 0.10), 0.95)
                return prob, 0.8, True
            return 0.20, 0.5, False
        return 0.50, 0.5, False # Neutral on API error
    except:
        return 0.50, 0.5, False

def openrouter_llm(text):
    try:
        prompt = f"Rate truthfulness (0.0-1.0). Reply ONLY with a decimal number between 0 and 1: {text[:200]}"
        data = {"model": "google/gemini-2.0-flash-lite-preview-02-05:free", "messages": [{"role": "user", "content": prompt}], "temperature": 0.1}
        headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
        r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=8)
        
        if r.status_code == 200:
            content = r.json()['choices'][0]['message']['content']
            numbers = re.findall(r"0\.\d+|1\.0|1|0", content)
            if numbers:
                return max(0.0, min(1.0, float(numbers[0]))), 0.8, True
            return 0.20, 0.5, False
        else:
            print(f"   [!] LLM API Error: Status {r.status_code} (Returning Neutral 50%)")
            return 0.50, 0.5, False
    except Exception as e:
        print(f"   [!] LLM API Failed: {e} (Returning Neutral 50%)")
        return 0.50, 0.5, False

# ================= DATA COLLECTION (FOR FUTURE ML) =================
def save_to_csv(text, model_prob, fc_prob, fc_found, news_prob, news_found, gnews_prob, gnews_found, llm_prob, llm_found, final_prob, label):
    file_exists = os.path.isfile(CSV_FILENAME)
    with open(CSV_FILENAME, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["claim_text", "bilstm_prob", "fc_prob", "fc_found", "newsapi_prob", "newsapi_found", "gnews_prob", "gnews_found", "llm_prob", "llm_found", "final_calculated_prob", "predicted_label"])
        writer.writerow([text, model_prob, fc_prob, int(fc_found), news_prob, int(news_found), gnews_prob, int(gnews_found), llm_prob, int(llm_found), final_prob, label])

# ================= MAIN DETECTOR =================
def detect_news(text):
    print("\n" + "="*60)
    print(f"📰 Claim: {text}")
    print("="*60)
    
    cleaned = clean_text(text)
    seq = text_to_seq(cleaned)
    tensor = torch.tensor(seq).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        raw_prob = torch.sigmoid(model(tensor)).item()
    
    model_prob = raw_prob * 0.80  
    print(f"\n📊 BiLSTM: raw={raw_prob:.1%}, adjusted={model_prob:.1%}")
    
    fc_prob, fc_conf, fc_found = google_factcheck(text)
    print(f"🔍 FactCheck: {fc_prob:.1%} (found={fc_found})")
    
    news_prob, news_conf, news_found = newsapi_check(text)
    print(f"📰 NewsAPI: {news_prob:.1%} (found={news_found})")
    
    gnews_prob, gnews_conf, gnews_found = gnews_check(text)
    print(f"🗞️ GNews: {gnews_prob:.1%} (found={gnews_found})")
    
    llm_prob, llm_conf, llm_found = openrouter_llm(text)
    print(f"🤖 LLM: {llm_prob:.1%} (found={llm_found})")
    
    evidence_found = fc_found or news_found or gnews_found or llm_found
    
    if not evidence_found:
        print("\n⚠️⚠️⚠️ NO EXTERNAL EVIDENCE FOUND ⚠️⚠️⚠️")
        print("   If it's real news, someone would be reporting it!")
        print("   → FORCING VERDICT TO FAKE")
        final_prob = 0.20  
        label = "❌ FAKE NEWS"
        reasoning = "No evidence found anywhere on the internet"
    
    else:
        # Base weights map to: [model, factcheck, newsapi, gnews, llm]
        
        if fc_found:
            print("\n   [!] Dynamic Route: Trusting FactCheck database.")
            weights = [0.05, 0.65, 0.10, 0.10, 0.10]
            reasoning = "FactCheck directly validated/invalidated the claim."
            
        elif gnews_found or news_found:
            if gnews_found and not news_found:
                print("\n   [!] Dynamic Route: GNews found evidence, NewsAPI missed. Trusting GNews heavily.")
                weights = [0.05, 0.10, 0.05, 0.70, 0.10] 
            elif news_found and not gnews_found:
                print("\n   [!] Dynamic Route: NewsAPI found evidence, GNews missed. Trusting NewsAPI heavily.")
                weights = [0.05, 0.10, 0.70, 0.05, 0.10]
            else:
                print("\n   [!] Dynamic Route: Both News APIs found evidence.")
                weights = [0.05, 0.10, 0.35, 0.40, 0.10]
            reasoning = "News search strongly weighted toward successful agents."
            
        elif llm_found and not (gnews_found or news_found):
            print("\n   [!] Dynamic Route: Trusting LLM for general knowledge/myth.")
            weights = [0.10, 0.10, 0.10, 0.10, 0.60]
            reasoning = "LLM knowledge utilized due to lack of recent news."
            
        else:
            print("\n   [!] Dynamic Route: Using balanced fallback weights.")
            weights = [0.05, 0.25, 0.15, 0.40, 0.15] 
            reasoning = "Fallback balanced routing."
        
        active_weights = weights
        active_probs = [model_prob, fc_prob, news_prob, gnews_prob, llm_prob]
        
        total_weight = sum(active_weights)
        norm_weights = [w/total_weight for w in active_weights]
        weighted_sum = sum(p * w for p, w in zip(active_probs, norm_weights))
        
        final_prob = weighted_sum
        label = "✅ REAL NEWS" if final_prob >= 0.5 else "❌ FAKE NEWS"
    
    print(f"\n{'='*60}")
    print(f"🎯 FINAL: {label}")
    print(f"   Probability: {final_prob:.1%} real, {1-final_prob:.1%} fake")
    print(f"   Reasoning: {reasoning}")
    print(f"   Evidence found: {'✓' if evidence_found else '✗ NONE'}")
    print(f"{'='*60}")
    
    save_to_csv(text, model_prob, fc_prob, fc_found, news_prob, news_found, gnews_prob, gnews_found, llm_prob, llm_found, final_prob, label)
    
    return {'label': label, 'probability': final_prob, 'evidence_found': evidence_found, 'reasoning': reasoning}

# ================= RUN =================
if __name__ == "__main__":
    print("="*60)
    print("🧠 DYNAMIC HYBRID NEWS DETECTOR")
    print("   Data Collection: ACTIVE (Saving to CSV)")
    print("="*60)
    
    samples = [
        """kim jong un fires ballistic missiles""",
        """The 2023 World Press Freedom Index, released annually by Reporters Without Borders (RSF), highlights the increasingly perilous situation for reporters on every continent, as journalists face political, social, and technological threats. The report was released on World Press Freedom Day, which GIJN marked with a specific look at how investigative journalists are confronted with challenges to their work in Latin America, Africa, Central Asia, and the Middle East.""",
    ]
    
    for i, s in enumerate(samples, 1):
        detect_news(s)
        if i < len(samples):
            input("\nPress Enter for next sample...")