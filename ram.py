import re
import requests
import csv
import os
from dotenv import load_dotenv

# Download required NLTK data for the smart query extractor
import nltk
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# ================= CONFIG =================
load_dotenv()
GOOGLE_FACT_CHECK_API_KEY = os.getenv("GOOGLE_FACT_CHECK_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")

CSV_FILENAME = "agent_training_data.csv"
WIKI_USER_AGENT = "FactChecker/1.0"

# ================= TEXT EXTRACT =================
def extract_smart_query(text):
    """
    Extracts nouns, proper nouns, verbs, and adjectives to form a search query.
    Keeps the core action of the claim intact to avoid false positives.
    """
    # If the claim is short (under 6 words), just search the whole thing
    words = text.split()
    if len(words) <= 6:
        return text.strip()

    # Otherwise, extract important parts of speech
    tokens = word_tokenize(text)
    tags = pos_tag(tokens)
    
    # Keep Nouns (NN, NNP), Verbs (VB, VBD), and Adjectives (JJ)
    important_words = [w for w, t in tags if t.startswith('NN') or t.startswith('VB') or t.startswith('JJ')]
    
    # Remove extremely common stop words that might have snuck in
    important_words = [w for w in important_words if w.lower() not in ['is', 'are', 'was', 'were', 'be', 'have', 'has']]
    
    return " ".join(important_words[:6])

# ================= AGENTS =================
def wikipedia_check(text):
    """Searches Wikipedia for the entities to verify baseline reality."""
    try:
        query = extract_smart_query(text)
        if not query: return 0.20, 0.5, False
        print(f"   [Wikipedia Searching for:] '{query}'")
        
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query,
            "utf8": 1,
            "srlimit": 3
        }
        headers = {"User-Agent": WIKI_USER_AGENT}
        r = requests.get(url, params=params, headers=headers, timeout=5)
        
        if r.status_code == 200:
            data = r.json()
            results = data.get("query", {}).get("search", [])
            if results:
                return 0.60, 0.7, True
            return 0.30, 0.6, False
        return 0.50, 0.5, False
    except Exception as e:
        print(f"   [!] Wikipedia API Failed: {e}")
        return 0.50, 0.5, False

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
            return 0.20, 0.5, False 
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
                
                prob = min(0.65 + (reputable_count * 0.10), 0.95)
                return prob, 0.8, True
            return 0.20, 0.5, False
        return 0.50, 0.5, False 
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
                
                prob = min(0.65 + (reputable_count * 0.10), 0.95)
                return prob, 0.8, True
            return 0.20, 0.5, False
        return 0.50, 0.5, False 
    except:
        return 0.50, 0.5, False

def openrouter_llm(text):
    """Uses a free model to rate the truthfulness of the claim."""
    try:
        prompt = f"Rate the truthfulness of this claim on a scale from 0.0 (completely false) to 1.0 (completely true). Reply ONLY with a decimal number: '{text[:200]}'"
        
        data = {
            "model": "google/gemini-2.0-flash:free", 
            "messages": [{"role": "user", "content": prompt}], 
            "temperature": 0.0 
        }
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}", 
            "Content-Type": "application/json"
        }
        
        r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=8)
        
        if r.status_code == 200:
            content = r.json()['choices'][0]['message']['content']
            numbers = re.findall(r"0\.\d+|1\.0|1|0", content)
            if numbers:
                return max(0.0, min(1.0, float(numbers[0]))), 0.8, True
            return 0.20, 0.5, False
        else:
            print(f"   [!] LLM API Error: Status {r.status_code} - {r.text}")
            return 0.50, 0.5, False
            
    except Exception as e:
        print(f"   [!] LLM API Failed Exception: {e}")
        return 0.50, 0.5, False

# ================= DATA COLLECTION =================
def save_to_csv(text, wiki_prob, wiki_found, fc_prob, fc_found, news_prob, news_found, gnews_prob, gnews_found, llm_prob, llm_found, final_prob, label):
    file_exists = os.path.isfile(CSV_FILENAME)
    with open(CSV_FILENAME, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["claim_text", "wiki_prob", "wiki_found", "fc_prob", "fc_found", "newsapi_prob", "newsapi_found", "gnews_prob", "gnews_found", "llm_prob", "llm_found", "final_calculated_prob", "predicted_label"])
        writer.writerow([text, wiki_prob, int(wiki_found), fc_prob, int(fc_found), news_prob, int(news_found), gnews_prob, int(gnews_found), llm_prob, int(llm_found), final_prob, label])

# ================= MAIN DETECTOR =================
def detect_news(text):
    print("\n" + "="*60)
    print(f"📰 Claim: {text}")
    print("="*60)
    
    # 1. Wikipedia Agent
    wiki_prob, wiki_conf, wiki_found = wikipedia_check(text)
    print(f"📚 Wikipedia: {wiki_prob:.1%} (found={wiki_found})")
    
    # 2. FactCheck Agent
    fc_prob, fc_conf, fc_found = google_factcheck(text)
    print(f"🔍 FactCheck: {fc_prob:.1%} (found={fc_found})")
    
    # 3. NewsAPI Agent
    news_prob, news_conf, news_found = newsapi_check(text)
    print(f"📰 NewsAPI: {news_prob:.1%} (found={news_found})")
    
    # 4. GNews Agent
    gnews_prob, gnews_conf, gnews_found = gnews_check(text)
    print(f"🗞️ GNews: {gnews_prob:.1%} (found={gnews_found})")
    
    # 5. LLM Agent
    llm_prob, llm_conf, llm_found = openrouter_llm(text)
    print(f"🤖 LLM: {llm_prob:.1%} (found={llm_found})")
    
    evidence_found = wiki_found or fc_found or news_found or gnews_found or llm_found
    
    if not evidence_found:
        print("\n⚠️⚠️⚠️ NO EXTERNAL EVIDENCE FOUND ⚠️⚠️⚠️")
        print("   If it's real news, someone would be reporting it!")
        print("   → FORCING VERDICT TO FAKE")
        final_prob = 0.20  
        label = "❌ FAKE NEWS"
        reasoning = "No evidence found anywhere on the internet"
        
    else:
        # Base weights map to: [wiki, factcheck, newsapi, gnews, llm]
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
            weights = [0.05, 0.10, 0.10, 0.10, 0.65]
            reasoning = "LLM knowledge utilized due to lack of recent news."
            
        else:
            print("\n   [!] Dynamic Route: Using balanced fallback weights.")
            weights = [0.20, 0.20, 0.20, 0.20, 0.20] 
            reasoning = "Fallback balanced routing."
        
        active_weights = weights
        active_probs = [wiki_prob, fc_prob, news_prob, gnews_prob, llm_prob]
        
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
    
    save_to_csv(text, wiki_prob, wiki_found, fc_prob, fc_found, news_prob, news_found, gnews_prob, gnews_found, llm_prob, llm_found, final_prob, label)
    
    return {'label': label, 'probability': final_prob, 'evidence_found': evidence_found, 'reasoning': reasoning}

# ================= RUN =================
if __name__ == "__main__":
    print("="*60)
    print("🧠 DYNAMIC HYBRID NEWS DETECTOR")
    print("   Data Collection: ACTIVE (Saving to CSV)")
    print("="*60)
    
    samples = [
        "kim jong un fires ballistic missiles",
        "usa attack iran",
        "Narendra modi is randwa"
    ]
    
    for i, s in enumerate(samples, 1):
        detect_news(s)
        if i < len(samples):
            input("\nPress Enter for next sample...")