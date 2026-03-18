import os
import io
import re
import csv
import pickle
import requests
import torch
import torch.nn as nn
import numpy as np
import tensorflow as tf
import librosa
import collections
import threading
import queue
import time
import asyncio
import cv2
import shutil
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from transformers import Wav2Vec2FeatureExtractor, AutoModel
import nltk
import uvicorn
from sarvamai import SarvamAI
from langdetect import detect, LangDetectException

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

load_dotenv()
GOOGLE_FACT_CHECK_API_KEY = os.getenv("GOOGLE_FACT_CHECK_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")
SARVAM_API_KEY = os.getenv("Sarvam_API_LANGUAGE") or os.getenv("SARVAM_API_KEY")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 150
CSV_FILENAME = "agent_training_data.csv"

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

if SARVAM_API_KEY:
    sarvam_client = SarvamAI(api_subscription_key=SARVAM_API_KEY)
else:
    sarvam_client = None
    print("⚠️ SARVAM_API_KEY not found. Translation will be disabled.")




def translate_to_english(text):
    if not sarvam_client:
        return text
    try:
        response = sarvam_client.text.translate(
            input=text,
            source_language_code="auto",
            target_language_code="en-IN", 
            speaker_gender="Male"
        )
        if hasattr(response, 'translated_text'):
            return response.translated_text
        elif isinstance(response, dict) and 'translated_text' in response:
            return response['translated_text']
        else:
            return str(response)
    except Exception as e:
        print(f"❌ Translation failed: {e}")
        return text

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
            print(f"   [!] LLM API Error: Status {r.status_code}")
            return 0.50, 0.5, False
    except Exception as e:
        print(f"   [!] LLM API Failed: {e}")
        return 0.50, 0.5, False

def save_to_csv(text, fc_prob, fc_found, news_prob, news_found, gnews_prob, gnews_found, llm_prob, llm_found, final_prob, label):
    file_exists = os.path.isfile(CSV_FILENAME)
    with open(CSV_FILENAME, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["claim_text", "fc_prob", "fc_found", "newsapi_prob", "newsapi_found", "gnews_prob", "gnews_found", "llm_prob", "llm_found", "final_calculated_prob", "predicted_label"])
        writer.writerow([text, fc_prob, int(fc_found), news_prob, int(news_found), gnews_prob, int(gnews_found), llm_prob, int(llm_found), final_prob, label])

def detect_news(text):
    print("\n" + "="*60)
    print(f"📰 Claim: {text}")
    print("="*60)
    
    try:
        detected_lang = detect(text)
        if detected_lang != 'en':
            print(f"🌐 Non-English text detected ({detected_lang}). Translating to English via Sarvam AI...")
            text = translate_to_english(text)
            print(f"🗣️ Translated Claim: {text}")
    except LangDetectException:
        print("⚠️ Could not detect language reliably. Proceeding with original text.")
        
    cleaned = clean_text(text)

    
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
        print("   → FORCING VERDICT TO FAKE")
        final_prob = 0.20  
        label = "❌ FAKE NEWS"
        reasoning = "No evidence found anywhere on the internet"
    
    else:
        if fc_found:
            print("\n   [!] Dynamic Route: Trusting FactCheck database.")
            weights = [0.65, 0.15, 0.10, 0.10]
            reasoning = "FactCheck directly validated/invalidated the claim."
            
        elif gnews_found or news_found:
            if gnews_found and not news_found:
                print("\n   [!] Dynamic Route: GNews found evidence, NewsAPI missed. Trusting GNews heavily.")
                weights = [0.10, 0.05, 0.75, 0.10] 
            elif news_found and not gnews_found:
                print("\n   [!] Dynamic Route: NewsAPI found evidence, GNews missed. Trusting NewsAPI heavily.")
                weights = [0.10, 0.75, 0.05, 0.10]
            else:
                print("\n   [!] Dynamic Route: Both News APIs found evidence.")
                weights = [0.10, 0.40, 0.40, 0.10]
            reasoning = "News search strongly weighted toward successful agents."
            
        elif llm_found and not (gnews_found or news_found):
            print("\n   [!] Dynamic Route: Trusting LLM for general knowledge/myth.")
            weights = [0.10, 0.15, 0.15, 0.60]
            reasoning = "LLM knowledge utilized due to lack of recent news."
            
        else:
            print("\n   [!] Dynamic Route: Using balanced fallback weights.")
            weights = [0.25, 0.25, 0.25, 0.25] 
            reasoning = "Fallback balanced routing."
        
        active_weights = weights
        active_probs = [fc_prob, news_prob, gnews_prob, llm_prob]
        
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
    save_to_csv(text, fc_prob, fc_found, news_prob, news_found, gnews_prob, gnews_found, llm_prob, llm_found, final_prob, label)
    
    return {
        'label': label,
        'probability_real': final_prob,
        'probability_fake': 1 - final_prob,
        'evidence_found': evidence_found,
        'reasoning': reasoning,
        'details': {
            'factcheck': fc_prob,
            'newsapi': news_prob,
            'gnews': gnews_prob,
            'llm': llm_prob
        }
    }

KERAS_MODEL_PATH = r"D:\Verification Syndicate\Fake News\ALL MODELS\wavlm_classifier_v2.keras"
WAVLM_MODEL_NAME = "microsoft/wavlm-base-plus"
SAMPLE_RATE = 16000
MAX_DURATION = 5

print("⏳ Loading Audio Models...")
try:
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(WAVLM_MODEL_NAME)
    wavlm_model = AutoModel.from_pretrained(WAVLM_MODEL_NAME).to(DEVICE)
    wavlm_model.eval()
    
    keras_model = tf.keras.models.load_model(KERAS_MODEL_PATH)
    print("✅ Audio models loaded successfully")
except Exception as e:
    print(f"❌ Error loading audio models: {e}")
    exit(1)

def predict_audio(file_bytes):
    try:

        audio_stream = io.BytesIO(file_bytes)
        y, sr = librosa.load(audio_stream, sr=SAMPLE_RATE)
        max_len = SAMPLE_RATE * MAX_DURATION
        
        if len(y) < max_len:
            y = np.pad(y, (0, max_len - len(y)), 'constant')
        else:
            y = y[:max_len]
        
        inputs = feature_extractor(y, sampling_rate=SAMPLE_RATE, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            outputs = wavlm_model(**inputs)
        
        hidden_states = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
        embedding = torch.mean(hidden_states, dim=1).cpu().numpy()
      
        score = keras_model.predict(embedding, verbose=0)[0][0]
        
        is_fake = score > 0.5
        confidence = score if is_fake else 1 - score
        
        return {
            'label': 'FAKE / SYNTHETIC' if is_fake else 'REAL HUMAN VOICE',
            'score': float(score),
            'confidence': float(confidence)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

VIDEO_MODEL_PATH = r"D:\Verification Syndicate\Fake News\ALL MODELS\best_celebdf_model_Krish.pt"
VIDEO_INPUT_SIZE = 224
VIDEO_SEQ_LENGTH = 16        
VIDEO_CONFIDENCE_THRESHOLD = 0.60
VIDEO_EMA_ALPHA = 0.15       

class CNN_BiLSTM_Video(nn.Module):
    def __init__(self):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        backbone = models.efficientnet_b0(weights=weights)
        self.cnn = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.lstm = nn.LSTM(
            input_size=1280, hidden_size=256, num_layers=2, 
            bidirectional=True, batch_first=True, dropout=0.3
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 2)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        feats = self.cnn(x)
        feats = self.pool(feats).flatten(1)
        feats = feats.view(B, T, -1)
        lstm_out, _ = self.lstm(feats)
        return self.classifier(lstm_out[:, -1, :])

class DeepfakeVideoDetector:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.face_model = YOLO(r"D:\Verification Syndicate\Fake News\yolov8s-face-lindevs.onnx", task='detect') 
        self.model = CNN_BiLSTM_Video().to(self.device)
        self.model.load_state_dict(torch.load(VIDEO_MODEL_PATH, map_location=self.device))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((VIDEO_INPUT_SIZE, VIDEO_INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.frame_buffer = collections.deque(maxlen=VIDEO_SEQ_LENGTH)
        self.running = True
        self.input_queue = queue.Queue(maxsize=1)
        
        self.current_prob = 0.0
        self.smoothed_prob = 0.0
        self.current_label = "Scanning..."
        self.current_box = None 
        
        self.thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.thread.start()

    def process_frame(self, frame):
        if not self.input_queue.full():
            self.input_queue.put(frame)

    def _inference_loop(self):
        while self.running:
            try:
                frame = self.input_queue.get(timeout=1)
                results = self.face_model(frame, verbose=False, conf=0.5, device='cpu')[0]

                if len(results.boxes) > 0:
                    boxes = results.boxes.xyxy.cpu().numpy()
                    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                    largest_idx = np.argmax(areas)
                    x1, y1, x2, y2 = map(int, boxes[largest_idx])

                    self.current_box = (x1, y1, x2, y2)

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, _ = frame.shape
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    face_crop = frame_rgb[y1:y2, x1:x2]
                    
                    if face_crop.size > 0:
                        pil_face = Image.fromarray(face_crop)
                        face_tensor = self.transform(pil_face)
                        self.frame_buffer.append(face_tensor)

                        if len(self.frame_buffer) == VIDEO_SEQ_LENGTH:
                            input_tensor = torch.stack(list(self.frame_buffer)).unsqueeze(0).to(self.device)
                            with torch.no_grad():
                                out = self.model(input_tensor)
                                new_prob = torch.softmax(out, dim=1)[0, 1].item()
                            
                            self.smoothed_prob = (VIDEO_EMA_ALPHA * new_prob) + ((1 - VIDEO_EMA_ALPHA) * self.smoothed_prob)

                            if self.smoothed_prob > VIDEO_CONFIDENCE_THRESHOLD:
                                self.current_label = "FAKE"
                            else:
                                self.current_label = "REAL"
                        else:
                            self.current_label = f"BUFFERING {len(self.frame_buffer)}/{VIDEO_SEQ_LENGTH}"
                else:
                    self.current_box = None

            except queue.Empty:
                continue
            except Exception as e:
                pass

    def get_status(self):
        return self.current_label, self.smoothed_prob, self.current_box

    def stop(self):
        self.running = False
        self.thread.join()

video_detector = None

app = FastAPI(title="Fake News & Deepfake Audio Detector")

app.mount("/static", StaticFiles(directory=r"D:\Verification Syndicate\Fake News\code\frontend"), name="static")
templates = Jinja2Templates(directory=r"D:\Verification Syndicate\Fake News\code\frontend")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

class TextRequest(BaseModel):
    text: str

@app.post("/predict/text")
async def predict_text(request: TextRequest):
    try:
        result = detect_news(request.text)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/audio")
async def predict_audio_endpoint(file: UploadFile = File(...)):
    # Validate file type
    if not file.filename.lower().endswith(('.wav', '.mp3', '.flac', '.m4a', '.ogg')):
        raise HTTPException(status_code=400, detail="Unsupported audio format. Please upload WAV, MP3, FLAC, M4A, or OGG.")
    
    try:
        contents = await file.read()
        result = predict_audio(contents)  
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/video")
async def predict_video_endpoint(file: UploadFile = File(...)):
    global video_detector
    
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
        raise HTTPException(status_code=400, detail="Unsupported video format.")
    
    if video_detector is None:
        video_detector = DeepfakeVideoDetector()
        
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        cap = cv2.VideoCapture(temp_path)
        
        all_probs = []
        fake_frames = 0
        total_analyzed = 0
        
        if not cap.isOpened():
            raise Exception("Failed to open uploaded video file.")


        frame_skip = 5 
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret: 
                break

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

          
            while video_detector.input_queue.full() and video_detector.running:
                await asyncio.sleep(0.01)

            video_detector.process_frame(frame)
            

            await asyncio.sleep(0.01)
            
            label, prob, box = video_detector.get_status()

            if label in ["FAKE", "REAL"]:
                all_probs.append(prob)
                total_analyzed += 1
                if label == "FAKE":
                    fake_frames += 1

        await asyncio.sleep(0.5)
        
        cap.release()
        os.remove(temp_path)
        
        if total_analyzed > 0:
            avg_prob = sum(all_probs) / len(all_probs)
            fake_percentage = (fake_frames / total_analyzed) * 100
            is_fake = avg_prob > 0.5
            
            return JSONResponse(content={
                'label': 'FAKE / SYNTHETIC VIDEO' if is_fake else 'REAL HUMAN VIDEO',
                'score': avg_prob,
                'confidence': avg_prob if is_fake else 1 - avg_prob,
                'details': {
                    'frames_analyzed': total_analyzed,
                    'fake_frames': fake_frames,
                    'fake_percentage': fake_percentage,
                }
            })
        else:
            return JSONResponse(content={
                'label': 'UNABLE TO DETECT',
                'score': 0.0,
                'confidence': 0.0,
                'details': {
                    'frames_analyzed': 0,
                    'message': "No face detected in video or video too short."
                }
            })
            
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)