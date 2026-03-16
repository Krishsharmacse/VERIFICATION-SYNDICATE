import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from torchvision import models
import shutil
import datetime
import json

# --- CONFIGURATION ---
HISTORY_FILE = "history.json"
MODEL_PATH = r"C:\Users\ASUS\Desktop\Fake News\ALL MODELS\best_celebdf_model.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224
FRAMES_SEQUENCE = 16

# --- JSON DATABASE SETUP ---
def save_to_json(data):
    history = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            try:
                history = json.load(f)
            except:
                history = []
    history.insert(0, data)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

# --- MODEL ARCHITECTURE ---
class CNN_BiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.efficientnet_b0(weights=None)
        self.cnn = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.lstm = nn.LSTM(1280, 256, num_layers=2, bidirectional=True, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256), 
            nn.ReLU(), 
            nn.Dropout(0.5), 
            nn.Linear(256, 2)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        feats = self.cnn(x)
        feats = self.pool(feats).flatten(1)
        feats = feats.view(B, T, -1)
        lstm_out, _ = self.lstm(feats)
        return self.classifier(lstm_out[:, -1, :])

# Initialize Model
model = CNN_BiLSTM().to(DEVICE)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"✅ Model Loaded Successfully on {DEVICE}")
else:
    print(f"❌ CRITICAL ERROR: Model not found at {MODEL_PATH}")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- FASTAPI APP ---
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    cap = cv2.VideoCapture(temp_path)
    frame_buffer = [] 
    fake_count = 0
    total_predictions = 0
    logs = []

    print(f"🚀 Analyzing Video: {file.filename}")

    while True:
        ret, frame = cap.read()
        if not ret: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))

        label = "ANALYZING..."
        color = (255, 255, 255)

        for (x, y, w, h) in faces:
            # Pre-processing face crop
            face_img = frame[y:y+h, x:x+w]
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_img = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
            
            # Normalization (ImageNet Stats)
            face_img = (face_img.astype(np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            frame_buffer.append(np.transpose(face_img, (2, 0, 1)))

            # Non-overlapping window trigger (Every 16 frames)
            if len(frame_buffer) == FRAMES_SEQUENCE:
                # FIX: Convert to numpy array first, then to Float Tensor
                input_arr = np.array(frame_buffer)
                input_seq = torch.from_numpy(input_arr).float().unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    # Optional: AMP for faster GPU inference
                    with torch.amp.autocast('cuda' if 'cuda' in DEVICE else 'cpu'):
                        output = model(input_seq)
                    
                    probs = torch.softmax(output, dim=1)
                    conf, pred = torch.max(probs, dim=1)
                    
                    total_predictions += 1
                    is_fake = pred.item() == 1
                    conf_val = conf.item() * 100

                    if is_fake:
                        fake_count += 1
                        logs.append(f"Sequence {total_predictions}: FAKE ({conf_val:.1f}%)")
                    
                    label = f"{'FAKE' if is_fake else 'REAL'} | {conf_val:.1f}%"
                    color = (0, 0, 255) if is_fake else (0, 255, 0)
                
                frame_buffer = [] # Reset buffer for next block
            
            # Draw on OpenCV Window (Server-side)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            break 

        cv2.imshow("TERMINUS FORENSIC ENGINE", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    if os.path.exists(temp_path): os.remove(temp_path)

    # FINAL CALCULATION
    fake_ratio = (fake_count / total_predictions * 100) if total_predictions > 0 else 0
    verdict = "DEEPFAKE" if fake_ratio > 15.0 else "REAL"

    # Save metadata to history.json
    result_data = {
        "filename": file.filename,
        "verdict": verdict,
        "fake_ratio": round(fake_ratio, 2),
        "total_seq": total_predictions,
        "fake_seq": fake_count,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    save_to_json(result_data)

    print(f"🏁 Finished. Verdict: {verdict} ({fake_ratio:.2f}%)")
    
    return {
        "verdict": verdict,
        "fake_ratio": round(fake_ratio, 2),
        "total_sequences": total_predictions,
        "fake_detections": fake_count,
        "logs": logs[-10:]
    }

@app.get("/history")
async def get_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)