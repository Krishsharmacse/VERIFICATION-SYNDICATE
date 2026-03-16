import cv2
import torch
import torch.nn as nn
import numpy as np
import collections
import threading
import queue
import time
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO 

# =========================
# CONFIGURATION
# =========================
# ✅ YOUR SPECIFIC VIDEO PATH
VIDEO_PATH = r"C:\Users\ASUS\Pictures\Camera Roll\WIN_20260317_00_12_35_Pro.mp4"

# ✅ YOUR TRAINED MODEL
MODEL_PATH = r"C:\Users\ASUS\Desktop\Fake News\ALL MODELS\best_celebdf_model_Krish.pt" 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_SIZE = 224
SEQ_LENGTH = 16        
CONFIDENCE_THRESHOLD = 0.60
EMA_ALPHA = 0.15       

# =========================
# MODEL DEFINITION
# =========================
class CNN_BiLSTM(nn.Module):
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

# =========================
# AI WORKER CLASS
# =========================
class DeepfakeDetector:
    def __init__(self):
        print(f"🚀 Loading AI on {DEVICE}...")
        self.device = DEVICE
        
        print("   • Loading YOLOv8-Face...")
        self.face_model = YOLO(r"C:\Users\ASUS\Downloads\yolov8s-face-lindevs.onnx", task='detect') 
        
        print(f"   • Loading {MODEL_PATH}...")
        self.model = CNN_BiLSTM().to(self.device)
        try:
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
            self.model.eval()
            print("✅ Model loaded successfully.")
        except FileNotFoundError:
            print(f"❌ Error: Could not find {MODEL_PATH}")
            exit()

        self.transform = transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.frame_buffer = collections.deque(maxlen=SEQ_LENGTH)
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
                
                # 1. YOLO Detection
                results = self.face_model(frame, verbose=False, conf=0.5, device='cpu')[0]

                if len(results.boxes) > 0:
                    boxes = results.boxes.xyxy.cpu().numpy()
                    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                    largest_idx = np.argmax(areas)
                    x1, y1, x2, y2 = map(int, boxes[largest_idx])

                    self.current_box = (x1, y1, x2, y2)

                    # 2. Crop
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, _ = frame.shape
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    face_crop = frame_rgb[y1:y2, x1:x2]
                    
                    if face_crop.size > 0:
                        pil_face = Image.fromarray(face_crop)
                        face_tensor = self.transform(pil_face)
                        self.frame_buffer.append(face_tensor)

                        # 3. Inference
                        if len(self.frame_buffer) == SEQ_LENGTH:
                            input_tensor = torch.stack(list(self.frame_buffer)).unsqueeze(0).to(self.device)
                            with torch.no_grad():
                                out = self.model(input_tensor)
                                new_prob = torch.softmax(out, dim=1)[0, 1].item()
                            
                            self.smoothed_prob = (EMA_ALPHA * new_prob) + ((1 - EMA_ALPHA) * self.smoothed_prob)

                            if self.smoothed_prob > CONFIDENCE_THRESHOLD:
                                self.current_label = "FAKE"
                            else:
                                self.current_label = "REAL"
                        else:
                            self.current_label = f"BUFFERING {len(self.frame_buffer)}/{SEQ_LENGTH}"
                else:
                    self.current_box = None

            except queue.Empty:
                continue
            except Exception as e:
                print(f"⚠️ Error: {e}")

    def get_status(self):
        return self.current_label, self.smoothed_prob, self.current_box

    def stop(self):
        self.running = False
        self.thread.join()

# =========================
# MAIN LOOP
# =========================
def main():
    detector = DeepfakeDetector()
    
    print(f"🎥 Opening video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("❌ Error: Could not open video file. Check the path!")
        return

    all_probs = []
    fake_frames = 0
    total_analyzed = 0

    while True:
        ret, frame = cap.read()
        
        # If video ends, break (or use cap.set(cv2.CAP_PROP_POS_FRAMES, 0) to loop)
        if not ret: 
            print("End of video.")
            break

        # Send frame to AI
        detector.process_frame(frame)
        label, prob, box = detector.get_status()

        if label in ["FAKE", "REAL"]:
            all_probs.append(prob)
            total_analyzed += 1
            if label == "FAKE":
                fake_frames += 1

        # Colors
        if label == "FAKE": color = (0, 0, 255)      # Red
        elif label == "REAL": color = (0, 255, 0)    # Green
        else: color = (0, 255, 255)                  # Yellow

        # Draw Box
        if box is not None:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(frame, (x1, y1-30), (x2, y1), color, -1)
            text = f"{label} ({prob*100:.0f}%)" if label in ["REAL", "FAKE"] else label
            cv2.putText(frame, text, (x1+5, y1-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

        # Draw Dashboard
        h, w, _ = frame.shape
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        bar_width = 300
        bar_x = 20
        bar_y = 35
        fill_width = int(bar_width * prob)
        bar_color = (0, 255 - int(255*prob), int(255*prob)) 
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 15), (50, 50, 50), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + 15), bar_color, -1)
        
        cv2.putText(frame, "DEEPFAKE PROBABILITY:", (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, f"{prob*100:.1f}%", (bar_x + bar_width + 10, bar_y + 13), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Single Video Test", frame)

        # Wait 30ms (approx 30fps) - Press 'q' to quit
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    detector.stop()
    cap.release()
    cv2.destroyAllWindows()

    # Calculate final output
    print("\n" + "="*50)
    print("📊 FINAL VIDEO VERDICT")
    print("="*50)

    if total_analyzed > 0:
        avg_prob = sum(all_probs) / len(all_probs)
        fake_percentage = (fake_frames / total_analyzed) * 100
        
        print(f"Frames Analyzed:   {total_analyzed}")
        print(f"Fake Frames Found: {fake_frames} ({fake_percentage:.1f}%)")
        print(f"Average Fake Prob: {avg_prob*100:.1f}%\n")

        # Thresholds can be adjusted (e.g., if > 50% average probability = FAKE)
        if avg_prob > 0.5:
            print("🚨 VERDICT: FAKE / DEEPFAKE DETECTED 🚨")
            print(f"Confidence: {avg_prob*100:.1f}%")
        else:
            print("✅ VERDICT: REAL / HUMAN VERIFIED ✅")
            print(f"Confidence: {(1-avg_prob)*100:.1f}%")
    else:
        print("⚠️ No valid frames were analyzed. No face detected?")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()