import os
import torch
import librosa
import numpy as np
import tensorflow as tf
from transformers import Wav2Vec2FeatureExtractor, AutoModel

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
# Look inside the 'ALL MODELS' folder based on your workspace
KERAS_MODEL_PATH = os.path.join("ALL MODELS", "wavlm_classifier_v2.keras")
WAVLM_MODEL_NAME = "microsoft/wavlm-base-plus"

SAMPLE_RATE = 16000
MAX_DURATION = 5  # Analyze 5 seconds of audio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. LOAD MODELS (LOAD ONCE)
# ==========================================
print(f"🚀 Initializing on: {device}")
print("⏳ Loading Models... (Please wait)")

try:
    # Load WavLM (The Feature Extractor)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(WAVLM_MODEL_NAME)
    wavlm_model = AutoModel.from_pretrained(WAVLM_MODEL_NAME).to(device)
    wavlm_model.eval()
    
    # Load Your Trained Classifier
    keras_model = tf.keras.models.load_model(KERAS_MODEL_PATH)
    print("✅ System Ready!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    exit()

# ==========================================
# 3. PREDICTION FUNCTION
# ==========================================
def predict_audio(file_path):
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found -> {file_path}")
        return

    print(f"\n📂 Analyzing: {os.path.basename(file_path)}")
    
    try:
        # 1. Load and standardise audio
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        max_len = SAMPLE_RATE * MAX_DURATION
        
        if len(y) < max_len:
            y = np.pad(y, (0, max_len - len(y)), 'constant')
        else:
            y = y[:max_len]
            
        # 2. Extract WavLM embeddings
        inputs = feature_extractor(y, sampling_rate=SAMPLE_RATE, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = wavlm_model(**inputs)
            
        # Get Mean Embedding
        hidden_states = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
        embedding = torch.mean(hidden_states, dim=1).cpu().numpy()
        
        # 3. Classify with Keras Model
        score = keras_model.predict(embedding, verbose=0)[0][0]
        
        # 4. Display Results
        is_fake = score > 0.5
        confidence = score if is_fake else 1 - score
        
        print("-" * 50)
        if is_fake:
            print(f"🛑 RESULT: FAKE / SYNTHETIC AUDIO")
            print(f"📊 Confidence: {confidence:.2%}")
            print(f"☠️ Score: {score:.4f} (Closer to 1 = Fake)")
        else:
            print(f"✅ RESULT: REAL HUMAN VOICE")
            print(f"📊 Confidence: {confidence:.2%}")
            print(f"😊 Score: {score:.4f} (Closer to 0 = Real)")
        print("-" * 50)
        
        return score
        
    except Exception as e:
        print(f"❌ Error processing file: {e}")
        return None

# ==========================================
# 4. TEST IT
# ==========================================
if __name__ == "__main__":
    # 👉 Change this path to the audio file you want to test
    # You can drag and drop an audio file into your VS Code terminal to get its path!
    
    test_file = r"C:\Users\ASUS\Downloads\Saiyaara Feat. Modi Ji 😀(MP3_160K).mp3"
    
    predict_audio(test_file)