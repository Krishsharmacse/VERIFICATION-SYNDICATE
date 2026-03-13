# config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Gemini API
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    # Redis (for state management)
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # APITube API Key
    APITUBE_API_KEY = os.getenv("APITUBE_API_KEY", "api_live_6OrF7WyLfCunoG45FnYaOPii75mVFf5myW2m5xktVqzDCN3eQ9DbcmMQmgh")
    
    # Model paths
    BHARAT_FAKE_NEWS_MODEL_PATH = "models/bharat_fake_news_kosh"
    NOVEMOFAKE_MODEL_PATH = "models/novemofake"
    TRAINING_DATA_PATH = "datasets/training_data.csv"
