import torch
import torch.nn as nn
import pickle
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# ================= DEVICE =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ================= NLTK TOOLS =================
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ================= LOAD VOCAB =================
with open("vocab_nltk.pkl", "rb") as f:
    vocab = pickle.load(f)

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
model = ImprovedBiLSTM(VOCAB_SIZE).to(device)
model.load_state_dict(torch.load("fake_news_bilstm_nltk.pth", map_location=device))
model.eval()
print("Model loaded successfully!")

# ================= TEXT CLEANING =================
source_words = set([
    'reuters', 'cnn', 'fox', 'breitbart', 'nytimes', 'washingtonpost',
    'bbc', 'abc', 'nbc', 'cbs', 'ap', 'associated', 'press', 'apnews',
    'guardian', 'wsj', 'usatoday', 'huffington', 'huffpost', 'buzzfeed',
    'said', 'told', 'reported', 'according', 'source', 'sources',
    'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
    'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august',
    'september', 'october', 'november', 'december'
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

# ================= TEXT → SEQUENCE =================
def text_to_seq(text):
    tokens = text.split()
    seq = [vocab.get(w, 1) for w in tokens]  # 1 = <UNK>
    if len(seq) < MAX_LEN:
        seq += [0] * (MAX_LEN - len(seq))
    else:
        seq = seq[:MAX_LEN]
    return seq

# ================= PREDICTION FUNCTION =================
def predict_news(text, confidence_threshold=0.6):
    cleaned = advanced_nltk_clean(text)
    seq = text_to_seq(cleaned)
    tensor = torch.tensor(seq).unsqueeze(0).to(device)
    
    with torch.no_grad():
        out = model(tensor)
        prob = torch.sigmoid(out).item()
    
    label = "Real News" if prob >= 0.5 else "Fake News"
    confidence = prob if prob >= 0.5 else 1 - prob
    is_certain = confidence >= confidence_threshold
    
    return {
        'label': label,
        'probability': prob,
        'confidence': confidence,
        'is_certain': is_certain,
        'cleaned_text': cleaned
    }

# ================= TEST SAMPLES =================
test_samples = [
    "Scientists discovered water on Mars in new NASA study",
    "Aliens met the president yesterday in secret White House meeting",
    "Government passes new economic bill to help small businesses",
    "BREAKING: SHOCKING video exposes government cover-up",
    "FDA approves new cancer treatment with 90% success rate",
    "You won't believe what this celebrity did next!"
    "The United Nations reported that global deforestation rates have declined significantly in the past year due to conservation policies and international cooperation. Experts say this is a major step in combating climate change"
]

for i, text in enumerate(test_samples, 1):
    result = predict_news(text)
    print(f"\n{i}. Original: {text}")
    print(f"   Cleaned: {result['cleaned_text'][:100]}...")
    print(f"   Prediction: {result['label']} (Confidence: {result['confidence']:.2%})")
    print(f"   Certain: {'✓' if result['is_certain'] else '⚠️ Low confidence'}")