import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
import re
from collections import Counter

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import warnings 
warnings.filterwarnings("ignore")

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from nltk import pos_tag, ne_chunk

print("Downloading NLTK data...")
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)
nltk.download('averaged_perceptron_tagger')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)



true_path = r"C:\Users\ASUS\Desktop\Fake News\Datasets\archive\True.csv"
fake_path = r"C:\Users\ASUS\Desktop\Fake News\Datasets\archive\Fake.csv"

liar_train_path = r"C:\Users\ASUS\Desktop\Fake News\Datasets\archive (1)\train.tsv"
liar_valid_path = r"C:\Users\ASUS\Desktop\Fake News\Datasets\archive (1)\valid.tsv"
liar_test_path = r"C:\Users\ASUS\Desktop\Fake News\Datasets\archive (1)\test.tsv"

print("\nLoading datasets...")

fake_df = pd.read_csv(fake_path)
true_df = pd.read_csv(true_path)

fake_df["label"] = 0
true_df["label"] = 1

df_news = pd.concat([fake_df, true_df])
df_news["content"] = df_news["title"] + " " + df_news["text"]
df_news = df_news[["content", "label"]]


liar_train = pd.read_csv(liar_train_path, sep="\t", header=None)
liar_valid = pd.read_csv(liar_valid_path, sep="\t", header=None)
liar_test = pd.read_csv(liar_test_path, sep="\t", header=None)

liar_train = liar_train.iloc[:, [2, 3]]
liar_valid = liar_valid.iloc[:, [2, 3]]
liar_test = liar_test.iloc[:, [2, 3]]

liar_train.columns = ["label_text", "content"]
liar_valid.columns = ["label_text", "content"]
liar_test.columns = ["label_text", "content"]

fake_labels = ["false", "pants-fire", "barely-true"]

for df in [liar_train, liar_valid, liar_test]:
    df["label"] = df["label_text"].apply(lambda x: 0 if x in fake_labels else 1)
    df.drop(columns=["label_text"], inplace=True)

liar_all = pd.concat([liar_train, liar_valid, liar_test])


merged_df = pd.concat([df_news, liar_all])
merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)

print("Total samples:", len(merged_df))
print(merged_df["label"].value_counts())


print("\nApplying NLTK preprocessing...")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

source_words = set([
    'reuters', 'cnn', 'fox', 'breitbart', 'nytimes', 'washingtonpost',
    'bbc', 'abc', 'nbc', 'cbs', 'ap', 'associated', 'press', 'apnews',
    'guardian', 'wsj', 'usatoday', 'huffington', 'huffpost', 'buzzfeed',
    'said', 'told', 'reported', 'according', 'source', 'sources',
    'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
    'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august',
    'september', 'october', 'november', 'december'
])

fake_indicators = set([
    'shocking', 'breaking', 'exposed', 'warning', 'alert', 'urgent',
    'viral', 'incredible', 'unbelievable', 'mindblowing', 'secret',
    'hidden', 'truth', 'conspiracy', 'coverup', 'they', 'government',
    'media', 'lying', 'deception', 'scam', 'hoax', 'exposed'
])

def advanced_nltk_clean(text):
    """
    Advanced text preprocessing using NLTK
    """
  
    text = str(text).lower()
    
    
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    
  
    text = re.sub(r"[^a-z\s]", " ", text)
  
    tokens = word_tokenize(text)
    
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [t for t in tokens if t not in source_words]
    

    pos_tags = pos_tag(tokens)

    lemmatized_tokens = []
    for word, tag in pos_tags:

        if tag.startswith('V'): 
            pos = 'v'
        elif tag.startswith('J'):  
            pos = 'a'
        elif tag.startswith('R'):  
            pos = 'r'
        else:  
            pos = 'n'
  
        lemmatized = lemmatizer.lemmatize(word, pos=pos)
        lemmatized_tokens.append(lemmatized)
    

    
    # Join tokens back into text
    cleaned_text = ' '.join(lemmatized_tokens)
    
    # Remove extra whitespace
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    
    return cleaned_text

# Apply advanced cleaning
merged_df["content"] = merged_df["content"].apply(advanced_nltk_clean)

# Remove empty content
merged_df = merged_df[merged_df["content"].str.len() > 0]
print("Samples after cleaning:", len(merged_df))

# Show sample of cleaned text
print("\nSample of cleaned text:")
print(merged_df["content"].iloc[0][:200])


def extract_named_entities(text):
    """
    Extract named entities (could be used as additional features)
    """
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    named_entities = ne_chunk(pos_tags)
    
    entities = []
    for chunk in named_entities:
        if hasattr(chunk, 'label'):
            entities.append(' '.join(c[0] for c in chunk))
    
    return entities


X = merged_df["content"].values
y = merged_df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("\nTrain size:", len(X_train))
print("Test size:", len(X_test))


MAX_VOCAB = 20000
MAX_LEN = 150

def build_vocab(texts):
    words = [word for text in texts for word in text.split()]
    counts = Counter(words)
    common = counts.most_common(MAX_VOCAB)
    vocab = {w: i + 2 for i, (w, _) in enumerate(common)}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1
    return vocab


vocab = build_vocab(X_train)
VOCAB_SIZE = len(vocab)
print("Vocabulary size:", VOCAB_SIZE)

def text_to_seq(text):
    tokens = text.split()
    seq = [vocab.get(w, 1) for w in tokens]
    
    if len(seq) < MAX_LEN:
        seq = seq + [0] * (MAX_LEN - len(seq))
    else:
        seq = seq[:MAX_LEN]
    
    return seq


X_train_seq = np.array([text_to_seq(t) for t in X_train])
X_test_seq = np.array([text_to_seq(t) for t in X_test])

class FakeDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.long),
            torch.tensor(self.y[idx], dtype=torch.float32)
        )


train_dataset = FakeDataset(X_train_seq, y_train)
test_dataset = FakeDataset(X_test_seq, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

class ImprovedBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_layers=2, dropout=0.5):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(dropout)
        
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        # Classifier
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
        # Embedding
        embedded = self.embedding(x)
        embedded = self.embedding_dropout(embedded)
        
        # LSTM
        lstm_out, _ = self.lstm(embedded)
        
        # Attention
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        attended = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Classification
        output = self.classifier(attended)
        
        return output.squeeze()


model = ImprovedBiLSTM(VOCAB_SIZE).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)


# ================= TRAINING WITH VALIDATION =================
print("\nTraining model...")

# Split training data for validation
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train_seq, y_train, test_size=0.15, stratify=y_train, random_state=42
)

train_dataset = FakeDataset(X_train_split, y_train_split)
val_dataset = FakeDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

EPOCHS = 10
best_val_loss = float('inf')
patience = 3
counter = 0

for epoch in range(EPOCHS):
    # Training
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for x, yb in train_loader:
        x = x.to(device)
        yb = yb.to(device)
        
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, yb)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        preds = (torch.sigmoid(out) >= 0.5).float()
        correct += (preds == yb).sum().item()
        total += yb.size(0)
    
    train_acc = correct / total
    avg_loss = total_loss / len(train_loader)
    
    # Validation
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for x, yb in val_loader:
            x = x.to(device)
            yb = yb.to(device)
            out = model(x)
            loss = criterion(out, yb)
            val_loss += loss.item()
            
            preds = (torch.sigmoid(out) >= 0.5).float()
            val_correct += (preds == yb).sum().item()
            val_total += yb.size(0)
    
    val_acc = val_correct / val_total
    avg_val_loss = val_loss / len(val_loader)
    
    print(f"Epoch {epoch+1:2d} | Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")
    
    # Learning rate scheduling
    scheduler.step(avg_val_loss)
    
    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        counter = 0
        torch.save(model.state_dict(), "best_model.pth")
        print(f"  → Best model saved!")
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# Load best model
model.load_state_dict(torch.load("best_model.pth"))


# ================= EVALUATION =================
print("\nEvaluating on test set...")
model.eval()
preds = []
truth = []
probs = []

with torch.no_grad():
    for x, yb in test_loader:
        x = x.to(device)
        out = model(x)
        prob = torch.sigmoid(out)
        p = (prob >= 0.5).float()
        
        preds.extend(p.cpu().numpy())
        truth.extend(yb.numpy())
        probs.extend(prob.cpu().numpy())


acc = accuracy_score(truth, preds)
print("\nTest Accuracy:", acc)
print(f"Probability range: {min(probs):.4f} - {max(probs):.4f}")
print(f"Probability mean: {np.mean(probs):.4f}")

print("\nClassification Report\n")
print(classification_report(truth, preds, target_names=["Fake", "Real"]))


# Confusion Matrix
cm = confusion_matrix(truth, preds)
plt.figure(figsize=(6,4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Fake","Real"],
    yticklabels=["Fake","Real"]
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix with NLTK Preprocessing")
plt.show()


# ================= CROSS VALIDATION =================
print("\nCross Validation with NLTK")
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_seq, y_train)):
    print(f"\nFold {fold+1}")
    
    model = ImprovedBiLSTM(VOCAB_SIZE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_ds = FakeDataset(X_train_seq[train_idx], y_train[train_idx])
    val_ds = FakeDataset(X_train_seq[val_idx], y_train[val_idx])
    
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=64)
    
    for epoch in range(3):
        model.train()
        for x, yb in train_dl:
            x = x.to(device)
            yb = yb.to(device)
            
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
    
    model.eval()
    vp = []
    vt = []
    
    with torch.no_grad():
        for x, yb in val_dl:
            x = x.to(device)
            out = model(x)
            p = (torch.sigmoid(out) >= 0.5).float()
            vp.extend(p.cpu().numpy())
            vt.extend(yb.numpy())
    
    fold_acc = accuracy_score(vt, vp)
    scores.append(fold_acc)
    print(f"Accuracy: {fold_acc:.4f}")

print(f"\nAverage CV Accuracy: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")


# ================= ENHANCED PREDICTION FUNCTION =================
def predict_news(text, return_details=False):
    """
    Enhanced prediction with NLTK preprocessing
    """
    model.eval()
    
    # Apply NLTK cleaning
    cleaned = advanced_nltk_clean(text)
    
    # Convert to sequence
    seq = text_to_seq(cleaned)
    tensor = torch.tensor(seq).unsqueeze(0).to(device)
    
    with torch.no_grad():
        out = model(tensor)
        prob = torch.sigmoid(out).item()
    
    label = "Real News" if prob >= 0.5 else "Fake News"
    confidence = prob if prob >= 0.5 else 1 - prob
    
    if return_details:
        return {
            'label': label,
            'probability': prob,
            'confidence': confidence,
            'cleaned_text': cleaned,
            'is_certain': confidence >= 0.6
        }
    return label, prob


# ================= TEST PREDICTIONS =================
print("\n" + "="*50)
print("TESTING PREDICTIONS WITH NLTK PREPROCESSING")
print("="*50)

test_samples = [
    "Scientists discovered water on Mars in new NASA study",
    "Aliens met the president yesterday in secret White House meeting",
    "Government passes new economic bill to help small businesses",
    "BREAKING: SHOCKING video exposes government cover-up",
    "FDA approves new cancer treatment with 90% success rate",
    "You won't believe what this celebrity did next!"
]

for i, sample in enumerate(test_samples, 1):
    result = predict_news(sample, return_details=True)
    print(f"\n{i}. Original: {sample}")
    print(f"   Cleaned: {result['cleaned_text'][:100]}...")
    print(f"   Prediction: {result['label']} (Confidence: {result['confidence']:.2%})")
    print(f"   Certain: {'✓' if result['is_certain'] else '⚠️ Low confidence'}")


# ================= ANALYZE WORD IMPORTANCE =================
def analyze_important_words(text):
    """
    Show which words most influenced the prediction
    """
    model.eval()
    cleaned = advanced_nltk_clean(text)
    words = cleaned.split()
    
    if not words:
        return
    
    print(f"\nAnalyzing: {text[:100]}...")
    print("Word-by-word importance:")
    
    base_seq = text_to_seq(cleaned)
    base_tensor = torch.tensor(base_seq).unsqueeze(0).to(device)
    
    with torch.no_grad():
        base_out = model(base_tensor)
        base_prob = torch.sigmoid(base_out).item()
    
    for i, word in enumerate(words[:10]):  # Check first 10 words
        if i >= len(base_seq):
            break
            
        # Create a version without this word
        modified_words = words.copy()
        modified_words.pop(i)
        modified_text = ' '.join(modified_words)
        
        if modified_text:
            modified_seq = text_to_seq(modified_text)
            modified_tensor = torch.tensor(modified_seq).unsqueeze(0).to(device)
            
            with torch.no_grad():
                modified_out = model(modified_tensor)
                modified_prob = torch.sigmoid(modified_out).item()
            
            prob_diff = abs(base_prob - modified_prob)
            print(f"   '{word}': impact {prob_diff:.4f}")


# Test word importance
analyze_important_words(test_samples[0])


# ================= SAVE MODEL =================
torch.save(model.state_dict(), "fake_news_bilstm_nltk.pth")
with open("vocab_nltk.pkl", "wb") as f:
    pickle.dump(vocab, f)

print("\n✅ Model saved with NLTK preprocessing!")


# ================= SUMMARY =================
print("\n" + "="*50)
print("SUMMARY")
print("="*50)
print(f"Vocabulary size: {VOCAB_SIZE}")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Test accuracy: {acc:.4f}")
print(f"Cross-validation mean: {np.mean(scores):.4f}")
print("\nNLTK preprocessing includes:")
print("✅ Tokenization")
print("✅ Stopword removal")
print("✅ Source word removal")
print("✅ POS tagging")
print("✅ Lemmatization")
print("✅ Named entity awareness")