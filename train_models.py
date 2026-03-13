# train_models.py
import os
import asyncio
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import torch
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import config
from Backend.config import Config


class ModelTrainer:
    """Train and save models for fake news detection"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Create models directory if it doesn't exist
        Path("models").mkdir(exist_ok=True)
        
    async def prepare_training_data(self):
        """Prepare training data from datasets"""
        logger.info("Preparing training data...")
        
        # Check if training data already exists
        if os.path.exists(Config.TRAINING_DATA_PATH):
            logger.info(f"Loading existing training data from {Config.TRAINING_DATA_PATH}")
            df = pd.read_csv(Config.TRAINING_DATA_PATH)
        else:
            # Create sample training data structure
            # In production, load from actual dataset
            logger.info("Creating sample training data structure...")
            df = self._create_sample_data()
            
            # Save for future use
            os.makedirs(os.path.dirname(Config.TRAINING_DATA_PATH), exist_ok=True)
            df.to_csv(Config.TRAINING_DATA_PATH, index=False)
            logger.info(f"Sample training data saved to {Config.TRAINING_DATA_PATH}")
        
        return df
    
    def _create_sample_data(self):
        """Create sample training data"""
        data = {
            'text': [
                "Breaking: New cure found for cancer - click here!",
                "Scientists publish peer-reviewed study on climate change",
                "URGENT: This secret trick doctors don't want you to know!",
                "Official announcement from government health ministry",
                "You won't believe what celebrities are hiding from us",
                "Research shows new vaccine approved by health authorities",
                "TERRIFYING: Check this out immediately or regret it!",
                "University research demonstrates effective treatment",
                "This one weird trick will shock you",
                "Established news outlet reports on recent events"
            ],
            'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 = fake, 0 = real
        }
        
        # Expand data by repeating with variations
        df = pd.DataFrame(data * 10)  # Repeat 10 times
        df = df.sample(frac=1).reset_index(drop=True)  # Shuffle
        
        return df
    
    async def train_dl_classifier(self, df: pd.DataFrame):
        """Train deep learning classifier using BERT"""
        logger.info("Training DL Classifier...")
        
        try:
            # Prepare data
            texts = df['text'].tolist()
            labels = df['label'].tolist()
            
            # Split data
            train_texts, test_texts, train_labels, test_labels = train_test_split(
                texts, labels, test_size=0.2, random_state=42
            )
            
            logger.info(f"Training set size: {len(train_texts)}")
            logger.info(f"Test set size: {len(test_texts)}")
            
            # Load pre-trained model
            logger.info("Loading BERT model...")
            model_name = "bert-base-uncased"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=2
            )
            
            model.to(self.device)
            
            # Tokenize data
            logger.info("Tokenizing training data...")
            train_encodings = tokenizer(
                train_texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Create simple training loop
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
            epochs = 3
            batch_size = 8
            
            logger.info(f"Starting training for {epochs} epochs...")
            
            for epoch in range(epochs):
                logger.info(f"Epoch {epoch + 1}/{epochs}")
                model.train()
                
                total_loss = 0
                num_batches = 0
                
                for i in range(0, len(train_texts), batch_size):
                    # Get batch
                    batch_texts = train_texts[i:i+batch_size]
                    batch_labels = train_labels[i:i+batch_size]
                    
                    # Tokenize batch
                    batch_encodings = tokenizer(
                        batch_texts,
                        truncation=True,
                        padding=True,
                        max_length=512,
                        return_tensors="pt"
                    )
                    
                    # Move to device
                    input_ids = batch_encodings['input_ids'].to(self.device)
                    attention_mask = batch_encodings['attention_mask'].to(self.device)
                    labels = torch.tensor(batch_labels).to(self.device)
                    
                    # Forward pass
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    loss = outputs.loss
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    if (num_batches) % 2 == 0:
                        logger.info(f"Batch {num_batches}, Loss: {loss.item():.4f}")
                
                avg_loss = total_loss / num_batches
                logger.info(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")
            
            # Save model and tokenizer
            model_path = "models/dl_classifier"
            os.makedirs(model_path, exist_ok=True)
            
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
            
            logger.info(f"DL Classifier saved to {model_path}")
            
            # Evaluate on test set
            logger.info("Evaluating on test set...")
            model.eval()
            
            correct = 0
            total = 0
            
            with torch.no_grad():
                for i in range(0, len(test_texts), batch_size):
                    batch_texts = test_texts[i:i+batch_size]
                    batch_labels = test_labels[i:i+batch_size]
                    
                    batch_encodings = tokenizer(
                        batch_texts,
                        truncation=True,
                        padding=True,
                        max_length=512,
                        return_tensors="pt"
                    )
                    
                    input_ids = batch_encodings['input_ids'].to(self.device)
                    attention_mask = batch_encodings['attention_mask'].to(self.device)
                    
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    predictions = torch.argmax(outputs.logits, dim=1)
                    
                    correct += (predictions.cpu().numpy() == np.array(batch_labels)).sum()
                    total += len(batch_labels)
            
            accuracy = correct / total
            logger.info(f"Test Accuracy: {accuracy:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training DL Classifier: {str(e)}")
            return False
    
    async def train_bharat_model(self, df: pd.DataFrame):
        """Train Bharat Fake News Kosh model"""
        logger.info("Training Bharat Fake News Kosh model...")
        
        try:
            # Extract features
            features = []
            for text in df['text'].tolist():
                feature_vec = self._extract_features(text)
                features.append(feature_vec)
            
            features = np.array(features)
            labels = df['label'].tolist()
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=0.2, random_state=42
            )
            
            logger.info(f"Training set size: {len(X_train)}")
            
            # Train Random Forest
            logger.info("Training Random Forest classifier...")
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            
            # Evaluate
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            logger.info(f"Training Accuracy: {train_score:.4f}")
            logger.info(f"Test Accuracy: {test_score:.4f}")
            
            # Save model
            model_path = "models/bharat_model.joblib"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            joblib.dump(model, model_path)
            
            logger.info(f"Bharat model saved to {model_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training Bharat model: {str(e)}")
            return False
    
    def _extract_features(self, text):
        """Extract features for classical ML model"""
        features = []
        
        # Text length
        features.append(len(text))
        
        # Word count
        features.append(len(text.split()))
        
        # Average word length
        words = text.split()
        features.append(np.mean([len(w) for w in words]) if words else 0)
        
        # Exclamation marks
        features.append(text.count('!'))
        
        # Question marks
        features.append(text.count('?'))
        
        # Uppercase ratio
        features.append(sum(1 for c in text if c.isupper()) / max(len(text), 1))
        
        # Suspicious words count
        suspicious = ['urgent', 'shocking', 'unbelievable', 'click here', 'secret']
        features.append(sum(1 for word in suspicious if word in text.lower()))
        
        return features
    
    async def train_all_models(self):
        """Train all models"""
        logger.info("Starting model training...")
        
        try:
            # Prepare data
            df = await self.prepare_training_data()
            
            # Train models
            results = {}
            
            results['dl_classifier'] = await self.train_dl_classifier(df)
            results['bharat_model'] = await self.train_bharat_model(df)
            
            logger.info("\n" + "="*50)
            logger.info("TRAINING SUMMARY")
            logger.info("="*50)
            
            for model_name, success in results.items():
                status = "✓ Success" if success else "✗ Failed"
                logger.info(f"{model_name}: {status}")
            
            logger.info("="*50)
            logger.info("Model training completed!")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            return {}


async def main():
    """Main training function"""
    trainer = ModelTrainer()
    await trainer.train_all_models()


if __name__ == "__main__":
    asyncio.run(main())
