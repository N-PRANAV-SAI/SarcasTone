import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import joblib
from sarcastone.config import *


class SarcasmTextDataset(Dataset):
    """Dataset for text-based sarcasm detection."""
    def __init__(self, texts, labels, tokenizer, max_length=TEXT_MAX_LENGTH):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }


class TextSarcasmClassifier(nn.Module):
    """DistilBERT-based text classifier for sarcasm detection."""
    def __init__(self, model_name=TEXT_MODEL_NAME, num_classes=2, dropout=0.3):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits

    def predict_with_confidence(self, input_ids, attention_mask):
        """Return prediction and confidence score."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            confidence, prediction = torch.max(probs, dim=1)
        return prediction.detach().cpu().numpy(), confidence.detach().cpu().numpy()


def get_bert_embeddings(texts, tokenizer, model, device='cpu', max_length=TEXT_MAX_LENGTH):
    """Extract DistilBERT [CLS] embeddings for a list of texts."""
    model.eval()
    embeddings = []

    for text in texts:
        encoding = tokenizer(
            str(text),
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(device)

        with torch.no_grad():
            outputs = model.bert(
                input_ids=encoding['input_ids'],
                attention_mask=encoding['attention_mask']
            )
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).detach().cpu().numpy()
            embeddings.append(cls_embedding)

    return np.array(embeddings)


def train_text_model(train_texts, train_labels, val_texts=None, val_labels=None,
                     epochs=TEXT_EPOCHS, lr=TEXT_LEARNING_RATE, batch_size=TEXT_BATCH_SIZE,
                     device='cpu'):
    """Train the text sarcasm classifier."""
    tokenizer = DistilBertTokenizer.from_pretrained(TEXT_MODEL_NAME)
    model = TextSarcasmClassifier().to(device)

    train_dataset = SarcasmTextDataset(train_texts, train_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    return model, tokenizer


def evaluate_text_model(model, tokenizer, texts, labels, device='cpu', batch_size=TEXT_BATCH_SIZE):
    """Evaluate the text model and return metrics."""
    dataset = SarcasmTextDataset(texts, labels, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            batch_labels = batch['label'].numpy()

            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_preds.extend(preds)
            all_labels.extend(batch_labels)
            all_probs.extend(probs[:, 1])

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    metrics = {
        'accuracy': float(accuracy_score(all_labels, all_preds)),
        'f1': float(f1_score(all_labels, all_preds, average='weighted')),
        'report': classification_report(all_labels, all_preds, output_dict=True, digits=4),
        'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist(),
        'predictions': all_preds.tolist(),
        'true_labels': all_labels.tolist()
    }

    return metrics


def save_text_model(model, tokenizer, path=None):
    """Save trained text model and tokenizer."""
    if path is None:
        path = os.path.join(MODELS_DIR, 'text_model')
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, 'model.pt'))
    tokenizer.save_pretrained(path)
    print(f"Text model saved to {path}")


def load_text_model(path=None, device='cpu'):
    """Load trained text model and tokenizer."""
    if path is None:
        path = os.path.join(MODELS_DIR, 'text_model')
    
    # Auto-reassemble model.pt from parts if parts exist (for GitHub/Streamlit Cloud deployment)
    model_file = os.path.join(path, 'model.pt')
    if not os.path.exists(model_file):
        parts = sorted([p for p in os.listdir(path) if p.startswith('model.pt.part')])
        if parts:
            print(f"Reassembling model from {len(parts)} parts...")
            with open(model_file, 'wb') as outfile:
                for part in parts:
                    with open(os.path.join(path, part), 'rb') as infile:
                        outfile.write(infile.read())
            print("Model reassembled successfully.")

    tokenizer = DistilBertTokenizer.from_pretrained(path)
    model = TextSarcasmClassifier().to(device)
    model.load_state_dict(torch.load(model_file, map_location=device, weights_only=True))
    model.eval()
    return model, tokenizer
