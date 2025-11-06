import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from transformers import (BertTokenizer, BertForSequenceClassification,
                          RobertaTokenizer, RobertaForSequenceClassification,
                          DistilBertTokenizer, DistilBertForSequenceClassification)
from torch.optim import AdamW
from tqdm import tqdm
import shap
import numpy as np

# Full data
train_df = pd.read_csv("Dataset(Preprocessed)/train.csv")
test_df = pd.read_csv("Dataset(Preprocessed)/test.csv")

le = LabelEncoder()
train_df['label'] = le.fit_transform(train_df['label'])
test_df['label'] = le.transform(test_df['label'])

X_train, y_train = train_df['text'].tolist(), train_df['label'].tolist()
X_test, y_test = test_df['text'].tolist(), test_df['label'].tolist()
num_labels = len(le.classes_)

# Models & Tokenizers
bert_tok = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)

roberta_tok = RobertaTokenizer.from_pretrained("roberta-base")
roberta_model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=num_labels)

distil_tok = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
distil_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)

models = [bert_model, roberta_model, distil_model]
tokenizers = [bert_tok, roberta_tok, distil_tok]

# Tokenize
encodings = [tok(X_train, truncation=True, padding=True, max_length=64, return_tensors="pt") for tok in tokenizers]
test_encodings = [tok(X_test, truncation=True, padding=True, max_length=64, return_tensors="pt") for tok in tokenizers]

class EnsDataset(Dataset):
    def __init__(self, encodings_list, labels):
        self.encodings_list = encodings_list
        self.labels = labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return [{k: v[idx] for k, v in enc.items()} for enc in self.encodings_list], torch.tensor(self.labels[idx])

train_dataset = EnsDataset(encodings, y_train)
test_dataset = EnsDataset(test_encodings, y_test)

device = torch.device("cpu")
for m in models: m.to(device)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
optimizers = [AdamW(m.parameters(), lr=5e-5) for m in models]

# Train 1 epoch
for batch in tqdm(train_loader, desc="Training"):
    batch_enc, labels = batch
    labels = labels.to(device)
    for enc, opt, model in zip(batch_enc, optimizers, models):
        opt.zero_grad()
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc, labels=labels)
        out.loss.backward()
        opt.step()

# Inference
all_preds = []
test_loader = DataLoader(test_dataset, batch_size=4)
with torch.no_grad():
    for batch in test_loader:
        batch_enc, _ = batch
        batch_logits = []
        for enc, model in zip(batch_enc, models):
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            batch_logits.append(out.logits.cpu())
        avg_logits = torch.stack(batch_logits).mean(0)
        pred = torch.argmax(avg_logits, dim=-1).numpy()
        all_preds.extend(pred)

acc = accuracy_score(y_test, all_preds)
print(f"Ensemble Test Accuracy: {acc:.4f}")

# === XAI: SHAP on Ensemble ===
def ensemble_predict(texts):
    logits_list = []
    for tok, model in zip(tokenizers, models):
        enc = tok(texts, truncation=True, padding=True, max_length=64, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**enc).logits.cpu().numpy()
        logits_list.append(logits)
    return np.mean(logits_list, axis=0)

# Background (50 samples)
background = X_train[::len(X_train)//50]
explainer = shap.Explainer(ensemble_predict, background, max_evals=1000)

# Explain function
def explain_input(text):
    shap_values = explainer([text])
    pred = np.argmax(ensemble_predict([text]))
    label = "Malicious" if pred == 1 else "Benign"
    print(f"Prediction: {label}")
    shap.plots.text(shap_values[0])  # Run in notebook

# Example
