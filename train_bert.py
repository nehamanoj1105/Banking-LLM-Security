import os
import warnings
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from tqdm import tqdm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

# 1. Load CSV Data
train_df = pd.read_csv("Dataset(Preprocessed)/train.csv")
test_df = pd.read_csv("Dataset(Preprocessed)/test.csv")
X_train = train_df['text'].tolist()
y_train = train_df['label'].tolist()
X_test = test_df['text'].tolist()
y_test = test_df['label'].tolist()

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# 2. Load BERT Tokenizer & Model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(set(y_train))
)

# 3. Tokenize Data (smaller max_length)
train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=64)
test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=64)

# 4. Dataset Class
class BankingDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

train_dataset = BankingDataset(train_encodings, y_train)
test_dataset = BankingDataset(test_encodings, y_test)

# 5. Setup Training (CPU, small batch)
device = torch.device("cpu")  # Force CPU
model.to(device)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
optimizer = AdamW(model.parameters(), lr=5e-5)

# 6. Training Loop (1 epoch)
model.train()
epochs = 1
for epoch in range(epochs):
    loop = tqdm(train_loader, leave=True)
    for batch in loop:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        loop.set_description(f'Epoch {epoch+1}')
        loop.set_postfix(loss=loss.item())

# 7. Evaluation
model.eval()
preds = []
true_labels = []
test_loader = DataLoader(test_dataset, batch_size=4)
for batch in test_loader:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
    preds.extend(predictions.cpu().numpy())
    true_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(true_labels, preds)
print(f"\nTest Accuracy: {accuracy:.4f}")
