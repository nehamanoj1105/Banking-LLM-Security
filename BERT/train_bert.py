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

# Encode labels
le = LabelEncoder()
train_df['label_enc'] = le.fit_transform(train_df['label'])
test_df['label_enc'] = le.transform(test_df['label'])

# 2. Subsample training data (20% stratified)
train_df = train_df.groupby('label_enc', group_keys=False).apply(
    lambda x: x.sample(frac=0.2, random_state=42)
).reset_index(drop=True)

X_train = train_df['text'].tolist()
y_train = train_df['label_enc'].tolist()
X_test = test_df['text'].tolist()
y_test = test_df['label_enc'].tolist()

# 3. Load BERT Tokenizer & Model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(le.classes_)
)

# 4. Tokenize (short length)
train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=64)
test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=64)

# 5. Dataset Class
class BankingDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

train_dataset = BankingDataset(train_encodings, y_train)
test_dataset = BankingDataset(test_encodings, y_test)

# 6. Setup (CPU, tiny batch)
device = torch.device("cpu")
model.to(device)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
optimizer = AdamW(model.parameters(), lr=5e-5)

# 7. Train (1 epoch)
model.train()
for batch in tqdm(train_loader, desc="Training"):
    optimizer.zero_grad()
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    outputs.loss.backward()
    optimizer.step()

# 8. Evaluate
model.eval()
preds, true_labels = [], []
test_loader = DataLoader(test_dataset, batch_size=4)
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        preds.extend(torch.argmax(outputs.logits, dim=-1).cpu().numpy())
        true_labels.extend(batch['labels'].cpu().numpy())

accuracy = accuracy_score(true_labels, preds)
print(f"Test Accuracy: {accuracy:.4f}")
