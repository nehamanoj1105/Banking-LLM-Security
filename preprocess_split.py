import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# ----------------------------
# CONFIGURE YOUR PATHS
# ----------------------------
adversarial_sub = "/home/deepaksg/Downloads/Banking-LLM-Security-main/Dataset(Not Preprocessed)/Adversarial"
benign_sub = "/home/deepaksg/Downloads/Banking-LLM-Security-main/Dataset(Not Preprocessed)/Benign"

# ----------------------------
# HELPER: CLEAN TEXT FUNCTION
# ----------------------------
def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # remove URLs
    text = re.sub(r"\d+", "", text)  # remove numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # remove punctuation/symbols
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ----------------------------
# HELPER: GET TEXT COLUMN
# ----------------------------
def get_text_column(df):
    candidates = ["text", "utterance", "prompt", "query", "content", "instruction", "input"]
    for col in df.columns:
        if col.lower() in candidates:
            return col
    return df.columns[0]  # fallback

# ----------------------------
# LOAD BENIGN DATA
# ----------------------------
benign_df = pd.read_csv(os.path.join(benign_sub, "banking77_combined.csv"))
benign_text_col = get_text_column(benign_df)
benign_df = benign_df.rename(columns={benign_text_col: "text"})
benign_df["text"] = benign_df["text"].apply(clean_text)
benign_df["label"] = 0

# ----------------------------
# LOAD ADVERSARIAL DATA
# ----------------------------
adv_dfs = []
for f in os.listdir(adversarial_sub):
    if f.endswith(".csv"):
        try:
            temp = pd.read_csv(os.path.join(adversarial_sub, f))
            text_col = get_text_column(temp)
            temp = temp.rename(columns={text_col: "text"})
            temp["text"] = temp["text"].apply(clean_text)
            temp["label"] = 1
            adv_dfs.append(temp[["text", "label"]])
        except Exception as e:
            print(f"Error loading {f}: {e}")

adversarial_df = pd.concat(adv_dfs, ignore_index=True)

# ----------------------------
# MERGE BOTH
# ----------------------------
combined_df = pd.concat([benign_df[["text", "label"]], adversarial_df], ignore_index=True)
combined_df = combined_df.drop_duplicates(subset=["text"]).reset_index(drop=True)

print("Final dataset shape:", combined_df.shape)
print(combined_df.head())

# ----------------------------
# TRAIN/TEST SPLIT
# ----------------------------
train_df, test_df = train_test_split(combined_df, test_size=0.2, stratify=combined_df["label"], random_state=42)

print("Train size:", train_df.shape, " Test size:", test_df.shape)

# ----------------------------
# TF-IDF VECTORIZATION
# ----------------------------
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))  # unigrams + bigrams
X_train = vectorizer.fit_transform(train_df["text"])
X_test = vectorizer.transform(test_df["text"])

y_train = train_df["label"].values
y_test = test_df["label"].values

print("TF-IDF train shape:", X_train.shape)
print("TF-IDF test shape:", X_test.shape)
print(combined_df.sample(5))
print(combined_df['label'].value_counts())
# ----------------------------
# SAVE RAW DATA
# ----------------------------
train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)
print("✅ Saved train.csv and test.csv")

# ----------------------------
# SAVE TF-IDF FEATURES + LABELS
# ----------------------------
import joblib

joblib.dump(X_train, "X_train.pkl")
joblib.dump(X_test, "X_test.pkl")
joblib.dump(y_train, "y_train.pkl")
joblib.dump(y_test, "y_test.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("✅ Saved X_train.pkl, X_test.pkl, y_train.pkl, y_test.pkl, vectorizer.pkl")
