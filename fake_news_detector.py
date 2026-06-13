# ============================================================
# Fake News / Misinformation Classifier
# Author: Siddharth
# Amazon ML Summer School 2026 - NLP Project
# ============================================================
# Stack: Python, scikit-learn, pandas, NLTK
# Dataset: LIAR Dataset (Wang, 2017) — ~12,800 labeled statements
# Task: Binary text classification (True vs Fake news)
# ============================================================

# ----- STEP 0: Install dependencies (run this in Colab) -----
# !pip install scikit-learn pandas nltk

import pandas as pd
import numpy as np
import nltk
import re
import urllib.request
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from sklearn.pipeline import Pipeline

nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# ============================================================
# STEP 1: Download the LIAR Dataset
# ============================================================
# LIAR Dataset: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
# For Colab, we use a mirror via GitHub (TSV format)

TRAIN_URL = "https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/train.tsv"
TEST_URL  = "https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/test.tsv"
VALID_URL = "https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/valid.tsv"

COLUMNS = [
    "id", "label", "statement", "subject", "speaker",
    "speaker_job", "state", "party", "barely_true_count",
    "false_count", "half_true_count", "mostly_true_count",
    "pants_on_fire_count", "context"
]

def load_liar(url):
    df = pd.read_csv(url, sep='\t', header=None, names=COLUMNS)
    return df

print("📥 Downloading LIAR dataset...")
train_df = load_liar(TRAIN_URL)
test_df  = load_liar(TEST_URL)
valid_df = load_liar(VALID_URL)

print(f"   Train: {len(train_df)} samples")
print(f"   Valid: {len(valid_df)} samples")
print(f"   Test:  {len(test_df)} samples")
print(f"   Labels found: {train_df['label'].unique()}")

# ============================================================
# STEP 2: Binarize Labels
# ============================================================
# LIAR has 6 labels: pants-fire, false, barely-true,
#                    half-true, mostly-true, true
# We simplify to: FAKE (0) vs REAL (1)

FAKE_LABELS = {"pants-fire", "false", "barely-true"}
REAL_LABELS = {"half-true", "mostly-true", "true"}

def binarize(label):
    if label in FAKE_LABELS:
        return 0   # FAKE
    elif label in REAL_LABELS:
        return 1   # REAL
    return None

for df in [train_df, test_df, valid_df]:
    df["binary_label"] = df["label"].apply(binarize)

# Drop any rows with unrecognized labels
train_df = train_df.dropna(subset=["binary_label"])
test_df  = test_df.dropna(subset=["binary_label"])
valid_df = valid_df.dropna(subset=["binary_label"])

print(f"\n🏷️  Label distribution (train):")
print(train_df["binary_label"].value_counts().rename({0: "FAKE", 1: "REAL"}))

# ============================================================
# STEP 3: Text Preprocessing
# ============================================================

STOP_WORDS = set(stopwords.words('english'))

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)          # remove non-alpha
    text = re.sub(r"\s+", " ", text).strip()       # normalize whitespace
    tokens = [w for w in text.split() if w not in STOP_WORDS]
    return " ".join(tokens)

print("\n🔧 Preprocessing text...")
train_df["clean"] = train_df["statement"].apply(preprocess)
test_df["clean"]  = test_df["statement"].apply(preprocess)
valid_df["clean"] = valid_df["statement"].apply(preprocess)

X_train, y_train = train_df["clean"], train_df["binary_label"].astype(int)
X_test,  y_test  = test_df["clean"],  test_df["binary_label"].astype(int)
X_valid, y_valid = valid_df["clean"], valid_df["binary_label"].astype(int)

# ============================================================
# STEP 4: Build & Train Models
# ============================================================

print("\n🤖 Training models...")

# --- Model 1: TF-IDF + Logistic Regression ---
lr_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),   # unigrams + bigrams
        max_features=50000,
        sublinear_tf=True
    )),
    ("clf", LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs"
    ))
])

lr_pipeline.fit(X_train, y_train)
lr_preds = lr_pipeline.predict(X_test)

lr_acc = accuracy_score(y_test, lr_preds)
lr_f1  = f1_score(y_test, lr_preds, average="weighted")

print(f"\n📊 Logistic Regression Results:")
print(f"   Accuracy : {lr_acc:.4f} ({lr_acc*100:.2f}%)")
print(f"   F1 Score : {lr_f1:.4f}")
print(f"\n   Classification Report:")
print(classification_report(y_test, lr_preds, target_names=["FAKE", "REAL"]))

# --- Model 2: TF-IDF + Naive Bayes (baseline comparison) ---
nb_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=50000,
        sublinear_tf=True
    )),
    ("clf", MultinomialNB(alpha=0.1))
])

nb_pipeline.fit(X_train, y_train)
nb_preds = nb_pipeline.predict(X_test)

nb_acc = accuracy_score(y_test, nb_preds)
nb_f1  = f1_score(y_test, nb_preds, average="weighted")

print(f"\n📊 Naive Bayes Results (baseline):")
print(f"   Accuracy : {nb_acc:.4f} ({nb_acc*100:.2f}%)")
print(f"   F1 Score : {nb_f1:.4f}")

# ============================================================
# STEP 5: Results Summary
# ============================================================

print("\n" + "="*55)
print("  FINAL RESULTS SUMMARY")
print("="*55)
print(f"  {'Model':<30} {'Accuracy':>10} {'F1 Score':>10}")
print(f"  {'-'*50}")
print(f"  {'TF-IDF + Logistic Regression':<30} {lr_acc*100:>9.2f}% {lr_f1:>10.4f}")
print(f"  {'TF-IDF + Naive Bayes':<30} {nb_acc*100:>9.2f}% {nb_f1:>10.4f}")
print("="*55)

# ============================================================
# STEP 6: Predict on Custom Headlines
# ============================================================

print("\n🔍 Testing on custom headlines:")

custom_headlines = [
    "Scientists confirm vaccines cause autism in children",
    "NASA announces new mission to explore Jupiter's moons",
    "Government secretly putting chemicals in drinking water",
    "India launches record number of satellites in single mission",
]

for headline in custom_headlines:
    clean = preprocess(headline)
    pred = lr_pipeline.predict([clean])[0]
    prob = lr_pipeline.predict_proba([clean])[0]
    label = "✅ REAL" if pred == 1 else "❌ FAKE"
    confidence = max(prob) * 100
    print(f"  {label} ({confidence:.1f}%) — \"{headline}\"")

print("\n✅ Project complete! Use the metrics above to fill your Amazon ML form.")
