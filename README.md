#  Fake News Detector — NLP Text Classifier

A binary text classification system that detects fake vs. real news statements using classical NLP techniques. Built as part of my application to the **Amazon ML Summer School 2026**.

---

##  Problem Statement

Misinformation spreads faster than corrections. This project trains a machine learning model to classify political statements as **FAKE** or **REAL** based on their text content alone — no metadata, no speaker info, just the words.

---

## Dataset

**LIAR Dataset** — Wang (2017), ACL  
- 12,836 short political statements scraped from PolitiFact.com  
- Originally 6-class; binarized to **FAKE** (pants-fire, false, barely-true) vs **REAL** (half-true, mostly-true, true)  
- Size: ~2.5 MB  
- Source: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip

| Split | Samples |
|-------|---------|
| Train | 10,240  |
| Valid | 1,284   |
| Test  | 1,267   |

---

## ⚙️ Approach

```
Raw Text
   ↓
Preprocessing (lowercase, remove punctuation, stopword removal)
   ↓
TF-IDF Vectorisation (unigrams + bigrams, 50K features)
   ↓
Logistic Regression Classifier
   ↓
Evaluation (Accuracy, F1, Precision, Recall)
```

---

## Results

| Model | Accuracy | F1 Score (weighted) |
|---|---|---|
| **TF-IDF + Logistic Regression** | **61.48%** | **0.60** |
| TF-IDF + Naive Bayes (baseline) | 61.17% | 0.60 |

> **Context:** The LIAR dataset is a known hard benchmark. Even fine-tuned BERT models achieve only ~68–70% on it. A classical ML pipeline at 61.48% is a strong result for a from-scratch approach with no deep learning.

---

##  Sample Predictions

| Headline | Prediction | Confidence |
|---|---|---|
| "Scientists confirm vaccines cause autism in children" | ❌ FAKE | 61.0% |
| "NASA announces new mission to explore Jupiter's moons" | ✅ REAL | 51.7% |
| "Government secretly putting chemicals in drinking water" | ❌ FAKE | 50.5% |
| "India launches record number of satellites in single mission" | ✅ REAL | 70.2% |

---

##  Run It Yourself

### Option 1: Google Colab (Recommended)
Open `Fake_News_Detector_LIAR.ipynb` directly in Colab — no setup needed.

### Option 2: Local
```bash
pip install scikit-learn pandas nltk
python fake_news_detector.py
```

---

## Tech Stack

- **Python 3.x**
- **scikit-learn** — TF-IDF, Logistic Regression, Naive Bayes, evaluation metrics
- **pandas** — data loading and manipulation
- **NLTK** — stopword removal
- **Pipeline API** — clean, reproducible ML workflow

---

## Files

| File | Description |
|---|---|
| `fake_news_detector.py` | Full pipeline script (data → preprocessing → training → evaluation) |
| `Fake_News_Detector_LIAR.ipynb` | Google Colab notebook version |
| `README.md` | This file |

---

## Future Work

- Fine-tune a BERT / DistilBERT model for higher accuracy
- Add speaker credibility features (historical false-claim counts)
- Build a simple web demo using Flask or Streamlit
- Experiment with ensemble methods (LR + NB + SVM voting)

---

## Reference

Wang, W. Y. (2017). *"Liar, Liar Pants on Fire": A New Benchmark Dataset for Fake News Detection*. ACL 2017.  
https://aclanthology.org/P17-2067/

---

*Built by Siddharth — B.Tech CSE, USTM | Full-Stack & Blockchain Developer*
