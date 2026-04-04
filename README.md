# Network Intrusion Detection System (IDS)

A machine learning-based Network Intrusion Detection System trained on the **NSL-KDD dataset** using an ensemble of Random Forest, XGBoost, and LSTM models with a real-time Streamlit dashboard.

---

## Dashboard Preview

> Real-time traffic analysis with color-coded severity alerts, confidence scoring, and model comparison across 3 ML models.

---

## Features

- **3-Model Ensemble** — Random Forest + XGBoost + LSTM majority voting
- **Real-time Detection** — Analyze network traffic sample by sample
- **Severity Levels** — CRITICAL / HIGH / MEDIUM / LOW based on confidence score
- **Interactive Dashboard** — Live charts, KPI metrics, model comparison
- **Alert Logging** — JSON structured logs saved to `logs/ids_alerts.log`
- **CSV Export** — Download detection results with one click
- **NSL-KDD Dataset** — Industry standard benchmark, superior to KDD Cup 99

---

## Project Structure

```
network-ids/
│
├── data/
│   ├── KDDTrain+.txt              # 125,973 training samples
│   ├── KDDTest+.txt               # 22,544 test samples
│   ├── KDDTrain+_20Percent.txt
│   └── KDDTest-21.txt
│
├── models/
│   ├── rf_model.pkl               # Trained Random Forest
│   ├── xgb_model.pkl              # Trained XGBoost
│   ├── lstm_model.keras           # Trained LSTM (TensorFlow)
│   └── scaler.pkl                 # StandardScaler
│
├── src/
│   ├── preprocess.py              # Data loading, encoding, scaling
│   ├── train.py                   # Model training & evaluation
│   ├── predict.py                 # Inference engine (single + batch)
│   ├── alert_engine.py            # Alert generation & severity logging
│   └── dashboard.py               # Streamlit web dashboard
│
├── logs/
│   └── ids_alerts.log             # JSON structured alert logs
│
├── notebooks/
│   └── eda.ipynb                  # Exploratory data analysis
│
├── tests/
│   └── test_preprocess.py         # Unit tests
│
├── requirements.txt
└── README.md
```

---

## Model Performance

| Model | Accuracy | F1 Score | Attack Recall |
|---|---|---|---|
| Random Forest | 77.17% | 76.86% | 62% |
| XGBoost | 80.56% | 80.46% | 68% |
| LSTM | 78.23% | 78.01% | 64% |
| **Ensemble (Majority Vote)** | **81.0%** | **80.5%** | **70%** |

> Note: Lower test accuracy compared to training is expected on NSL-KDD — the test set contains novel attack types not present in training data, making it a realistic evaluation benchmark.

---

## Dataset

**NSL-KDD** — An improved version of the KDD Cup 1999 dataset.

| File | Samples | Description |
|---|---|---|
| KDDTrain+ | 125,973 | Full training set |
| KDDTest+ | 22,544 | Full test set (includes novel attacks) |

- **41 features** covering network connection properties
- **Binary classification**: Normal (0) vs Attack (1)
- **Attack categories**: DoS, Probe, R2L, U2R

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| ML Models | scikit-learn, XGBoost, TensorFlow/Keras |
| Dashboard | Streamlit, Plotly |
| Data | pandas, numpy |
| Logging | Python logging (JSON format) |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/network-ids.git
cd network-ids
```

### 2. Create virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NSL-KDD dataset

Download from [Kaggle NSL-KDD](https://www.kaggle.com/datasets/hassan06/nslkdd) and place files in `data/` folder:

```
data/
├── KDDTrain+.txt
├── KDDTest+.txt
├── KDDTrain+_20Percent.txt
└── KDDTest-21.txt
```

---

## Usage

### Step 1 — Preprocess data

```bash
python src/preprocess.py
```

### Step 2 — Train all models

```bash
python src/train.py
```

### Step 3 — Test predictions

```bash
python src/predict.py
```

### Step 4 — Run alert engine

```bash
python src/alert_engine.py
```

### Step 5 — Launch dashboard

```bash
streamlit run src/dashboard.py
```

Open browser at `http://localhost:8501`

---

## How It Works

### Preprocessing Pipeline
1. Load NSL-KDD with 43 column names
2. Drop difficulty column
3. Label encode categorical features (`protocol_type`, `service`, `flag`)
4. Binary encode labels (0=Normal, 1=Attack)
5. StandardScaler normalization → save `scaler.pkl`

### Ensemble Prediction
Each traffic sample is passed through all 3 models independently. The final prediction uses **majority voting** — if 2 or more models agree on ATTACK, the sample is flagged.

### Severity Scoring
| Confidence | Severity |
|---|---|
| >= 90% | CRITICAL |
| >= 70% | HIGH |
| >= 50% | MEDIUM |
| < 50% | LOW |

---

## Requirements

Create `requirements.txt` with:

```
pandas
numpy
scikit-learn
xgboost
tensorflow
streamlit
plotly
```

---

## Author

**Surisetti Manoj Akash**  
B.Tech — Cyber Security  
[LinkedIn](www.linkedin.com/in/manoj-akash-surisetti-616a032b9) | [GitHub](https://github.com/Surisetti-4002)

---

## References

- NSL-KDD Dataset — Canadian Institute for Cybersecurity, University of New Brunswick