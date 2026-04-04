import numpy as np
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import load_data, preprocess

# ── Paths ───────────────────────────────────────────────────────────────────
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)


# ── Evaluation helper ────────────────────────────────────────────────────────
def evaluate(name, y_true, y_pred):
    print(f"\n{'='*50}")
    print(f"  {name} Results")
    print(f"{'='*50}")
    print(f"  Accuracy  : {accuracy_score(y_true, y_pred):.4f}")
    print(f"  F1 Score  : {f1_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"\nClassification Report:\n")
    print(classification_report(y_true, y_pred,
                                target_names=['Normal', 'Attack']))
    print(f"Confusion Matrix:\n{confusion_matrix(y_true, y_pred)}")


# ── Model 1: Random Forest ───────────────────────────────────────────────────
def train_random_forest(X_train, y_train, X_test, y_test):
    print("\n[1/3] Training Random Forest...")

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    evaluate("Random Forest", y_test, y_pred)

    path = os.path.join(MODEL_DIR, 'rf_model.pkl')
    with open(path, 'wb') as f:
        pickle.dump(rf, f)
    print(f"  Saved → {path}")
    return rf


# ── Model 2: XGBoost ────────────────────────────────────────────────────────
def train_xgboost(X_train, y_train, X_test, y_test):
    print("\n[2/3] Training XGBoost...")

    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        verbosity=0
    )
    xgb.fit(X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False)
    y_pred = xgb.predict(X_test)
    evaluate("XGBoost", y_test, y_pred)

    path = os.path.join(MODEL_DIR, 'xgb_model.pkl')
    with open(path, 'wb') as f:
        pickle.dump(xgb, f)
    print(f"  Saved → {path}")
    return xgb


# ── Model 3: LSTM ────────────────────────────────────────────────────────────
def train_lstm(X_train, y_train, X_test, y_test):
    print("\n[3/3] Training LSTM...")

    # Reshape for LSTM: (samples, timesteps, features)
    X_train_lstm = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test_lstm  = X_test.reshape(X_test.shape[0],  1, X_test.shape[1])
    from tensorflow.keras.layers import Input
    model = Sequential([
        Input(shape=(1, X_train.shape[1])),
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1,  activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    early_stop = EarlyStopping(monitor='val_loss',
                               patience=3,
                               restore_best_weights=True)

    model.fit(
        X_train_lstm, y_train,
        epochs=15,
        batch_size=512,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )

    # Evaluate
    y_pred_prob = model.predict(X_test_lstm, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    evaluate("LSTM", y_test, y_pred)

    path = os.path.join(MODEL_DIR, 'lstm_model.keras')
    model.save(path)
    print(f"  Saved → {path}")
    return model


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Loading and preprocessing data...")
    df_train, df_test = load_data()
    X_train, X_test, y_train, y_test, scaler, encoders = preprocess(
        df_train, df_test, mode='binary', save_scaler=True
    )

    train_random_forest(X_train, y_train, X_test, y_test)
    train_xgboost(X_train, y_train, X_test, y_test)
    train_lstm(X_train, y_train, X_test, y_test)

    print("\nAll models trained and saved successfully!")
    print(f"Models saved in: {os.path.abspath(MODEL_DIR)}/")