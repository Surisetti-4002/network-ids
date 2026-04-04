import numpy as np
import pickle
import os
import pandas as pd
FEATURE_COLS = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes',
    'land','wrong_fragment','urgent','hot','num_failed_logins','logged_in',
    'num_compromised','root_shell','su_attempted','num_root',
    'num_file_creations','num_shells','num_access_files','num_outbound_cmds',
    'is_host_login','is_guest_login','count','srv_count','serror_rate',
    'srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate',
    'diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
    'dst_host_serror_rate','dst_host_srv_serror_rate',
    'dst_host_rerror_rate','dst_host_srv_rerror_rate'
]
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.models import load_model

# ── Paths ────────────────────────────────────────────────────────────────────
MODEL_DIR = 'models'
SCALER_PATH  = os.path.join(MODEL_DIR, 'scaler.pkl')
RF_PATH      = os.path.join(MODEL_DIR, 'rf_model.pkl')
XGB_PATH     = os.path.join(MODEL_DIR, 'xgb_model.pkl')
LSTM_PATH    = os.path.join(MODEL_DIR, 'lstm_model.keras')


# ── Load all models once ─────────────────────────────────────────────────────
def load_all_models():
    print("Loading models...")

    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    with open(RF_PATH, 'rb') as f:
        rf_model = pickle.load(f)

    with open(XGB_PATH, 'rb') as f:
        xgb_model = pickle.load(f)

    lstm_model = load_model(LSTM_PATH)

    print("All models loaded successfully!")
    return scaler, rf_model, xgb_model, lstm_model


# ── Predict single sample ─────────────────────────────────────────────────────
def predict_single(features: np.ndarray, scaler, rf_model,
                   xgb_model, lstm_model):
    """
    Predict on a single network traffic sample.

    Parameters
    ----------
    features : np.ndarray of shape (41,) — raw unscaled feature vector

    Returns
    -------
    dict with predictions from all 3 models + ensemble vote
    """
    df = pd.DataFrame(features.reshape(1, -1), columns=FEATURE_COLS)
    features_scaled = scaler.transform(df)
    # Scale
    features_scaled = scaler.transform(features.reshape(1, -1))

    # RF prediction
    rf_pred  = rf_model.predict(features_scaled)[0]
    rf_prob  = rf_model.predict_proba(features_scaled)[0][1]

    # XGB prediction
    xgb_pred = xgb_model.predict(features_scaled)[0]
    xgb_prob = xgb_model.predict_proba(features_scaled)[0][1]

    # LSTM prediction
    lstm_input = features_scaled.reshape(1, 1, features_scaled.shape[1])
    lstm_prob  = lstm_model.predict(lstm_input, verbose=0)[0][0]
    lstm_pred  = int(lstm_prob > 0.5)

    # Ensemble vote (majority of 3 models)
    votes        = [rf_pred, xgb_pred, lstm_pred]
    ensemble_pred = int(sum(votes) >= 2)

    # Average confidence
    avg_confidence = round(float(np.mean([rf_prob, xgb_prob, lstm_prob])) * 100, 2)

    return {
        'random_forest' : {'prediction': label(rf_pred),  'confidence': f"{rf_prob*100:.2f}%"},
        'xgboost'       : {'prediction': label(xgb_pred), 'confidence': f"{xgb_prob*100:.2f}%"},
        'lstm'          : {'prediction': label(lstm_pred),'confidence': f"{lstm_prob*100:.2f}%"},
        'ensemble'      : {'prediction': label(ensemble_pred),
                           'confidence': f"{avg_confidence}%"},
    }


# ── Predict batch ─────────────────────────────────────────────────────────────
def predict_batch(X: np.ndarray, scaler, rf_model, xgb_model, lstm_model):
    """
    Predict on a batch of samples.
    Returns array of ensemble predictions (0=Normal, 1=Attack).
    """
    X_scaled    = scaler.transform(X)
    X_lstm      = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

    rf_preds    = rf_model.predict(X_scaled)
    xgb_preds   = xgb_model.predict(X_scaled)
    lstm_probs  = lstm_model.predict(X_lstm, verbose=0).flatten()
    lstm_preds  = (lstm_probs > 0.5).astype(int)

    # Majority vote across all 3
    ensemble = ((rf_preds + xgb_preds + lstm_preds) >= 2).astype(int)

    return {
        'rf_predictions'   : rf_preds,
        'xgb_predictions'  : xgb_preds,
        'lstm_predictions' : lstm_preds,
        'ensemble'         : ensemble,
    }


# ── Helper ────────────────────────────────────────────────────────────────────
def label(pred):
    return 'ATTACK' if pred == 1 else 'NORMAL'


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.preprocess import load_data, preprocess

    # Load models
    scaler, rf_model, xgb_model, lstm_model = load_all_models()

    # Load a few test samples
    df_train, df_test = load_data()
    X_train, X_test, y_train, y_test, _, _ = preprocess(
        df_train, df_test, mode='binary', save_scaler=False
    )

    print("\n--- Testing 5 samples from test set ---\n")
    for i in range(5):
        raw_features = X_test[i]
        result = predict_single(raw_features, scaler,
                                rf_model, xgb_model, lstm_model)
        actual = label(y_test.iloc[i])

        print(f"Sample {i+1}  |  Actual: {actual}")
        print(f"  RF       → {result['random_forest']['prediction']:6s}  "
              f"({result['random_forest']['confidence']})")
        print(f"  XGBoost  → {result['xgboost']['prediction']:6s}  "
              f"({result['xgboost']['confidence']})")
        print(f"  LSTM     → {result['lstm']['prediction']:6s}  "
              f"({result['lstm']['confidence']})")
        print(f"  Ensemble → {result['ensemble']['prediction']:6s}  "
              f"({result['ensemble']['confidence']})")
        print()