import os
import json
import logging
from datetime import datetime

# ── Log file setup ────────────────────────────────────────────────────────────
LOG_DIR  = 'logs'
LOG_FILE = os.path.join(LOG_DIR, 'ids_alerts.log')
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('IDS_AlertEngine')

# ── Severity thresholds ───────────────────────────────────────────────────────
# Based on ensemble confidence score
SEVERITY_LEVELS = {
    'CRITICAL' : 90,   # confidence >= 90%
    'HIGH'     : 70,   # confidence >= 70%
    'MEDIUM'   : 50,   # confidence >= 50%
    'LOW'      : 0,    # confidence >= 0%
}


# ── Severity calculator ───────────────────────────────────────────────────────
def get_severity(confidence: float) -> str:
    """
    Returns severity level based on attack confidence score.
    confidence should be 0-100 float.
    """
    for level, threshold in SEVERITY_LEVELS.items():
        if confidence >= threshold:
            return level
    return 'LOW'


# ── Alert colors for console ──────────────────────────────────────────────────
SEVERITY_COLORS = {
    'CRITICAL' : '\033[91m',   # Red
    'HIGH'     : '\033[93m',   # Yellow
    'MEDIUM'   : '\033[94m',   # Blue
    'LOW'      : '\033[92m',   # Green
    'NORMAL'   : '\033[92m',   # Green
    'RESET'    : '\033[0m',
}


# ── Core alert function ───────────────────────────────────────────────────────
def trigger_alert(prediction: dict, sample_id: int = None):
    """
    Trigger an alert based on prediction result.

    Parameters
    ----------
    prediction : dict returned by predict_single()
    sample_id  : optional identifier for the traffic sample
    """
    ensemble    = prediction['ensemble']
    result      = ensemble['prediction']       # 'NORMAL' or 'ATTACK'
    confidence  = float(ensemble['confidence'].replace('%', ''))

    timestamp   = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    sid         = f"Sample-{sample_id}" if sample_id else "Unknown"

    if result == 'ATTACK':
        severity = get_severity(confidence)
        alert    = build_alert(sid, severity, confidence, prediction, timestamp)
        log_alert(alert)
        print_alert(alert)
        return alert

    else:
        # Normal traffic — just log quietly
        msg = f"[{timestamp}] {sid} | NORMAL traffic | Confidence: {confidence:.2f}%"
        logger.info(msg)
        color = SEVERITY_COLORS['NORMAL']
        reset = SEVERITY_COLORS['RESET']
        print(f"{color}[NORMAL]  {sid} | Confidence: {confidence:.2f}%{reset}")
        return None


# ── Alert builder ─────────────────────────────────────────────────────────────
def build_alert(sid, severity, confidence, prediction, timestamp):
    return {
        'timestamp'  : timestamp,
        'sample_id'  : sid,
        'severity'   : severity,
        'confidence' : confidence,
        'models'     : {
            'random_forest' : prediction['random_forest'],
            'xgboost'       : prediction['xgboost'],
            'lstm'          : prediction['lstm'],
        },
        'ensemble'   : prediction['ensemble'],
    }


# ── Logger ────────────────────────────────────────────────────────────────────
def log_alert(alert: dict):
    """Write alert to log file as JSON line."""
    log_entry = json.dumps(alert)
    if alert['severity'] == 'CRITICAL':
        logger.critical(log_entry)
    elif alert['severity'] == 'HIGH':
        logger.error(log_entry)
    elif alert['severity'] == 'MEDIUM':
        logger.warning(log_entry)
    else:
        logger.info(log_entry)


# ── Console printer ───────────────────────────────────────────────────────────
def print_alert(alert: dict):
    """Print colored alert to console."""
    color = SEVERITY_COLORS.get(alert['severity'], '')
    reset = SEVERITY_COLORS['RESET']
    sep   = '=' * 55

    print(f"\n{color}{sep}")
    print(f"  [{alert['severity']}] INTRUSION DETECTED")
    print(f"{sep}")
    print(f"  Time       : {alert['timestamp']}")
    print(f"  Sample     : {alert['sample_id']}")
    print(f"  Confidence : {alert['confidence']:.2f}%")
    print(f"  Models     :")
    for model, result in alert['models'].items():
        print(f"    {model:15s} → {result['prediction']:6s} "
              f"({result['confidence']})")
    print(f"  Ensemble   → {alert['ensemble']['prediction']} "
          f"({alert['ensemble']['confidence']})")
    print(f"{sep}{reset}\n")


# ── Summary reporter ──────────────────────────────────────────────────────────
def print_summary(alerts: list, total: int):
    """Print detection summary after batch processing."""
    attacks  = len(alerts)
    normal   = total - attacks
    rate     = (attacks / total * 100) if total > 0 else 0

# NEW (fixed) - replace with this
    from collections import Counter
    severity_counts = Counter(a['severity'] for a in alerts)
    critical = severity_counts.get('CRITICAL', 0)
    high     = severity_counts.get('HIGH',     0)
    medium   = severity_counts.get('MEDIUM',   0)
    low      = severity_counts.get('LOW',      0)

    print("\n" + "=" * 55)
    print("  IDS DETECTION SUMMARY")
    print("=" * 55)
    print(f"  Total Samples  : {total}")
    print(f"  Normal Traffic : {normal}")
    print(f"  Attacks Found  : {attacks}  ({rate:.1f}%)")
    print(f"\n  Severity Breakdown:")
    print(f"    CRITICAL : {critical}")
    print(f"    HIGH     : {high}")
    print(f"    MEDIUM   : {medium}")
    print(f"    LOW      : {low}")
    print("=" * 55)
    print(f"  Log saved → {os.path.abspath(LOG_FILE)}")
    print("=" * 55 + "\n")


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    import numpy as np
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from src.preprocess import load_data, preprocess
    from src.predict    import load_all_models, predict_single

    # Load models
    scaler, rf_model, xgb_model, lstm_model = load_all_models()

    # Load test data
    df_train, df_test = load_data()
    X_train, X_test, y_train, y_test, _, _ = preprocess(
        df_train, df_test, mode='binary', save_scaler=False
    )

    print("\nRunning IDS Alert Engine on 10 test samples...\n")

    alerts = []
    total  = 10

    for i in range(total):
        features   = X_test[i]
        prediction = predict_single(features, scaler,
                                    rf_model, xgb_model, lstm_model)
        alert = trigger_alert(prediction, sample_id=i + 1)
        if alert:
            alerts.append(alert)

    print_summary(alerts, total)