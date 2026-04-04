import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ── Column definitions ──────────────────────────────────────────────────────
COL_NAMES = [
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
    'dst_host_rerror_rate','dst_host_srv_rerror_rate',
    'label','difficulty'
]

CATEGORICAL_COLS = ['protocol_type', 'service', 'flag']

# Attack category mapping (multi-class)
ATTACK_MAP = {
    'normal': 'normal',
    # DoS
    'back':'dos','land':'dos','neptune':'dos','pod':'dos','smurf':'dos',
    'teardrop':'dos','apache2':'dos','udpstorm':'dos','processtable':'dos',
    'worm':'dos','mailbomb':'dos',
    # Probe
    'satan':'probe','ipsweep':'probe','nmap':'probe','portsweep':'probe',
    'mscan':'probe','saint':'probe',
    # R2L
    'guess_passwd':'r2l','ftp_write':'r2l','imap':'r2l','phf':'r2l',
    'multihop':'r2l','warezmaster':'r2l','warezclient':'r2l','spy':'r2l',
    'xlock':'r2l','xsnoop':'r2l','snmpguess':'r2l','snmpgetattack':'r2l',
    'httptunnel':'r2l','sendmail':'r2l','named':'r2l',
    # U2R
    'buffer_overflow':'u2r','loadmodule':'u2r','rootkit':'u2r','perl':'u2r',
    'sqlattack':'u2r','xterm':'u2r','ps':'u2r',
}


def load_data(train_path='data/nsl-kdd/KDDTrain+.txt',
              test_path='data/nsl-kdd/KDDTest+.txt'):
    """Load raw NSL-KDD files and attach column names."""
    df_train = pd.read_csv(train_path, header=None, names=COL_NAMES)
    df_test  = pd.read_csv(test_path,  header=None, names=COL_NAMES)

    print(f"Train shape : {df_train.shape}")
    print(f"Test  shape : {df_test.shape}")
    return df_train, df_test


def preprocess(df_train, df_test, mode='binary', save_scaler=True):
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    mode : 'binary'     → label = 0 (normal) / 1 (attack)
           'multiclass' → label = normal / dos / probe / r2l / u2r
    save_scaler : saves models/scaler.pkl when True
    """

    # 1. Drop difficulty column
    for df in [df_train, df_test]:
        df.drop(columns=['difficulty'], inplace=True, errors='ignore')

    # 2. Encode categorical features
    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        # Fit on combined to handle unseen categories in test
        combined = pd.concat([df_train[col], df_test[col]], axis=0)
        le.fit(combined)
        df_train[col] = le.transform(df_train[col])
        df_test[col]  = le.transform(df_test[col])
        encoders[col] = le

    # 3. Encode labels
    if mode == 'binary':
        df_train['label'] = df_train['label'].apply(
            lambda x: 0 if x == 'normal' else 1)
        df_test['label']  = df_test['label'].apply(
            lambda x: 0 if x == 'normal' else 1)

    elif mode == 'multiclass':
        df_train['label'] = df_train['label'].map(ATTACK_MAP).fillna('unknown')
        df_test['label']  = df_test['label'].map(ATTACK_MAP).fillna('unknown')
        le_label = LabelEncoder()
        combined_labels = pd.concat([df_train['label'], df_test['label']])
        le_label.fit(combined_labels)
        df_train['label'] = le_label.transform(df_train['label'])
        df_test['label']  = le_label.transform(df_test['label'])
        encoders['label'] = le_label

    # 4. Split features / labels
    X_train = df_train.drop(columns=['label'])
    y_train = df_train['label']
    X_test  = df_test.drop(columns=['label'])
    y_test  = df_test['label']

    # 5. Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # 6. Save scaler
    if save_scaler:
        os.makedirs('models', exist_ok=True)
        with open('models/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        print("Scaler saved → models/scaler.pkl")

    # 7. Print class distribution
    print(f"\nClass distribution (train):\n{y_train.value_counts()}")
    print(f"\nClass distribution (test):\n{y_test.value_counts()}")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, encoders


# ── Quick test ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    df_train, df_test = load_data()
    X_train, X_test, y_train, y_test, scaler, encoders = preprocess(
        df_train, df_test, mode='binary'
    )
    print(f"\nX_train : {X_train.shape}")
    print(f"X_test  : {X_test.shape}")
    print("Preprocessing complete ✓")