import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess   import load_data, preprocess
from src.predict      import load_all_models, predict_single
from src.alert_engine import trigger_alert, print_summary

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Network IDS Dashboard",
    page_icon  = "shield",
    layout     = "wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .critical { color: #ff4b4b; font-weight: bold; }
    .high     { color: #ffa500; font-weight: bold; }
    .medium   { color: #4b9dff; font-weight: bold; }
    .low      { color: #00cc66; font-weight: bold; }
    .normal   { color: #00cc66; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ── Load models (cached) ──────────────────────────────────────────────────────
@st.cache_resource
def get_models():
    return load_all_models()


@st.cache_data
def get_test_data():
    df_train, df_test = load_data()
    X_train, X_test, y_train, y_test, _, _ = preprocess(
        df_train, df_test, mode='binary', save_scaler=False
    )
    return X_test, y_test


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/shield.png", width=80)
st.sidebar.title("IDS Control Panel")
st.sidebar.markdown("---")

mode = st.sidebar.selectbox(
    "Select Mode",
    ["Live Detection", "Batch Analysis", "Log Viewer"]
)

n_samples = st.sidebar.slider(
    "Number of samples to analyze",
    min_value=10, max_value=500, value=50, step=10
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Models Loaded**")
st.sidebar.success("Random Forest")
st.sidebar.success("XGBoost")
st.sidebar.success("LSTM")


# ── Main title ────────────────────────────────────────────────────────────────
st.title("Network Intrusion Detection System")
st.markdown("Real-time traffic analysis using Random Forest, XGBoost & LSTM ensemble")
st.markdown("---")


# ── Helper: run detection ─────────────────────────────────────────────────────
def run_detection(X_test, y_test, n):
    scaler, rf, xgb, lstm = get_models()
    results = []

    progress = st.progress(0)
    status   = st.empty()

    for i in range(n):
        features   = X_test[i]
        prediction = predict_single(features, scaler, rf, xgb, lstm)
        ensemble   = prediction['ensemble']
        confidence = float(ensemble['confidence'].replace('%', ''))
        actual     = 'ATTACK' if y_test.iloc[i] == 1 else 'NORMAL'

        severity = 'NORMAL'
        if ensemble['prediction'] == 'ATTACK':
            if confidence >= 90:   severity = 'CRITICAL'
            elif confidence >= 70: severity = 'HIGH'
            elif confidence >= 50: severity = 'MEDIUM'
            else:                  severity = 'LOW'

        results.append({
            'sample'     : i + 1,
            'actual'     : actual,
            'predicted'  : ensemble['prediction'],
            'confidence' : confidence,
            'severity'   : severity,
            'rf'         : prediction['random_forest']['prediction'],
            'xgb'        : prediction['xgboost']['prediction'],
            'lstm'        : prediction['lstm']['prediction'],
            'correct'    : actual == ensemble['prediction'],
        })

        progress.progress((i + 1) / n)
        status.text(f"Analyzing sample {i+1}/{n}...")

    progress.empty()
    status.empty()
    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════════════════
# MODE 1: Live Detection
# ══════════════════════════════════════════════════════════════════════════════
if mode == "Live Detection":
    st.subheader("Live Traffic Analysis")

    if st.button("Start Detection", type="primary"):
        X_test, y_test = get_test_data()
        df = run_detection(X_test, y_test, n_samples)

        # ── KPI Metrics ───────────────────────────────────────────────────────
        total    = len(df)
        attacks  = len(df[df['predicted'] == 'ATTACK'])
        normal   = len(df[df['predicted'] == 'NORMAL'])
        accuracy = df['correct'].mean() * 100
        critical = len(df[df['severity'] == 'CRITICAL'])
        high     = len(df[df['severity'] == 'HIGH'])

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Samples",  total)
        col2.metric("Attacks Found",  attacks, delta=f"{attacks/total*100:.1f}%")
        col3.metric("Normal Traffic", normal)
        col4.metric("Accuracy",       f"{accuracy:.1f}%")
        col5.metric("Critical Alerts",critical, delta_color="inverse")

        st.markdown("---")

        # ── Charts row ────────────────────────────────────────────────────────
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            fig = px.pie(
                values=[normal, attacks],
                names=['Normal', 'Attack'],
                title='Traffic Distribution',
                color_discrete_sequence=['#00cc66', '#ff4b4b']
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            sev_counts = df['severity'].value_counts().reset_index()
            sev_counts.columns = ['Severity', 'Count']
            color_map = {
                'CRITICAL':'#ff4b4b','HIGH':'#ffa500',
                'MEDIUM':'#4b9dff','LOW':'#00cc66','NORMAL':'#00cc66'
            }
            fig2 = px.bar(
                sev_counts, x='Severity', y='Count',
                title='Severity Breakdown',
                color='Severity',
                color_discrete_map=color_map
            )
            fig2.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        with col_c:
            fig3 = px.histogram(
                df, x='confidence', color='predicted',
                title='Confidence Distribution',
                color_discrete_map={
                    'NORMAL':'#00cc66','ATTACK':'#ff4b4b'
                },
                nbins=20
            )
            fig3.update_layout(height=300)
            st.plotly_chart(fig3, use_container_width=True)

        st.markdown("---")

        # ── Confidence timeline ───────────────────────────────────────────────
        fig4 = px.line(
            df, x='sample', y='confidence',
            color='predicted',
            title='Confidence Score per Sample',
            color_discrete_map={
                'NORMAL':'#00cc66', 'ATTACK':'#ff4b4b'
            }
        )
        fig4.add_hline(y=90, line_dash='dash',
                       line_color='red',   annotation_text='CRITICAL')
        fig4.add_hline(y=70, line_dash='dash',
                       line_color='orange',annotation_text='HIGH')
        fig4.add_hline(y=50, line_dash='dash',
                       line_color='blue',  annotation_text='MEDIUM')
        st.plotly_chart(fig4, use_container_width=True)

        st.markdown("---")

        # ── Results table ─────────────────────────────────────────────────────
        st.subheader("Detection Results")

        def highlight_row(row):
            if row['severity'] == 'CRITICAL':
                return ['background-color: #3d0000'] * len(row)
            elif row['severity'] == 'HIGH':
                return ['background-color: #3d2000'] * len(row)
            elif row['severity'] == 'MEDIUM':
                return ['background-color: #00003d'] * len(row)
            elif row['predicted'] == 'NORMAL':
                return ['background-color: #003d00'] * len(row)
            return [''] * len(row)

        st.dataframe(
            df.style.apply(highlight_row, axis=1),
            use_container_width=True,
            height=400
        )

        # ── Download button ───────────────────────────────────────────────────
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name=f"ids_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime='text/csv'
        )


# ══════════════════════════════════════════════════════════════════════════════
# MODE 2: Batch Analysis
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "Batch Analysis":
    st.subheader("Batch Analysis — Model Comparison")
    st.info("Compare all 3 models side by side on the test dataset.")

    if st.button("Run Batch Analysis", type="primary"):
        X_test, y_test = get_test_data()
        df = run_detection(X_test, y_test, n_samples)

        # Model agreement chart
        model_results = pd.DataFrame({
            'Model'    : ['Random Forest', 'XGBoost', 'LSTM'],
            'Attacks'  : [
                len(df[df['rf']   == 'ATTACK']),
                len(df[df['xgb']  == 'ATTACK']),
                len(df[df['lstm'] == 'ATTACK']),
            ],
            'Normal'   : [
                len(df[df['rf']   == 'NORMAL']),
                len(df[df['xgb']  == 'NORMAL']),
                len(df[df['lstm'] == 'NORMAL']),
            ],
        })

        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(
                model_results,
                x='Model', y=['Attacks', 'Normal'],
                title='Model-wise Detection Count',
                barmode='group',
                color_discrete_map={
                    'Attacks':'#ff4b4b', 'Normal':'#00cc66'
                }
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Agreement matrix
            df['all_agree'] = (
                (df['rf'] == df['xgb']) &
                (df['xgb'] == df['lstm'])
            )
            agree_count    = df['all_agree'].sum()
            disagree_count = len(df) - agree_count

            fig2 = px.pie(
                values=[agree_count, disagree_count],
                names=['All Models Agree', 'Models Disagree'],
                title='Model Agreement Rate',
                color_discrete_sequence=['#4b9dff', '#ffa500']
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Accuracy per model
        st.subheader("Per-Model Accuracy")
        col3, col4, col5 = st.columns(3)

        rf_acc  = (df['rf']  == df['actual']).mean() * 100
        xgb_acc = (df['xgb'] == df['actual']).mean() * 100
        lstm_acc= (df['lstm']== df['actual']).mean() * 100

        col3.metric("Random Forest", f"{rf_acc:.1f}%")
        col4.metric("XGBoost",       f"{xgb_acc:.1f}%")
        col5.metric("LSTM",          f"{lstm_acc:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# MODE 3: Log Viewer
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "Log Viewer":
    st.subheader("Alert Log Viewer")

    log_path = os.path.join('logs', 'ids_alerts.log')

    if not os.path.exists(log_path):
        st.warning("No log file found. Run Live Detection first.")
    else:
        with open(log_path, 'r') as f:
            lines = f.readlines()

        st.info(f"Total log entries: {len(lines)}")

        # Parse JSON alerts from log
        alerts = []
        for line in lines:
            try:
                parts = line.split(' | ', 2)
                if len(parts) == 3:
                    data = json.loads(parts[2].strip())
                    if isinstance(data, dict) and 'severity' in data:
                        alerts.append(data)
            except Exception:
                continue

        if alerts:
            df_log = pd.DataFrame([{
                'timestamp'  : a.get('timestamp', ''),
                'sample_id'  : a.get('sample_id', ''),
                'severity'   : a.get('severity', ''),
                'confidence' : a.get('confidence', 0),
            } for a in alerts])

            col1, col2 = st.columns(2)

            with col1:
                sev_counts = df_log['severity'].value_counts()
                fig = px.bar(
                    x=sev_counts.index,
                    y=sev_counts.values,
                    title='Logged Alerts by Severity',
                    color=sev_counts.index,
                    color_discrete_map={
                        'CRITICAL':'#ff4b4b','HIGH':'#ffa500',
                        'MEDIUM':'#4b9dff','LOW':'#00cc66'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig2 = px.histogram(
                    df_log, x='confidence',
                    title='Confidence Distribution of Logged Alerts',
                    color_discrete_sequence=['#ff4b4b']
                )
                st.plotly_chart(fig2, use_container_width=True)

            st.subheader("Raw Alert Log")
            st.dataframe(df_log, use_container_width=True)

        # Show raw log
        with st.expander("View Raw Log File"):
            st.text(''.join(lines[-50:]))