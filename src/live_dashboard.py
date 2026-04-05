import os
import json
import time
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

LIVE_FEED_PATH = 'logs/live_feed.json'

st.set_page_config(
    page_title = "Live IDS Monitor",
    page_icon  = "shield",
    layout     = "wide"
)

st.markdown("""
<style>
.critical { color: #ff4b4b; font-size: 20px; font-weight: bold; }
.high     { color: #ffa500; font-size: 20px; font-weight: bold; }
.medium   { color: #4b9dff; font-size: 20px; font-weight: bold; }
.normal   { color: #00cc66; font-size: 20px; font-weight: bold; }
.blink    { animation: blink 1s step-start infinite; }
@keyframes blink { 50% { opacity: 0; } }
</style>
""", unsafe_allow_html=True)


def load_feed():
    if not os.path.exists(LIVE_FEED_PATH):
        return None
    try:
        with open(LIVE_FEED_PATH, 'r') as f:
            return json.load(f)
    except Exception:
        return None


# ── Header ────────────────────────────────────────────────────────────────────
st.title("Live Network Intrusion Detection")
st.markdown("Real-time packet analysis — auto-refreshes every 2 seconds")

status_box = st.empty()
st.markdown("---")

# ── KPI Row ───────────────────────────────────────────────────────────────────
kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)

# ── Charts Row ────────────────────────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Traffic Distribution")
    pie_chart = st.empty()

with col_right:
    st.subheader("Severity Breakdown")
    bar_chart = st.empty()

st.markdown("---")

# ── Timeline ──────────────────────────────────────────────────────────────────
st.subheader("Confidence Timeline")
timeline_chart = st.empty()

st.markdown("---")

# ── Live Alert Feed ───────────────────────────────────────────────────────────
st.subheader("Live Alert Feed")
alert_table = st.empty()

# ── Auto refresh loop ─────────────────────────────────────────────────────────
refresh_rate = st.sidebar.slider("Refresh rate (seconds)", 1, 10, 2)
st.sidebar.markdown("---")
st.sidebar.markdown("**How to use:**")
st.sidebar.info(
    "1. Run live_capture.py as Admin\n"
    "2. Select your Wi-Fi interface\n"
    "3. Browse the web\n"
    "4. Watch detections appear here!"
)

while True:
    data = load_feed()

    if data is None:
        status_box.warning(
            "Waiting for live capture... "
            "Run: python src/live_capture.py"
        )
        time.sleep(refresh_rate)
        st.rerun()
        continue

    # Status
    running = data.get('running', True)
    if running:
        status_box.success(
            f"LIVE — Capturing since {data.get('started_at', '')}  "
            f"| Last update: {datetime.now().strftime('%H:%M:%S')}"
        )
    else:
        status_box.error("Capture stopped.")

    total   = data.get('total', 0)
    attacks = data.get('attacks', 0)
    normal  = data.get('normal', 0)
    sev     = data.get('severity', {})
    rate    = (attacks / total * 100) if total > 0 else 0

    # KPIs
    kpi1.metric("Total Flows",    total)
    kpi2.metric("Attacks",        attacks, delta=f"{rate:.1f}%",
                delta_color="inverse")
    kpi3.metric("Normal",         normal)
    kpi4.metric("Critical",       sev.get('CRITICAL', 0),
                delta_color="inverse")
    kpi5.metric("High",           sev.get('HIGH', 0),
                delta_color="inverse")
    kpi6.metric("Medium",         sev.get('MEDIUM', 0),
                delta_color="inverse")

    # Pie chart
    if total > 0:
        fig_pie = px.pie(
            values=[normal, attacks],
            names=['Normal', 'Attack'],
            color_discrete_sequence=['#00cc66', '#ff4b4b'],
            hole=0.4
        )
        fig_pie.update_layout(height=300, margin=dict(t=0,b=0))
        pie_chart.plotly_chart(fig_pie, use_container_width=True)

    # Severity bar
    fig_bar = px.bar(
        x=['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'],
        y=[sev.get('CRITICAL',0), sev.get('HIGH',0),
           sev.get('MEDIUM',0),   sev.get('LOW',0)],
        color=['CRITICAL','HIGH','MEDIUM','LOW'],
        color_discrete_map={
            'CRITICAL':'#ff4b4b','HIGH':'#ffa500',
            'MEDIUM':'#4b9dff','LOW':'#00cc66'
        }
    )
    fig_bar.update_layout(height=300, showlegend=False,
                          margin=dict(t=0,b=0))
    bar_chart.plotly_chart(fig_bar, use_container_width=True)

    # Timeline
    detections = data.get('detections', [])
    if detections:
        df = pd.DataFrame(detections)
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=df['time'], y=df['confidence'],
            mode='lines+markers',
            marker=dict(
                color=['#ff4b4b' if r == 'ATTACK'
                       else '#00cc66' for r in df['result']],
                size=8
            ),
            line=dict(color='#888', width=1)
        ))
        fig_line.add_hline(y=90, line_dash='dash',
                           line_color='red',
                           annotation_text='CRITICAL')
        fig_line.add_hline(y=70, line_dash='dash',
                           line_color='orange',
                           annotation_text='HIGH')
        fig_line.add_hline(y=50, line_dash='dash',
                           line_color='blue',
                           annotation_text='MEDIUM')
        fig_line.update_layout(
            height=250,
            yaxis_title='Confidence %',
            xaxis_title='Time',
            margin=dict(t=10, b=10)
        )
        timeline_chart.plotly_chart(fig_line, use_container_width=True)

        # Alert table — attacks only, most recent first
        attack_df = df[df['result'] == 'ATTACK'].iloc[::-1]
        if not attack_df.empty:
            alert_table.dataframe(
                attack_df[['time','sample','severity',
                           'confidence','rf','xgb','lstm']],
                use_container_width=True,
                height=300
            )
        else:
            alert_table.info("No attacks detected yet.")

    time.sleep(refresh_rate)
    st.rerun()