import os
import sys
import time
import numpy as np
import psutil
from datetime import datetime
from collections import defaultdict

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scapy.all       import sniff, IP, TCP, UDP, ICMP
from src.predict      import load_all_models, predict_single
from src.alert_engine import trigger_alert, print_summary

# ── Globals ───────────────────────────────────────────────────────────────────
scaler, rf_model, xgb_model, lstm_model = None, None, None, None
alerts    = []
pkt_count = 0

# Connection flow table — key = (src_ip, dst_ip, src_port, dst_port, proto)
flow_table = defaultdict(lambda: {
    'start_time'        : None,
    'end_time'          : None,
    'src_bytes'         : 0,
    'dst_bytes'         : 0,
    'src_pkts'          : 0,
    'dst_pkts'          : 0,
    'flags'             : [],
    'wrong_fragment'    : 0,
    'urgent'            : 0,
    'land'              : 0,
    'protocol'          : 0,
    'service'           : 0,
    'state'             : 'OTH',
    'fin_seen'          : False,
    'syn_seen'          : False,
    'rst_seen'          : False,
})

# Track same-service / same-host connections (last 2 seconds window)
recent_connections = []
WINDOW_SECONDS     = 2


# ── Mappings ──────────────────────────────────────────────────────────────────
PROTOCOL_MAP = {'tcp': 0, 'udp': 1, 'icmp': 2}

SERVICE_MAP = {
    80: 'http', 443: 'https', 21: 'ftp', 22: 'ssh',
    23: 'telnet', 25: 'smtp', 53: 'domain', 110: 'pop3',
    143: 'imap', 3306: 'sql_net', 8080: 'http_8001',
    0: 'other'
}
SERVICE_ENCODE = {
    'http': 0, 'https': 1, 'ftp': 2, 'ssh': 3, 'telnet': 4,
    'smtp': 5, 'domain': 6, 'pop3': 7, 'imap': 8,
    'sql_net': 9, 'http_8001': 10, 'other': 11
}
FLAG_ENCODE = {
    'SF': 0, 'S0': 1, 'S1': 2, 'S2': 3, 'S3': 4,
    'SF': 5, 'REJ': 6, 'RSTO': 7, 'RSTOS0': 8,
    'RSTR': 9, 'SH': 10, 'OTH': 11
}


# ── Flow key builder ──────────────────────────────────────────────────────────
def get_flow_key(packet):
    if not packet.haslayer(IP):
        return None

    ip    = packet[IP]
    proto = 'other'
    sport = 0
    dport = 0

    if packet.haslayer(TCP):
        proto = 'tcp'
        sport = packet[TCP].sport
        dport = packet[TCP].dport
    elif packet.haslayer(UDP):
        proto = 'udp'
        sport = packet[UDP].sport
        dport = packet[UDP].dport
    elif packet.haslayer(ICMP):
        proto = 'icmp'

    # Normalize direction — always smaller IP first
    if ip.src < ip.dst:
        return (ip.src, ip.dst, sport, dport, proto)
    else:
        return (ip.dst, ip.src, dport, sport, proto)


# ── TCP flag decoder ──────────────────────────────────────────────────────────
def decode_tcp_state(flow):
    flags = flow['flags']
    if flow['fin_seen']:
        return 'SF'
    elif flow['rst_seen']:
        return 'RSTO' if flow['syn_seen'] else 'RSTOS0'
    elif flow['syn_seen'] and not flow['fin_seen']:
        return 'S0'
    elif len(flags) == 0:
        return 'OTH'
    return 'SF'


# ── Window-based rate features ────────────────────────────────────────────────
def compute_rate_features(flow_key, current_time):
    """
    Compute count, srv_count, and rate features
    using a 2-second sliding window of recent connections.
    """
    _, dst_ip, _, dport, proto = flow_key
    service = SERVICE_MAP.get(dport, 'other')

    # Filter connections within window
    window = [
        c for c in recent_connections
        if current_time - c['time'] <= WINDOW_SECONDS
    ]

    count         = len(window)
    srv_count     = sum(1 for c in window if c['service'] == service)
    same_srv      = srv_count / count if count > 0 else 0.0
    diff_srv      = 1.0 - same_srv
    serror_rate   = sum(1 for c in window if c['state'] in ['S0','S1','S2','S3']) / max(count,1)
    rerror_rate   = sum(1 for c in window if c['state'] in ['REJ']) / max(count,1)

    # dst_host window (last 100 connections to same host)
    host_window   = [c for c in recent_connections if c['dst_ip'] == dst_ip][-100:]
    dst_host_count= len(host_window)
    dst_srv_count = sum(1 for c in host_window if c['service'] == service)
    dst_same_srv  = dst_srv_count / max(dst_host_count, 1)
    dst_diff_srv  = 1.0 - dst_same_srv
    dst_serror    = sum(1 for c in host_window if c['state'] in ['S0','S1']) / max(dst_host_count,1)
    dst_rerror    = sum(1 for c in host_window if c['state'] == 'REJ') / max(dst_host_count,1)

    return {
        'count'                      : min(count, 511),
        'srv_count'                  : min(srv_count, 511),
        'serror_rate'                : serror_rate,
        'srv_serror_rate'            : serror_rate,
        'rerror_rate'                : rerror_rate,
        'srv_rerror_rate'            : rerror_rate,
        'same_srv_rate'              : same_srv,
        'diff_srv_rate'              : diff_srv,
        'srv_diff_host_rate'         : diff_srv,
        'dst_host_count'             : min(dst_host_count, 255),
        'dst_host_srv_count'         : min(dst_srv_count, 255),
        'dst_host_same_srv_rate'     : dst_same_srv,
        'dst_host_diff_srv_rate'     : dst_diff_srv,
        'dst_host_same_src_port_rate': 0.0,
        'dst_host_srv_diff_host_rate': 0.0,
        'dst_host_serror_rate'       : dst_serror,
        'dst_host_srv_serror_rate'   : dst_serror,
        'dst_host_rerror_rate'       : dst_rerror,
        'dst_host_srv_rerror_rate'   : dst_rerror,
    }


# ── Feature builder from completed flow ───────────────────────────────────────
def build_feature_vector(flow_key, flow):
    """
    Build complete 41-feature NSL-KDD vector
    from a completed connection flow.
    """
    _, dst_ip, _, dport, proto = flow_key

    duration    = (flow['end_time'] - flow['start_time']) if flow['end_time'] else 0
    state       = decode_tcp_state(flow) if proto == 'tcp' else 'SF'
    service_str = SERVICE_MAP.get(dport, 'other')
    rates       = compute_rate_features(flow_key, time.time())

    features = np.array([
        duration,                                   # 0  duration
        PROTOCOL_MAP.get(proto, 0),                 # 1  protocol_type
        SERVICE_ENCODE.get(service_str, 11),        # 2  service
        FLAG_ENCODE.get(state, 11),                 # 3  flag
        flow['src_bytes'],                          # 4  src_bytes
        flow['dst_bytes'],                          # 5  dst_bytes
        flow['land'],                               # 6  land
        flow['wrong_fragment'],                     # 7  wrong_fragment
        flow['urgent'],                             # 8  urgent
        0,                                          # 9  hot
        0,                                          # 10 num_failed_logins
        0,                                          # 11 logged_in
        0,                                          # 12 num_compromised
        0,                                          # 13 root_shell
        0,                                          # 14 su_attempted
        0,                                          # 15 num_root
        0,                                          # 16 num_file_creations
        0,                                          # 17 num_shells
        0,                                          # 18 num_access_files
        0,                                          # 19 num_outbound_cmds
        0,                                          # 20 is_host_login
        0,                                          # 21 is_guest_login
        rates['count'],                             # 22 count
        rates['srv_count'],                         # 23 srv_count
        rates['serror_rate'],                       # 24 serror_rate
        rates['srv_serror_rate'],                   # 25 srv_serror_rate
        rates['rerror_rate'],                       # 26 rerror_rate
        rates['srv_rerror_rate'],                   # 27 srv_rerror_rate
        rates['same_srv_rate'],                     # 28 same_srv_rate
        rates['diff_srv_rate'],                     # 29 diff_srv_rate
        rates['srv_diff_host_rate'],                # 30 srv_diff_host_rate
        rates['dst_host_count'],                    # 31 dst_host_count
        rates['dst_host_srv_count'],                # 32 dst_host_srv_count
        rates['dst_host_same_srv_rate'],            # 33 dst_host_same_srv_rate
        rates['dst_host_diff_srv_rate'],            # 34 dst_host_diff_srv_rate
        rates['dst_host_same_src_port_rate'],       # 35 dst_host_same_src_port_rate
        rates['dst_host_srv_diff_host_rate'],       # 36 dst_host_srv_diff_host_rate
        rates['dst_host_serror_rate'],              # 37 dst_host_serror_rate
        rates['dst_host_srv_serror_rate'],          # 38 dst_host_srv_serror_rate
        rates['dst_host_rerror_rate'],              # 39 dst_host_rerror_rate
        rates['dst_host_srv_rerror_rate'],          # 40 dst_host_srv_rerror_rate
    ], dtype=np.float32)

    return features


# ── Packet handler ────────────────────────────────────────────────────────────
def handle_packet(packet):
    global pkt_count, alerts

    flow_key = get_flow_key(packet)
    if flow_key is None:
        return

    _, dst_ip, _, dport, proto = flow_key
    flow = flow_table[flow_key]
    now  = time.time()

    # Initialize flow
    if flow['start_time'] is None:
        flow['start_time'] = now
        flow['protocol']   = PROTOCOL_MAP.get(proto, 0)
        flow['service']    = SERVICE_ENCODE.get(
                                SERVICE_MAP.get(dport, 'other'), 11)
        flow['land']       = 1 if flow_key[0] == flow_key[1] else 0

    flow['end_time'] = now

    # Accumulate bytes
    if packet.haslayer(IP):
        ip = packet[IP]
        if ip.src == flow_key[0]:
            flow['src_bytes'] += len(packet)
        else:
            flow['dst_bytes'] += len(packet)

        # Wrong fragment
        if ip.flags == 1 or ip.frag > 0:
            flow['wrong_fragment'] += 1

    # TCP flag tracking
    if packet.haslayer(TCP):
        tcp = packet[TCP]
        flags = str(tcp.flags)
        flow['flags'].append(flags)
        if 'F' in flags: flow['fin_seen'] = True
        if 'S' in flags: flow['syn_seen'] = True
        if 'R' in flags: flow['rst_seen'] = True
        if tcp.urgptr > 0: flow['urgent'] += 1

    pkt_count += 1

    # Analyze completed flows (FIN or RST seen, or UDP/ICMP)
    should_analyze = (
        flow['fin_seen'] or
        flow['rst_seen'] or
        proto in ['udp', 'icmp'] or
        (now - flow['start_time']) > 5  # timeout after 5 seconds
    )

    if should_analyze:
        features = build_feature_vector(flow_key, flow)

        try:
            prediction = predict_single(
                features, scaler, rf_model, xgb_model, lstm_model
            )
            alert = trigger_alert(prediction, sample_id=pkt_count)
            if alert:
                alerts.append(alert)

            # Add to recent connections window
            recent_connections.append({
                'time'    : now,
                'dst_ip'  : dst_ip,
                'service' : SERVICE_MAP.get(dport, 'other'),
                'state'   : decode_tcp_state(flow) if proto == 'tcp' else 'SF',
            })

        except Exception as e:
            print(f"  Prediction error: {e}")

        # Remove completed flow from table
        del flow_table[flow_key]


# ── Interface selector ────────────────────────────────────────────────────────
def list_interfaces():
    print("\nAvailable Network Interfaces:")
    ifaces = list(psutil.net_if_addrs().keys())
    for i, iface in enumerate(ifaces):
        print(f"  [{i}] {iface}")
    return ifaces


# ── Main ──────────────────────────────────────────────────────────────────────
import json
import threading

LIVE_FEED_PATH = 'logs/live_feed.json'

def write_live_feed(results):
    """Write current results to shared JSON for dashboard."""
    os.makedirs('logs', exist_ok=True)
    with open(LIVE_FEED_PATH, 'w') as f:
        json.dump(results, f)

# ── Live feed state ───────────────────────────────────────────────────────────
live_results = {
    'total'     : 0,
    'attacks'   : 0,
    'normal'    : 0,
    'detections': [],   # last 100 detections
    'severity'  : {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0},
    'running'   : True,
    'started_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
}

def handle_packet_live(packet):
    """Extended packet handler that writes to live feed."""
    global pkt_count, alerts

    flow_key = get_flow_key(packet)
    if flow_key is None:
        return

    _, dst_ip, _, dport, proto = flow_key
    flow = flow_table[flow_key]
    now  = time.time()

    if flow['start_time'] is None:
        flow['start_time'] = now
        flow['protocol']   = PROTOCOL_MAP.get(proto, 0)
        flow['service']    = SERVICE_ENCODE.get(
                                SERVICE_MAP.get(dport, 'other'), 11)
        flow['land']       = 1 if flow_key[0] == flow_key[1] else 0

    flow['end_time'] = now

    if packet.haslayer(IP):
        ip = packet[IP]
        if ip.src == flow_key[0]:
            flow['src_bytes'] += len(packet)
        else:
            flow['dst_bytes'] += len(packet)
        if ip.flags == 1 or ip.frag > 0:
            flow['wrong_fragment'] += 1

    if packet.haslayer(TCP):
        tcp = packet[TCP]
        flags = str(tcp.flags)
        flow['flags'].append(flags)
        if 'F' in flags: flow['fin_seen'] = True
        if 'S' in flags: flow['syn_seen'] = True
        if 'R' in flags: flow['rst_seen'] = True
        if tcp.urgptr > 0: flow['urgent'] += 1

    pkt_count += 1

    should_analyze = (
        flow['fin_seen'] or flow['rst_seen'] or
        proto in ['udp', 'icmp'] or
        (now - flow['start_time']) > 5
    )

    if should_analyze:
        features = build_feature_vector(flow_key, flow)

        try:
            prediction  = predict_single(
                features, scaler, rf_model, xgb_model, lstm_model
            )
            ensemble    = prediction['ensemble']
            result      = ensemble['prediction']
            confidence  = float(ensemble['confidence'].replace('%', ''))
            timestamp   = datetime.now().strftime('%H:%M:%S')

            # Severity
            severity = 'NORMAL'
            if result == 'ATTACK':
                if confidence >= 90:   severity = 'CRITICAL'
                elif confidence >= 70: severity = 'HIGH'
                elif confidence >= 50: severity = 'MEDIUM'
                else:                  severity = 'LOW'

            # Update live state
            live_results['total'] += 1
            if result == 'ATTACK':
                live_results['attacks'] += 1
                live_results['severity'][severity] += 1
                alert = trigger_alert(prediction, sample_id=pkt_count)
                if alert:
                    alerts.append(alert)
            else:
                live_results['normal'] += 1

            # Keep last 100 detections
            live_results['detections'].append({
                'time'      : timestamp,
                'sample'    : pkt_count,
                'result'    : result,
                'confidence': round(confidence, 2),
                'severity'  : severity,
                'rf'        : prediction['random_forest']['prediction'],
                'xgb'       : prediction['xgboost']['prediction'],
                'lstm'      : prediction['lstm']['prediction'],
            })
            live_results['detections'] = live_results['detections'][-100:]

            # Write to shared feed
            write_live_feed(live_results)

            recent_connections.append({
                'time'    : now,
                'dst_ip'  : dst_ip,
                'service' : SERVICE_MAP.get(dport, 'other'),
                'state'   : decode_tcp_state(flow) if proto == 'tcp' else 'SF',
            })

        except Exception as e:
            print(f"  Error: {e}")

        del flow_table[flow_key]


if __name__ == '__main__':
    print("=" * 55)
    print("  NETWORK IDS — LIVE CAPTURE + DASHBOARD MODE")
    print("=" * 55)

    print("\nLoading models...")
    scaler, rf_model, xgb_model, lstm_model = load_all_models()

    ifaces = list_interfaces()
    print()
    idx   = int(input("Select interface number: "))
    iface = ifaces[idx]

    print(f"\nStarting capture on [{iface}]...")
    print("Open a second terminal and run:")
    print("  streamlit run src/live_dashboard.py")
    print("\nThen browse the web to generate traffic!")
    print("-" * 55)

    try:
        sniff(
            iface = iface,
            prn   = handle_packet_live,
            store = False       # capture indefinitely
        )
    except KeyboardInterrupt:
        live_results['running'] = False
        write_live_feed(live_results)
        print("\nCapture stopped.")
        print_summary(alerts, pkt_count)
    except PermissionError:
        print("\nRun PowerShell as Administrator!")