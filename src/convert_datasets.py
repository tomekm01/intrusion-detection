import pandas as pd
import numpy as np
import os

# --- CONFIGURATION ---
TARGET_COLUMNS = ['duration', 'protocol', 'src_bytes', 'dst_bytes', 'label']

# --- HELPER FUNCTIONS ---

def clean_protocol(val):
    """Maps varied protocol representations to standard integers (6=TCP, 17=UDP, 1=ICMP)."""
    try:
        s = str(val).lower().strip()
        if s.endswith('.0'): s = s[:-2] # Handle '6.0' -> '6'
        
        if s in ['tcp', '6']: return 6
        if s in ['udp', '17']: return 17
        if s in ['icmp', '1']: return 1
        return 0 # Unknown
    except:
        return 0

def clean_label(val):
    """Binarizes the label: 0 for normal, 1 for attack."""
    try:
        s = str(val).lower().strip()
        # Normal patterns
        if s in ['0', '0.0', 'normal', 'benign', 'no']: return 0
        return 1
    except:
        return 1

def safe_log_transform(df, cols):
    """Applies log1p to numerical columns to handle massive skew."""
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            df[col] = np.log1p(df[col])
    return df

# --- DATASET PROCESSING ---

def process_kdd(file_path):
    print(f"Processing KDD Cup 99 from {file_path}...")
    kdd_cols = [
        "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", 
        "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", 
        "logged_in", "num_compromised", "root_shell", "su_attempted", 
        "num_root", "num_file_creations", "num_shells", "num_access_files", 
        "num_outbound_cmds", "is_host_login", "is_guest_login", "count", 
        "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", 
        "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", 
        "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", 
        "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", 
        "dst_host_srv_diff_host_rate", "dst_host_serror_rate", 
        "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", 
        "label"
    ]
    
    try:
        df = pd.read_csv(file_path, names=kdd_cols, header=None)
    except Exception as e:
        print(f"Error reading KDD: {e}")
        return None

    out = pd.DataFrame()
    out['duration'] = df['duration']
    out['protocol'] = df['protocol_type'].apply(clean_protocol)
    out['src_bytes'] = df['src_bytes']
    out['dst_bytes'] = df['dst_bytes']
    out['label'] = df['label'].apply(lambda x: 0 if str(x).startswith('normal') else 1)
    
    out = safe_log_transform(out, ['duration', 'src_bytes', 'dst_bytes'])
    return out

def process_cores(file_path):
    print(f"Processing CORES-IoT from {file_path}...")
    
    # Col 0:  Duration (Large float)
    # Col 7:  Src Bytes / Volume 1 (e.g., 2599.0)
    # Col 10: PROTOCOL (1.0=ICMP, 6.0=TCP)
    # Col 14: Dst Bytes / Volume 2 (e.g., 118.0)
    # Col 19: Label (Last column)
    
    try:
        df = pd.read_csv(file_path, header=None)
    except Exception as e:
        print(f"Error reading CORES: {e}")
        return None
        
    out = pd.DataFrame()
    
    # 1. Map Features using CORRECTED Indices
    # Convert duration (microseconds) to seconds
    out['duration']  = pd.to_numeric(df.iloc[:, 0], errors='coerce').fillna(0) / 1_000_000.0
    
    # Protocol is definitely Column 10
    out['protocol']  = df.iloc[:, 10].apply(clean_protocol)
    
    # Bytes
    out['src_bytes'] = df.iloc[:, 7]
    out['dst_bytes'] = df.iloc[:, 14]
    
    # Label
    out['label'] = df.iloc[:, -1].apply(clean_label)
    
    # 2. Normalize
    out = safe_log_transform(out, ['duration', 'src_bytes', 'dst_bytes'])
    
    return out

def process_netflow(file_path):
    print(f"Processing NetFlow v9 from {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading NetFlow: {e}")
        return None

    df.columns = [c.strip().upper() for c in df.columns]
    out = pd.DataFrame()

    # Duration: FLOW_DURATION_MILLISECONDS -> Seconds
    if 'FLOW_DURATION_MILLISECONDS' in df.columns:
        out['duration'] = pd.to_numeric(df['FLOW_DURATION_MILLISECONDS'], errors='coerce').fillna(0) / 1000.0
    else:
        out['duration'] = 0

    # Protocol
    if 'PROTOCOL' in df.columns:
        out['protocol'] = df['PROTOCOL'].apply(clean_protocol)
    elif 'PROTOCOL_MAP' in df.columns:
        out['protocol'] = df['PROTOCOL_MAP'].apply(clean_protocol)
    else:
        out['protocol'] = 0

    # Bytes
    if 'IN_BYTES' in df.columns:
        out['src_bytes'] = df['IN_BYTES']
    else:
        out['src_bytes'] = df.get('TOTAL_BYTES_EXP', 0)

    if 'OUT_BYTES' in df.columns:
        out['dst_bytes'] = df['OUT_BYTES']
    else:
        out['dst_bytes'] = 0

    # Label
    if 'ANOMALY' in df.columns:
        out['label'] = df['ANOMALY'].apply(clean_label)
    else:
        out['label'] = df.get('LABEL', 0).apply(clean_label)

    out = safe_log_transform(out, ['duration', 'src_bytes', 'dst_bytes'])
    return out

def main():
    files = {
        'kdd': '../data/raw/kddcup.csv',
        'cores': '../data/raw/cores_iot.csv',
        'netflow': '../data/raw/netflow.csv'
    }
    
    # Process all
    for name, path in files.items():
        if os.path.exists(path):
            if name == 'kdd': df = process_kdd(path)
            elif name == 'cores': df = process_cores(path)
            elif name == 'netflow': df = process_netflow(path)
            
            if df is not None:
                output_name = f'processed_{name}.csv'
                df.to_csv(output_name, index=False)
                print(f"Success: Saved '{output_name}' ({len(df)} rows)")

if __name__ == "__main__":
    main()