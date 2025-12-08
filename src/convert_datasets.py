import pandas as pd
import numpy as np
import os


TARGET_COLUMNS = ['duration', 'protocol', 'src_bytes', 'dst_bytes', 'label']

def clean_protocol(val):
    try:
        s = str(val).lower().strip()
        if s.endswith('.0'): s = s[:-2]
        if s in ['tcp', '6']: return 6
        if s in ['udp', '17']: return 17
        if s in ['icmp', '1']: return 1
        return 0
    except:
        return 0

def clean_label(val):
    try:
        s = str(val).lower().strip()
        if s in ['0', '0.0', 'normal', 'benign', 'no']: return 0
        return 1
    except:
        return 1

def safe_log_transform(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            df[col] = np.log1p(df[col])
    return df


def process_kdd(file_path):
    print(f"Processing KDD Cup 99 from {file_path}...")
    try:
        df = pd.read_csv(file_path, header=None)
    except Exception as e:
        print(f"Error reading KDD: {e}")
        return None
    out = pd.DataFrame()
    out['duration'] = df[0]
    out['protocol'] = df[1].apply(clean_protocol)
    out['src_bytes'] = df[4]
    out['dst_bytes'] = df[5]
    out['label'] = df.iloc[:, -1].apply(lambda x: 0 if str(x).startswith('normal') else 1)
    out = safe_log_transform(out, ['duration', 'src_bytes', 'dst_bytes'])
    return out


def process_cores(file_path):
    print(f"Processing CORES-IoT from {file_path}...")
    try:
        df = pd.read_csv(file_path, header=None)
    except Exception as e:
        print(f"Error reading CORES: {e}")
        return None
    out = pd.DataFrame()
    out['duration']  = pd.to_numeric(df.iloc[:, 0], errors='coerce').fillna(0) / 1_000_000.0
    out['protocol']  = df.iloc[:, 10].apply(clean_protocol)
    out['src_bytes'] = df.iloc[:, 7]
    out['dst_bytes'] = df.iloc[:, 14]
    out['label'] = df.iloc[:, -1].apply(clean_label)
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
    if 'FLOW_DURATION_MILLISECONDS' in df.columns:
        out['duration'] = pd.to_numeric(df['FLOW_DURATION_MILLISECONDS'], errors='coerce').fillna(0) / 1000.0
    else:
        out['duration'] = 0
    if 'PROTOCOL' in df.columns:
        out['protocol'] = df['PROTOCOL'].apply(clean_protocol)
    elif 'PROTOCOL_MAP' in df.columns:
        out['protocol'] = df['PROTOCOL_MAP'].apply(clean_protocol)
    else:
        out['protocol'] = 0
    if 'IN_BYTES' in df.columns:
        out['src_bytes'] = df['IN_BYTES']
    else:
        out['src_bytes'] = df.get('TOTAL_BYTES_EXP', 0)
    if 'OUT_BYTES' in df.columns:
        out['dst_bytes'] = df['OUT_BYTES']
    else:
        out['dst_bytes'] = 0
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