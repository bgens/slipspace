import os
import re
import argparse
import pickle
import threading
from pathlib import Path
from datetime import datetime
from impacket.smbconnection import SMBConnection
from concurrent.futures import ThreadPoolExecutor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN

# Global locks for thread safety
lock = threading.Lock()
print_lock = threading.Lock()

# Constants for saved states
DISCOVERY_STATE_FILE = 'discovery_state.pkl'
RESULT_STATE_FILE = 'result_state.pkl'
DYNAMIC_IGNORE_LIST = 'dynamic_ignore_list.pkl'

# Verbose mode flag
VERBOSE = False

def log(message):
    with print_lock:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

def verbose_log(message):
    if VERBOSE:
        log(message)

def save_state(filename, data):
    with lock:
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

def load_state(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return None

def discover_files(smb_conn, share_name, base_path, discovered, ignore_list):
    try:
        for root, dirs, files in smb_conn.walk(share_name, base_path):
            for file in files:
                file_path = os.path.join(root, file)
                if ignore_list and any(file.endswith(ign) for ign in ignore_list):
                    verbose_log(f"Ignoring file (ignore list match): {file_path}")
                    continue
                try:
                    size = smb_conn.getAttributes(share_name, file_path).getSize()
                    discovered[file_path] = size
                    verbose_log(f"Discovered file: {file_path} (Size: {size} bytes)")
                except Exception as e:
                    verbose_log(f"Failed to access file: {file_path} - {e}")
            save_state(DISCOVERY_STATE_FILE, discovered)
    except Exception as e:
        log(f"Error during discovery: {e}")

def scan_file(smb_conn, share_name, file_path, patterns, max_file_size_bytes):
    try:
        size = smb_conn.getAttributes(share_name, file_path).getSize()
        if size > max_file_size_bytes:
            verbose_log(f"Skipping large file: {file_path}")
            return None

        with smb_conn.openFile(share_name, file_path, mode='r') as f:
            content = f.read().decode(errors='ignore')
            matches = [p for p in patterns if re.search(p, content)]
            return matches if matches else None
    except Exception as e:
        log(f"Error reading file {file_path}: {e}")
        return None

def perform_clustering(file_paths):
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
    file_names = [os.path.basename(path) for path in file_paths]
    vectors = vectorizer.fit_transform(file_names)

    clustering = DBSCAN(eps=0.5, min_samples=2, metric='cosine').fit(vectors)
    clusters = {}
    for idx, label in enumerate(clustering.labels_):
        clusters.setdefault(label, []).append(file_paths[idx])
    return clusters

def scan_cluster(smb_conn, share_name, cluster_files, patterns, max_file_size_bytes, results, dynamic_ignore_list):
    sample_files = cluster_files[:15]
    any_matches = False

    for file_path in sample_files:
        matches = scan_file(smb_conn, share_name, file_path, patterns, max_file_size_bytes)
        if matches:
            any_matches = True
            break

    if not any_matches:
        representative_pattern = os.path.basename(cluster_files[0]).split('.')[0]
        dynamic_ignore_list.append(representative_pattern)
        verbose_log(f"Skipping cluster with pattern: {representative_pattern}")
    else:
        for file_path in cluster_files:
            matches = scan_file(smb_conn, share_name, file_path, patterns, max_file_size_bytes)
            if matches:
                with lock:
                    results[file_path] = {'matches': matches, 'size': smb_conn.getAttributes(share_name, file_path).getSize()}
                    save_state(RESULT_STATE_FILE, results)

def main():
    global VERBOSE

    parser = argparse.ArgumentParser(description="SMB Share File Scanner with Impacket Authentication")
    parser.add_argument("target_ip", help="Target SMB server IP")
    parser.add_argument("username", help="Username for SMB authentication")
    parser.add_argument("password", help="Password for SMB authentication")
    parser.add_argument("domain", help="Domain for SMB authentication")
    parser.add_argument("share", help="SMB Share name")
    parser.add_argument("--patterns", required=True, help="File containing regex patterns")
    parser.add_argument("--ignore-list", help="File containing patterns to ignore")
    parser.add_argument("--max-size", type=int, default=1, help="Max file size to scan (MB)")
    parser.add_argument("--threads", type=int, default=5, help="Number of threads to use")
    parser.add_argument("--verbose", action='store_true', help="Enable verbose output")
    parser.add_argument("--enable-clustering", action='store_true', help="Enable clustering optimization for scanning")

    args = parser.parse_args()
    VERBOSE = args.verbose
    max_file_size_bytes = args.max_size * 1024 * 1024

    with open(args.patterns, 'r') as pf:
        patterns = [line.strip() for line in pf if line.strip()]

    ignore_list = []
    if args.ignore_list:
        with open(args.ignore_list, 'r') as igf:
            ignore_list = [line.strip() for line in igf if line.strip()]

    dynamic_ignore_list = load_state(DYNAMIC_IGNORE_LIST) or []
    discovered = load_state(DISCOVERY_STATE_FILE) or {}
    results = load_state(RESULT_STATE_FILE) or {}

    smb_conn = SMBConnection(args.username, args.password, '', args.target_ip, domain=args.domain)
    smb_conn.connect(args.target_ip, 445)

    if not discovered:
        log("Starting discovery phase...")
        discover_files(smb_conn, args.share, '/', discovered, ignore_list + dynamic_ignore_list)
    else:
        log("Resuming from saved discovery state.")

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        if args.enable_clustering:
            clusters = perform_clustering(list(discovered.keys()))
            for cluster_files in clusters.values():
                executor.submit(scan_cluster, smb_conn, args.share, cluster_files, patterns, max_file_size_bytes, results, dynamic_ignore_list)
        else:
            for file_path in discovered.keys():
                executor.submit(scan_file, smb_conn, args.share, file_path, patterns, max_file_size_bytes)

    save_state(DYNAMIC_IGNORE_LIST, dynamic_ignore_list)
    smb_conn.close()

    log("Scan completed. Results saved.")

if __name__ == "__main__":
    main()
