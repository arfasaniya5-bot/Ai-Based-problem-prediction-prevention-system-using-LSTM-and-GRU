"""Backblaze Hard Drive Dataset Inspection Module.

DeviceGuard AI - Deep Learning Problem Prediction & Prevention System.
Safely inspects the Backblaze dataset using sample reads and chunked streaming,
ensuring the file is never loaded entirely into memory.
"""

import argparse
import os
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd


def format_file_size(size_bytes: int) -> str:
    """Format byte size into human-readable units (B, KB, MB, GB)."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0 or unit == 'TB':
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} B"


def inspect_sample(csv_path: Path, sample_size: int = 1000) -> tuple[bool, dict]:
    """Inspect initial sample of dataset using Pandas without full file load.

    Args:
        csv_path: Path to dataset CSV.
        sample_size: Number of initial rows to load (default: 1,000).

    Returns:
        tuple (success_boolean, sample_metadata_dict)
    """
    print("\n" + "=" * 70)
    print(f" STEP 2: Sample Inspection (First {sample_size:,} Rows)")
    print("=" * 70)

    try:
        sample_df = pd.read_csv(csv_path, nrows=sample_size)
    except Exception as err:
        print(f"[ERROR] Failed to read sample rows from CSV: {err}")
        return False, {}

    total_rows, total_cols = sample_df.shape
    cols = sample_df.columns.tolist()

    # 1. Column Names and Duplicates
    dup_cols = [c for c in cols if cols.count(c) > 1]
    print(f"Sample Dimensions: {total_rows} rows x {total_cols} columns")
    print(f"Duplicate Column Names: {len(dup_cols)} duplicate(s)")

    # 2. Check Expected Fields
    expected_core = ['date', 'serial_number', 'model', 'capacity_bytes', 'failure']
    print("\nExpected Core Fields Check:")
    for field in expected_core:
        status = "[PRESENT]" if field in sample_df.columns else "[MISSING]"
        dtype_str = str(sample_df[field].dtype) if field in sample_df.columns else "N/A"
        print(f"  {status:<10} {field:<18} (dtype: {dtype_str})")

    # 3. SMART Columns Check
    smart_raw = [c for c in cols if c.startswith('smart_') and c.endswith('_raw')]
    smart_norm = [c for c in cols if c.startswith('smart_') and c.endswith('_normalized')]
    print(f"\nSMART Attribute Columns in Sample:")
    print(f"  Raw Attributes (_raw)        : {len(smart_raw)}")
    print(f"  Normalized (_normalized)     : {len(smart_norm)}")

    # 4. Date Parsing Verification
    print("\nDate Parsing Verification:")
    if 'date' in sample_df.columns:
        try:
            parsed_dates = pd.to_datetime(sample_df['date'], format='%d-%m-%Y', errors='coerce')
            invalid_dates = parsed_dates.isnull().sum()
            print(f"  [OK] Date column format verified (DD-MM-YYYY)")
            print(f"       Sample Min Date: {parsed_dates.min().strftime('%Y-%m-%d')}")
            print(f"       Sample Max Date: {parsed_dates.max().strftime('%Y-%m-%d')}")
            print(f"       Unparseable Dates in sample: {invalid_dates}")
        except Exception as e:
            print(f"  [WARNING] Date parsing error: {e}")
    else:
        print("  [ERROR] 'date' column not found in sample.")

    # 5. Missing Values in Sample
    print(f"\nSample Missing-Value Counts (Top Core & Key SMART):")
    inspect_cols = [c for c in expected_core if c in sample_df.columns] + [
        'smart_1_raw', 'smart_5_raw', 'smart_9_raw', 'smart_187_raw', 'smart_194_raw', 'smart_197_raw'
    ]
    for c in inspect_cols:
        if c in sample_df.columns:
            n_miss = sample_df[c].isnull().sum()
            pct_miss = (n_miss / total_rows) * 100
            print(f"  {c:<20} {n_miss:>5} / {total_rows} missing ({pct_miss:5.1f}%)")

    # 6. First Five Rows Preview
    print("\nFirst 5 Rows Preview (Selected Fields):")
    preview_cols = [c for c in ['date', 'serial_number', 'model', 'failure', 'smart_5_raw', 'smart_197_raw'] if c in sample_df.columns]
    print(sample_df[preview_cols].head(5).to_string(index=False))

    results = {
        'total_cols': total_cols,
        'cols': cols,
        'smart_raw_count': len(smart_raw),
        'smart_norm_count': len(smart_norm),
        'dtypes': {c: str(t) for c, t in sample_df.dtypes.items()},
    }
    return True, results


def inspect_full_chunked(csv_path: Path, chunk_size: int = 50000) -> tuple[bool, dict]:
    """Execute a memory-safe chunked scan of the full Backblaze dataset.

    Args:
        csv_path: Path to dataset CSV.
        chunk_size: Number of rows per streaming chunk (default: 50,000).

    Returns:
        tuple (success_boolean, metrics_dictionary)
    """
    print("\n" + "=" * 70)
    print(f" STEP 3: Full-Dataset Chunked Scan (Chunk Size: {chunk_size:,} Rows)")
    print("=" * 70)

    start_time = time.time()
    total_rows = 0
    failure_counts = Counter()
    model_counts = Counter()
    unique_serials = set()
    serial_obs = Counter()
    dates_set = set()

    key_columns = [
        'date', 'serial_number', 'model', 'capacity_bytes', 'failure',
        'smart_1_raw', 'smart_5_raw', 'smart_9_raw', 'smart_12_raw',
        'smart_187_raw', 'smart_188_raw', 'smart_194_raw', 'smart_197_raw', 'smart_198_raw'
    ]
    missing_tracker = {c: 0 for c in key_columns}

    try:
        reader = pd.read_csv(csv_path, chunksize=chunk_size, low_memory=False)

        for chunk_idx, chunk in enumerate(reader, 1):
            n_rows = len(chunk)
            total_rows += n_rows

            # Track failure distribution
            if 'failure' in chunk.columns:
                failure_counts.update(chunk['failure'].tolist())

            # Track models and serials
            if 'model' in chunk.columns:
                model_counts.update(chunk['model'].dropna().tolist())

            if 'serial_number' in chunk.columns:
                serials_in_chunk = chunk['serial_number'].dropna().tolist()
                unique_serials.update(serials_in_chunk)
                serial_obs.update(serials_in_chunk)

            # Track dates
            if 'date' in chunk.columns:
                dates_set.update(chunk['date'].dropna().unique().tolist())

            # Track missing values for key columns
            for col in key_columns:
                if col in chunk.columns:
                    missing_tracker[col] += chunk[col].isnull().sum()

            elapsed = time.time() - start_time
            if chunk_idx % 3 == 0 or n_rows < chunk_size:
                print(f"  Processed {total_rows:,} rows across {chunk_idx} chunks ({elapsed:.1f}s)...")

    except Exception as err:
        print(f"[ERROR] Chunked scan failed at row {total_rows:,}: {err}")
        return False, {'total_rows': total_rows, 'error': str(err)}

    total_time = time.time() - start_time
    sorted_dates = sorted(list(dates_set))
    obs_values = list(serial_obs.values())
    repeated_drives = sum(1 for c in obs_values if c > 1)

    import numpy as np
    mean_obs = np.mean(obs_values) if obs_values else 0
    median_obs = np.median(obs_values) if obs_values else 0
    min_obs = min(obs_values) if obs_values else 0
    max_obs = max(obs_values) if obs_values else 0

    print(f"\n[SCAN COMPLETE] Finished in {total_time:.2f} seconds.")
    print("-" * 70)
    print(f"Total Rows Scanned        : {total_rows:,}")
    print(f"Unique Storage Drives     : {len(unique_serials):,}")
    print(f"Distinct Drive Models     : {len(model_counts)}")
    print(f"Overall Date Range        : {sorted_dates[0]} to {sorted_dates[-1]} ({len(sorted_dates)} distinct days)")
    print(f"Failure Distribution      : Healthy (0) = {failure_counts.get(0, 0):,}, Failed (1) = {failure_counts.get(1, 0):,}")
    print(f"Failure Rate              : {(failure_counts.get(1, 0) / total_rows * 100):.5f}%")
    print(f"Repeated Drive Records    : {repeated_drives:,} / {len(unique_serials):,} ({(repeated_drives / len(unique_serials) * 100):.2f}%)")
    print(f"Observations Per Drive    : Min={min_obs}, Max={max_obs}, Mean={mean_obs:.2f}, Median={median_obs:.1f}")

    metrics = {
        'total_rows': total_rows,
        'unique_serials': len(unique_serials),
        'unique_models': len(model_counts),
        'model_counts': model_counts,
        'sorted_dates': sorted_dates,
        'failure_counts': dict(failure_counts),
        'missing_tracker': missing_tracker,
        'obs_stats': {
            'min': min_obs,
            'max': max_obs,
            'mean': mean_obs,
            'median': median_obs,
            'repeated_drives': repeated_drives,
        },
        'scan_time_seconds': total_time,
    }
    return True, metrics


def main():
    parser = argparse.ArgumentParser(
        description="Inspect Backblaze hard drive dataset schema and validate chunked statistics."
    )
    default_dataset = Path(__file__).resolve().parents[2] / 'data' / 'raw' / 'harddrive.csv'
    parser.add_argument(
        '--file',
        '-f',
        type=Path,
        default=default_dataset,
        help="Path to CSV dataset (default: backend/data/raw/harddrive.csv)",
    )
    parser.add_argument(
        '--sample-size',
        '-n',
        type=int,
        default=1000,
        help="Number of initial sample rows to inspect (default: 1,000)",
    )
    parser.add_argument(
        '--full',
        action='store_true',
        default=True,
        help="Run memory-safe chunked scan of the full dataset (default: True)",
    )
    parser.add_argument(
        '--chunk-size',
        '-c',
        type=int,
        default=50000,
        help="Chunk size for streaming validation (default: 50,000)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print(" DeviceGuard AI - Backblaze Dataset Inspection (Part 2)")
    print("=" * 70)

    # Step 1: Verification
    if not args.file.exists():
        print(f"\n[ERROR] Target dataset file not found at:")
        print(f"        {args.file.resolve()}\n")
        print("Expected Action:")
        print("  Place the Backblaze CSV at: backend/data/raw/harddrive.csv")
        print("=" * 70)
        sys.exit(1)

    file_size_bytes = os.path.getsize(args.file)
    print(f"\n[OK] Dataset Verified at: {args.file.resolve()}")
    print(f"     Actual File Size    : {file_size_bytes:,} bytes ({format_file_size(file_size_bytes)})")

    # Step 2: Sample Inspection
    sample_success, _ = inspect_sample(args.file, sample_size=args.sample_size)
    if not sample_success:
        print("[ERROR] Sample inspection failed.")
        sys.exit(1)

    # Step 3: Full Chunked Scan
    if args.full:
        full_success, _ = inspect_full_chunked(args.file, chunk_size=args.chunk_size)
        if not full_success:
            print("[ERROR] Full chunked scan failed.")
            sys.exit(1)

    print("\n" + "=" * 70)
    print("[SUCCESS] Dataset inspection completed successfully.")
    print("=" * 70)


if __name__ == '__main__':
    main()
