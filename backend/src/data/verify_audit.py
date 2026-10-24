"""
audit_verification.py
Deep audit and verification of Backblaze dataset for DeviceGuard AI Backend Part 3.
Investigates duplicates, failure counts, date continuity, sequence windows, and horizons.
"""

import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from collections import defaultdict

RAW_CSV = os.path.join("backend", "data", "raw", "harddrive.csv")

def run_audit():
    print("=" * 70)
    print("DeviceGuard AI - Rigorous Preprocessing Audit & Verification")
    print("=" * 70)

    if not os.path.exists(RAW_CSV):
        print(f"ERROR: {RAW_CSV} not found!")
        return

    file_size = os.path.getsize(RAW_CSV)
    print(f"Raw CSV: {RAW_CSV} ({file_size:,} bytes)")

    # ---------------------------------------------------------
    # 1. Investigate Duplicates & Conflicts
    # ---------------------------------------------------------
    print("\n--- TASK 1: Duplicate & Conflict Investigation ---")
    
    # We will read key columns + check row identity
    # Since reading 757,765 rows of all 95 columns takes ~500MB RAM,
    # let's test if we can do this chunked or read selective columns.
    # To check EXACT duplicate across all 95 columns, we can compute MD5 or hash of each row,
    # or read in chunks and check.
    import hashlib

    row_count = 0
    date_serial_map = defaultdict(list) # (date, serial) -> list of (row_idx, failure, model)
    row_hashes = {} # row_hash -> first_seen_row_idx
    exact_duplicate_indices = []
    date_serial_counts = defaultdict(int)
    date_counts = defaultdict(int)
    failure_rows_raw = []

    # Stream through CSV with chunksize 100,000
    chunk_size = 100000
    print("Streaming through all 95 columns to verify exact vs partial duplicates...")

    # Let's inspect column names first
    first_chunk = pd.read_csv(RAW_CSV, nrows=5)
    cols = list(first_chunk.columns)
    print(f"Total columns in CSV: {len(cols)}")

    # We want to check:
    # 1. Exact duplicates across all 95 columns
    # 2. Duplicate (date, serial_number)
    # 3. Any conflicting failure labels or SMART values for identical (date, serial_number)

    conflicting_labels = []
    conflicting_features = 0
    date_distribution_dups = defaultdict(int)

    # We will store for each (date, serial_number) a representative hash of the entire row
    seen_date_serial = {} # (date, serial) -> (row_idx, full_row_hash, failure_val)

    exact_dup_count = 0
    date_serial_dup_count = 0

    for chunk_idx, chunk in enumerate(pd.read_csv(RAW_CSV, chunksize=chunk_size, low_memory=False)):
        # Normalize date format string
        chunk['date_str'] = chunk['date'].astype(str).str.strip()
        chunk['serial_str'] = chunk['serial_number'].astype(str).str.strip()
        
        # Check raw failure rows
        fail_sub = chunk[chunk['failure'] == 1]
        for idx, r in fail_sub.iterrows():
            failure_rows_raw.append({
                'row_idx': idx,
                'date': r['date_str'],
                'serial_number': r['serial_str'],
                'model': r['model']
            })

        # Count dates
        for d, c in chunk['date_str'].value_counts().items():
            date_counts[d] += c

        # Check full row duplicates
        # To be completely rigorous, let's hash the string representation of all columns for each row
        # Or use pandas .duplicated() within chunks or globally
        for row_tuple in chunk[['date_str', 'serial_str', 'failure', 'model']].itertuples():
            key = (row_tuple.date_str, row_tuple.serial_str)
            date_serial_counts[key] += 1
            if date_serial_counts[key] > 1:
                date_serial_dup_count += 1
                date_distribution_dups[row_tuple.date_str] += 1

        row_count += len(chunk)
        print(f"  Processed {row_count:,} rows...")

    print(f"\nTotal raw rows scanned: {row_count:,}")
    print(f"Unique (date, serial_number) pairs: {len(date_serial_counts):,}")
    print(f"Total duplicate (date, serial_number) occurrences: {date_serial_dup_count:,}")
    print(f"Dates where duplicates occur:")
    for d, cnt in date_distribution_dups.items():
        print(f"  - {d}: {cnt:,} duplicate occurrences")

    # Now verify if ALL 57,545 duplicates on 01-01-2016 are EXACT duplicates across all 95 columns
    print("\nVerifying if duplicates on 01-01-2016 are 100% exact across all 95 columns...")
    # Read all rows of 01-01-2016
    df_day1 = []
    for chunk in pd.read_csv(RAW_CSV, chunksize=chunk_size, low_memory=False):
        c1 = chunk[chunk['date'].astype(str).str.strip() == '01-01-2016']
        if len(c1) > 0:
            df_day1.append(c1)
    df_day1 = pd.concat(df_day1, ignore_index=True)
    print(f"Total rows on 01-01-2016: {len(df_day1):,}")
    
    # Check exact duplicates across all 95 columns in day 1
    exact_dups_day1 = df_day1.duplicated(keep='first').sum()
    unique_rows_day1 = len(df_day1) - exact_dups_day1
    unique_serials_day1 = df_day1['serial_number'].nunique()
    print(f"Exact duplicates across ALL 95 columns on 01-01-2016: {exact_dups_day1:,}")
    print(f"Unique rows on 01-01-2016: {unique_rows_day1:,}")
    print(f"Unique serial numbers on 01-01-2016: {unique_serials_day1:,}")

    # Check if any (date, serial_number) on any other date is duplicated
    other_date_dups = {d: cnt for d, cnt in date_distribution_dups.items() if d != '01-01-2016'}
    print(f"Duplicates on other dates: {other_date_dups}")

    # Check for any conflicting feature values or failure labels
    # Check if unique_rows_day1 == unique_serials_day1
    conflicts_found = (unique_rows_day1 != unique_serials_day1)
    print(f"Are there any conflicting rows for the same serial number on 01-01-2016? {conflicts_found}")
    if conflicts_found:
        print("WARNING: Found differences between unique rows and unique serial numbers on Day 1!")

    # ---------------------------------------------------------
    # 2. Verify Failure Counts & Details
    # ---------------------------------------------------------
    print("\n--- TASK 2: Failure Counts Re-calculation ---")
    df_fails = pd.DataFrame(failure_rows_raw)
    print(f"Total raw failure-labelled rows (failure == 1): {len(df_fails):,}")
    print(f"Unique serial numbers with failure == 1 in raw CSV: {df_fails['serial_number'].nunique():,}")
    
    # Failure breakdown by date in raw data
    raw_fail_dates = df_fails['date'].value_counts().sort_index()
    print("\nRaw failure rows by date:")
    for d, c in raw_fail_dates.items():
        print(f"  {d}: {c} failure rows")

    # Check which failure rows are duplicates
    fail_dups = df_fails[df_fails.duplicated(subset=['date', 'serial_number'], keep=False)]
    print(f"\nFailure rows involved in duplication: {len(fail_dups)} rows")
    if len(fail_dups) > 0:
        print("Duplicated failure records:")
        print(fail_dups[['date', 'serial_number', 'model']])

    # Deduplicated failure records
    df_fails_dedup = df_fails.drop_duplicates(subset=['date', 'serial_number'])
    print(f"\nUnique (date, serial_number) failure records: {len(df_fails_dedup)}")
    print(f"Unique serial numbers with failure after deduplication: {df_fails_dedup['serial_number'].nunique()}")

    # Check if any drive failed on MULTIPLE DATES
    drive_fail_date_counts = df_fails_dedup.groupby('serial_number')['date'].nunique()
    multi_fail_drives = drive_fail_date_counts[drive_fail_date_counts > 1]
    print(f"Drives with failure recorded on multiple different dates: {len(multi_fail_drives)}")

    failed_serial_set = set(df_fails_dedup['serial_number'].unique())

    # ---------------------------------------------------------
    # 3. Post-Failure Telemetry & Date Continuity
    # ---------------------------------------------------------
    print("\n--- TASK 3: Date Continuity & Post-Failure Telemetry ---")
    
    # We need to collect the full observation timeline for every drive (or especially failed drives and a sample of healthy drives)
    # Let's collect for all failed drives their full list of (date, failure) records
    failed_drive_records = defaultdict(list)
    drive_date_sets = defaultdict(set) # serial -> set of dates
    drive_models = {}

    print("Collecting daily observation timelines for all drives across dataset...")
    # To be memory efficient, we can store dates for each serial
    # We have 57,659 unique serials and 13 days
    for chunk in pd.read_csv(RAW_CSV, usecols=['date', 'serial_number', 'failure', 'model'], chunksize=chunk_size):
        chunk['date_str'] = chunk['date'].astype(str).str.strip()
        chunk['serial_str'] = chunk['serial_number'].astype(str).str.strip()
        
        for r in chunk.itertuples():
            drive_date_sets[r.serial_str].add(r.date_str)
            if r.serial_str in failed_serial_set:
                failed_drive_records[r.serial_str].append((r.date_str, r.failure))
            if r.serial_str not in drive_models:
                drive_models[r.serial_str] = r.model

    print(f"Total drives tracked: {len(drive_date_sets):,}")

    # Convert dates to datetime objects
    # Date formats in CSV: e.g. '01-01-2016' or '13-01-2016'
    def parse_d(d_str):
        # Could be DD-MM-YYYY
        return datetime.strptime(d_str, '%d-%m-%Y').date()

    all_dates = sorted(list({parse_d(d) for s in drive_date_sets for d in drive_date_sets[s]}))
    print(f"Calendar Date Range: {all_dates[0]} to {all_dates[-1]} ({len(all_dates)} distinct dates)")
    print(f"All dates: {[str(d) for d in all_dates]}")

    # Check post-failure observations for failed drives
    post_failure_violations = 0
    pre_failure_obs_counts = {}
    failure_day_info = {}

    for serial, records in failed_drive_records.items():
        # Deduplicate records by date
        records_by_date = {}
        for d_str, fail_val in records:
            if d_str not in records_by_date:
                records_by_date[d_str] = fail_val
            else:
                # verify consistency
                if records_by_date[d_str] != fail_val:
                    print(f"Conflict on {serial} {d_str}: {records_by_date[d_str]} vs {fail_val}")

        # Find failure date
        fail_dates = [parse_d(d) for d, f in records_by_date.items() if f == 1]
        fail_date = min(fail_dates)
        failure_day_info[serial] = fail_date

        # Check if any observation exists strictly AFTER fail_date
        post_obs = [parse_d(d) for d in records_by_date if parse_d(d) > fail_date]
        if len(post_obs) > 0:
            post_failure_violations += 1
            print(f"  Drive {serial} has {len(post_obs)} observations AFTER failure date {fail_date}: {post_obs}")

        # Check pre-failure observations (strictly before fail_date)
        pre_obs = sorted([parse_d(d) for d in records_by_date if parse_d(d) < fail_date])
        pre_failure_obs_counts[serial] = pre_obs

    print(f"\nFailed drives reporting telemetry AFTER their failure date: {post_failure_violations}")
    if post_failure_violations == 0:
        print("  -> Confirmation: Once a drive fails, it ceases reporting telemetry in subsequent days.")

    # Check date continuity for all drives
    # Are there missing dates / gaps?
    gaps_count = 0
    continuous_drives = 0
    total_drives = len(drive_date_sets)

    for serial, d_strs in drive_date_sets.items():
        d_list = sorted([parse_d(d) for d in d_strs])
        # Check if d_list forms a contiguous range of dates
        is_continuous = True
        for i in range(len(d_list) - 1):
            if (d_list[i+1] - d_list[i]).days != 1:
                is_continuous = False
                break
        if is_continuous:
            continuous_drives += 1
        else:
            gaps_count += 1

    print(f"\nDate continuity across all {total_drives:,} drives:")
    print(f"  Continuous observation timeline (no missing calendar days): {continuous_drives:,} ({continuous_drives/total_drives*100:.2f}%)")
    print(f"  Discontinuous observation timeline (has calendar gaps): {gaps_count:,} ({gaps_count/total_drives*100:.2f}%)")

    # ---------------------------------------------------------
    # 4. Rigorous Sliding Sequence Feasibility & Example Counting
    # ---------------------------------------------------------
    print("\n--- TASK 4: Exact Sequence Feasibility & Usable Examples ---")
    # Let's evaluate sequence lengths T in {3, 7}
    # And prediction horizons H in {1, 2, 3}
    
    # Let's define the precise rules:
    # An input sequence of length T is a tuple of T consecutive calendar days:
    # [d, d + 1 day, ..., d + (T-1) days].
    # For a drive, this sequence can only be formed if the drive has actual observations on every single one of those T days.
    # The end of the input window is t_end = d + (T-1) days.
    # For a failed drive, to prevent leakage:
    # The input window MUST strictly precede the failure date: t_end < t_failure.
    #
    # Target definition for horizon H:
    # If the drive fails on date t_failure where:
    # t_end < t_failure <= t_end + H days
    # then this sequence is a POSITIVE example (y = 1).
    #
    # If the drive never fails in the dataset:
    # This sequence is a NEGATIVE example (y = 0).
    #
    # What if the drive fails, but t_failure > t_end + H days?
    # (i.e. Failure is beyond the horizon H):
    # Option A: In many failure prediction studies, this is treated as a negative example (y = 0) because failure is not imminent within H days.
    # Option B: Or it is buffered/excluded.
    # Let's count BOTH:
    # - Positives (fails in [t_end + 1, t_end + H])
    # - Healthy Negatives (from drives that never fail)
    # - Early Negatives (from failed drives where failure > t_end + H)
    # - Excluded (t_end >= t_failure, i.e. on or after failure day)

    for T in [3, 7]:
        print(f"\n=======================================================")
        print(f" SEQUENCE LENGTH T = {T} CONSECUTIVE DAYS")
        print(f"=======================================================")

        # Check how many failed drives have at least T consecutive days strictly before failure
        failed_drives_with_T_pre = []
        for serial, pre_obs in pre_failure_obs_counts.items():
            # Check if pre_obs contains at least T consecutive days
            has_T_consec = False
            for i in range(len(pre_obs) - T + 1):
                window = pre_obs[i:i+T]
                if (window[-1] - window[0]).days == T - 1:
                    has_T_consec = True
                    break
            if has_T_consec:
                failed_drives_with_T_pre.append(serial)

        print(f"Failed drives with >= {T} consecutive pre-failure observations: {len(failed_drives_with_T_pre)} / {len(failed_serial_set)}")

        # Evaluate horizons H in {1, 2, 3}
        for H in [1, 2, 3]:
            positive_examples = 0
            positive_drives = set()
            negative_examples_healthy = 0
            negative_examples_failed_drive_early = 0

            # Scan all drives
            for serial, d_strs in drive_date_sets.items():
                d_list = sorted([parse_d(d) for d in d_strs])
                is_failed = serial in failed_serial_set
                fail_d = failure_day_info.get(serial, None)

                # Find all valid consecutive windows of length T
                for i in range(len(d_list) - T + 1):
                    window = d_list[i:i+T]
                    # Check continuity
                    if (window[-1] - window[0]).days != T - 1:
                        continue # gap in calendar days

                    t_end = window[-1]

                    if is_failed:
                        # Must strictly precede failure
                        if t_end >= fail_d:
                            continue # Excluded (leakage prevention: on or after failure day)

                        # Check if failure occurs within [t_end + 1, t_end + H]
                        days_until_fail = (fail_d - t_end).days
                        if 1 <= days_until_fail <= H:
                            positive_examples += 1
                            positive_drives.add(serial)
                        elif days_until_fail > H:
                            negative_examples_failed_drive_early += 1
                    else:
                        # Drive never fails
                        negative_examples_healthy += 1

            total_negatives = negative_examples_healthy + negative_examples_failed_drive_early
            print(f"\n  Horizon H = {H} day(s) (Failure occurs within 1 to {H} days ahead):")
            print(f"    - Valid Positive Sequence Windows : {positive_examples}")
            print(f"    - Unique Failed Drives in Positives: {len(positive_drives)}")
            print(f"    - Negative Sequences (Healthy)    : {negative_examples_healthy:,}")
            print(f"    - Negative Sequences (Pre-buffer) : {negative_examples_failed_drive_early}")
            print(f"    - Total Negative Sequence Windows : {total_negatives:,}")
            if positive_examples > 0:
                print(f"    - Class Ratio (Neg : Pos)         : {total_negatives // positive_examples}:1")
            else:
                print(f"    - Class Ratio (Neg : Pos)         : N/A (0 positives)")

    # ---------------------------------------------------------
    # 5. Vendor / Model Breakdown for ST4000DM000
    # ---------------------------------------------------------
    print("\n--- TASK 5: Seagate ST4000DM000 Model-Specific Subset ---")
    st_drives = {s for s, m in drive_models.items() if m == 'ST4000DM000'}
    st_failed = st_drives.intersection(failed_serial_set)
    print(f"Seagate ST4000DM000 drives: {len(st_drives):,} ({len(st_drives)/len(drive_models)*100:.2f}% of fleet)")
    print(f"Seagate ST4000DM000 failures: {len(st_failed)} ({len(st_failed)/len(failed_serial_set)*100:.2f}% of all failures)")

    for T in [3, 7]:
        for H in [1, 2, 3]:
            st_pos_examples = 0
            st_pos_drives = set()
            st_neg_examples = 0

            for serial in st_drives:
                d_list = sorted([parse_d(d) for d in drive_date_sets[serial]])
                is_failed = serial in failed_serial_set
                fail_d = failure_day_info.get(serial, None)

                for i in range(len(d_list) - T + 1):
                    window = d_list[i:i+T]
                    if (window[-1] - window[0]).days != T - 1:
                        continue
                    t_end = window[-1]

                    if is_failed:
                        if t_end >= fail_d:
                            continue
                        days_until_fail = (fail_d - t_end).days
                        if 1 <= days_until_fail <= H:
                            st_pos_examples += 1
                            st_pos_drives.add(serial)
                        elif days_until_fail > H:
                            st_neg_examples += 1
                    else:
                        st_neg_examples += 1

            print(f"  ST4000DM000 [T={T}, H={H}]: Positives={st_pos_examples} (from {len(st_pos_drives)} drives), Negatives={st_neg_examples:,}")

if __name__ == "__main__":
    run_audit()
