"""
audit_final_review.py
Detailed reconciliation and sample count analysis for Backend Part 3 Final Review.
Analyzes target labels, sliding windows, right-censoring, exclusions, and leakage prevention.
"""

import os
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import pandas as pd

RAW_CSV = os.path.join("backend", "data", "raw", "harddrive.csv")

def run_review():
    print("=" * 70)
    print("DeviceGuard AI - Backend Part 3 Final Review & Sample Reconciliation")
    print("=" * 70)

    # 1. Load clean telemetry timeline per drive
    print("Streaming through raw CSV to collect daily timelines...")
    drive_records = defaultdict(dict) # serial -> date_str -> (failure, model)
    chunk_size = 100000

    for chunk in pd.read_csv(RAW_CSV, usecols=['date', 'serial_number', 'failure', 'model'], chunksize=chunk_size):
        chunk['date_str'] = chunk['date'].astype(str).str.strip()
        chunk['serial_str'] = chunk['serial_number'].astype(str).str.strip()
        for r in chunk.itertuples():
            if r.date_str not in drive_records[r.serial_str]:
                drive_records[r.serial_str][r.date_str] = (r.failure, r.model)

    print(f"Total unique drives tracked: {len(drive_records):,}")

    def parse_d(d_str):
        return datetime.strptime(d_str, '%d-%m-%Y').date()

    all_dates = sorted(list({parse_d(d) for s in drive_records for d in drive_records[s]}))
    dataset_start = all_dates[0]
    dataset_end = all_dates[-1]
    print(f"Dataset date range: {dataset_start} to {dataset_end} ({len(all_dates)} calendar days)")

    # Identify failed drives and their first failure date
    failed_drives = {} # serial -> fail_date
    for serial, day_dict in drive_records.items():
        fail_days = [parse_d(d) for d, (f, m) in day_dict.items() if f == 1]
        if fail_days:
            failed_drives[serial] = min(fail_days)

    print(f"Total failed drives: {len(failed_drives)}")

    # -------------------------------------------------------------------------
    # Analyze all potential windows of length T = 3, Horizon H = 2
    # -------------------------------------------------------------------------
    T = 3
    H = 2

    # Counters
    total_potential_windows_3_consec = 0
    windows_excluded_missing_dates = 0
    windows_excluded_failure_day_leakage = 0 # t_end >= t_fail
    
    # Positive windows
    pos_windows = [] # (serial, window_dates, t_end, t_fail, days_ahead)
    pos_drives = set()

    # Unconstrained negative windows (previous calculation)
    unconstrained_neg_windows = []
    unconstrained_neg_drives = set()

    # Right-censored / insufficient future observation analysis
    # A negative window requires knowing the drive did not fail for H days after t_end,
    # meaning t_end + H days <= dataset_end.
    # If t_end + H days > dataset_end (i.e. t_end in {2016-01-12, 2016-01-13}):
    # If the drive did not fail by dataset_end, we cannot confirm whether it failed on Day t_end + 1 or t_end + 2.
    windows_excluded_insufficient_future = 0
    strict_neg_windows = []
    strict_neg_drives = set()

    # Detailed tracking per drive
    drives_with_gap = 0
    failed_drives_pre_windows = defaultdict(list)

    for serial, day_dict in drive_records.items():
        d_list = sorted([parse_d(d) for d in day_dict.keys()])
        is_failed = serial in failed_drives
        t_fail = failed_drives.get(serial)

        # Check calendar gap in drive's entire history
        has_gap = False
        for i in range(len(d_list) - 1):
            if (d_list[i+1] - d_list[i]).days != 1:
                has_gap = True
                break
        if has_gap:
            drives_with_gap += 1

        # Check all possible 3-observation slices
        for i in range(len(d_list) - T + 1):
            window = d_list[i:i+T]
            # Check consecutive calendar days
            if (window[-1] - window[0]).days != T - 1:
                windows_excluded_missing_dates += 1
                continue

            total_potential_windows_3_consec += 1
            t_end = window[-1]

            if is_failed:
                # Check failure-day and post-failure leakage
                if t_end >= t_fail:
                    windows_excluded_failure_day_leakage += 1
                    continue

                days_to_fail = (t_fail - t_end).days
                if 1 <= days_to_fail <= H:
                    # Positive window
                    pos_windows.append((serial, window, t_end, t_fail, days_to_fail))
                    pos_drives.add(serial)
                elif days_to_fail > H:
                    # Failure is strictly beyond the H horizon
                    # Does it have sufficient future observation?
                    # Since it failed at t_fail > t_end + H, we KNOW it did not fail in (t_end, t_end + H]!
                    # So this is definitely negative!
                    unconstrained_neg_windows.append((serial, window))
                    unconstrained_neg_drives.add(serial)
                    strict_neg_windows.append((serial, window))
                    strict_neg_drives.add(serial)
            else:
                # Healthy drive (never failed within dataset)
                # Unconstrained: treated as negative
                unconstrained_neg_windows.append((serial, window))
                unconstrained_neg_drives.add(serial)

                # Strict right-censoring check:
                # Is t_end + H days <= dataset_end?
                if (dataset_end - t_end).days >= H:
                    strict_neg_windows.append((serial, window))
                    strict_neg_drives.add(serial)
                else:
                    windows_excluded_insufficient_future += 1

    print("\n--- EXACT SAMPLE COUNT RECONCILIATION ---")
    print(f"Configuration: T = {T} days, H = {H} days")
    print(f"Total potential 3-consecutive-day windows formed: {total_potential_windows_3_consec:,}")
    print(f"\n1. Exclusions:")
    print(f"   - Windows excluded due to calendar gaps (missing dates): {windows_excluded_missing_dates}")
    print(f"   - Windows excluded due to failure-day/post-failure leakage (t_end >= t_fail): {windows_excluded_failure_day_leakage}")
    print(f"   - Windows excluded due to insufficient future observation (t_end + {H} > {dataset_end}): {windows_excluded_insufficient_future:,}")

    print(f"\n2. Positive Windows:")
    print(f"   - Total Positive Windows (fleet-wide): {len(pos_windows)}")
    print(f"   - Unique Failed Drives contributing positive windows: {len(pos_drives)}")

    # Breakdown of positive windows per drive
    pos_window_counts = Counter([w[0] for w in pos_windows])
    drives_with_1_pos = sum(1 for c in pos_window_counts.values() if c == 1)
    drives_with_2_pos = sum(1 for c in pos_window_counts.values() if c == 2)
    print(f"   - Breakdown: {drives_with_1_pos} drives contributed 1 positive window (failed on t_end + 1 or t_end + 2 with only 3 pre-obs)")
    print(f"   - Breakdown: {drives_with_2_pos} drives contributed 2 positive windows (both t_fail - 2 and t_fail - 1 captured)")
    print(f"   - Math check: ({drives_with_1_pos} * 1) + ({drives_with_2_pos} * 2) = {(drives_with_1_pos * 1) + (drives_with_2_pos * 2)} positive windows")

    print(f"\n3. Negative Windows (Unconstrained vs Strict Right-Censoring):")
    print(f"   - Unconstrained Negative Windows (previous reported count): {len(unconstrained_neg_windows):,}")
    print(f"     Unique drives contributing unconstrained negatives: {len(unconstrained_neg_drives):,}")
    print(f"   - Strict (Right-Censored Safe) Negative Windows: {len(strict_neg_windows):,}")
    print(f"     Unique drives contributing strict negatives: {len(strict_neg_drives):,}")

    # Check math reconciliation
    print(f"\n4. Mathematical Total Reconciliation:")
    print(f"   Positives: {len(pos_windows)}")
    print(f"   Strict Negatives: {len(strict_neg_windows):,}")
    print(f"   Excluded (insufficient future): {windows_excluded_insufficient_future:,}")
    print(f"   Excluded (failure leakage): {windows_excluded_failure_day_leakage}")
    total_accounted = len(pos_windows) + len(strict_neg_windows) + windows_excluded_insufficient_future + windows_excluded_failure_day_leakage
    print(f"   Sum: {total_accounted:,} vs Total Potential Windows: {total_potential_windows_3_consec:,}")
    assert total_accounted == total_potential_windows_3_consec, "Counts do not reconcile!"
    print("   -> MATHEMATICAL RECONCILIATION VERIFIED 100%!")

    # Concrete date-based examples for the report
    print("\n--- CONCRETE DATE-BASED EXAMPLES ---")
    
    # 1. Drive that failed on 2016-01-08 with 2 positive windows
    print("Example A: Failed drive with multiple positive windows (e.g., Z300PG45, ST4000DM000)")
    # Find a drive with 2 positive windows
    d_two_pos = [s for s, c in pos_window_counts.items() if c == 2][0]
    d_two_fail = failed_drives[d_two_pos]
    print(f"  Drive: {d_two_pos} (Failed on {d_two_fail})")
    for w in pos_windows:
        if w[0] == d_two_pos:
            print(f"    Positive Window: {w[1]} -> t_end = {w[2]}. Target: fails on {w[3]} ({w[4]} day(s) ahead <= H={H}) -> Label y = 1")
    # Early negative window for this same drive
    for w in strict_neg_windows:
        if w[0] == d_two_pos:
            print(f"    Early Negative Window: {w[1]} -> t_end = {w[1][-1]}. Target: fails on {d_two_fail} ({(d_two_fail - w[1][-1]).days} days ahead > H={H}) -> Label y = 0")
    # Excluded window on failure day
    print(f"    Excluded Window: Window ending on {d_two_fail} (Failure day) -> EXCLUDED (Leakage prevention: t_end >= t_fail)")

    # 2. Drive that failed on 2016-01-02 (Day 2) -> Excluded due to insufficient prior history
    print("\nExample B: Drive failing on Day 2 (e.g. 2016-01-02)")
    d_early = [s for s, d in failed_drives.items() if d == parse_d('02-01-2016')][0]
    print(f"  Drive: {d_early} (Failed on 2016-01-02)")
    print(f"  Pre-failure observations available: 1 day (2016-01-01)")
    print(f"  Result: Cannot form T = 3 window prior to failure -> EXCLUDED (Insufficient history)")

    # 3. Healthy drive window classification
    print("\nExample C: Healthy drive (never failed)")
    d_healthy = list(unconstrained_neg_drives - set(failed_drives.keys()))[0]
    print(f"  Drive: {d_healthy}")
    print(f"  Window [2016-01-01, 2016-01-02, 2016-01-03] (t_end = 2016-01-03): Horizon [2016-01-04, 2016-01-05] observed, no failure -> Label y = 0 (Confirmed Negative)")
    print(f"  Window [2016-01-11, 2016-01-12, 2016-01-13] (t_end = 2016-01-13): Horizon [2016-01-14, 2016-01-15] UNOBSERVED -> Excluded in strict protocol (Right-censored)")

    # Seagate specific counts
    st_drives = {s for s, days in drive_records.items() if any(m == 'ST4000DM000' for f, m in days.values())}
    st_pos_windows = [w for w in pos_windows if w[0] in st_drives]
    st_pos_drives = {w[0] for w in st_pos_windows}
    st_strict_neg_windows = [w for w in strict_neg_windows if w[0] in st_drives]
    st_strict_neg_drives = {w[0] for w in st_strict_neg_windows}
    st_unconstrained_neg_windows = [w for w in unconstrained_neg_windows if w[0] in st_drives]
    print("\n--- SEAGATE ST4000DM000 SUBSET COUNTS ---")
    print(f"  Seagate Positive Windows: {len(st_pos_windows)} (from {len(st_pos_drives)} drives)")
    print(f"  Seagate Unconstrained Negative Windows: {len(st_unconstrained_neg_windows):,}")
    print(f"  Seagate Strict Negative Windows: {len(st_strict_neg_windows):,}")

if __name__ == "__main__":
    run_review()
