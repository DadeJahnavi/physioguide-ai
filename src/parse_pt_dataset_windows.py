# src/parse_pt_dataset_windows.py
"""
Parse the pt_exercises dataset into sliding windows and extract features.

Outputs:
  data/imu_windows.csv
Config:
  WINDOW = 50
  STEP   = 25
Adjust CORRECT_U_FOLDERS if your mapping differs.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import signal
import math

ROOT_DIR = Path("data/pt_exercises")
OUT_CSV = Path("data/imu_windows.csv")

WINDOW = 50    # samples per window (recommended)
STEP = 25      # hop size (50% overlap)

CORRECT_U_FOLDERS = {"u1", "u2"}    # default mapping (change if needed)
INCORRECT_U_FOLDERS = {"u3", "u4", "u5"}

def safe_read_imu(file_path):
    try:
        df = pd.read_csv(file_path, sep=';', engine='python')
        return df
    except Exception as e:
        print(f"Failed to read {file_path}: {e}")
        return None

def find_col_indices(cols):
    lower = [c.lower().strip() for c in cols]
    def idx_of(prefixes):
        for p in prefixes:
            for i,c in enumerate(lower):
                if c.startswith(p):
                    return i
        return None
    ax_i = idx_of(["acc_x","accx","acc x","accel_x"])
    ay_i = idx_of(["acc_y","accy","acc y","accel_y"])
    az_i = idx_of(["acc_z","accz","acc z","accel_z"])
    gx_i = idx_of(["gyr_x","gyrx","gyr x","gyro_x"])
    gy_i = idx_of(["gyr_y","gyry","gyr y","gyro_y"])
    gz_i = idx_of(["gyr_z","gyrz","gyr z","gyro_z"])
    return ax_i,ay_i,az_i,gx_i,gy_i,gz_i

def windowize_and_extract(df, window=WINDOW, step=STEP):
    # returns list of dicts (features) for windows
    cols = list(df.columns)
    ax_i,ay_i,az_i,gx_i,gy_i,gz_i = find_col_indices(cols)
    if ax_i is None or ay_i is None or az_i is None:
        return []

    data = df.values.astype(float)
    n = data.shape[0]
    rows = []
    for start in range(0, max(1, n - window + 1), step):
        w = data[start:start+window, :]
        # convert columns using indices
        ax = w[:, ax_i]
        ay = w[:, ay_i]
        az = w[:, az_i]
        gyr_x = w[:, gx_i] if gx_i is not None else None
        gyr_y = w[:, gy_i] if gy_i is not None else None
        gyr_z = w[:, gz_i] if gz_i is not None else None

        feats = {}
        # basic time-domain features for each axis
        for name, arr in (("ax", ax), ("ay", ay), ("az", az)):
            feats[f"{name}_mean"] = float(np.nanmean(arr))
            feats[f"{name}_std"] = float(np.nanstd(arr))
            feats[f"{name}_min"] = float(np.nanmin(arr))
            feats[f"{name}_max"] = float(np.nanmax(arr))
            feats[f"{name}_median"] = float(np.nanmedian(arr))
            feats[f"{name}_rms"] = float(np.sqrt(np.nanmean(np.square(arr))))
            feats[f"{name}_ptp"] = float(np.nanmax(arr) - np.nanmin(arr))
            # simple shape features
            feats[f"{name}_skew"] = float(pd.Series(arr).skew())
            feats[f"{name}_kurtosis"] = float(pd.Series(arr).kurt())

        # magnitude features
        acc_mag = np.sqrt(ax**2 + ay**2 + az**2)
        feats["acc_mag_mean"] = float(np.nanmean(acc_mag))
        feats["acc_mag_std"] = float(np.nanstd(acc_mag))
        feats["acc_mag_ptp"] = float(np.nanmax(acc_mag) - np.nanmin(acc_mag))
        # jerk (difference)
        jerk = np.diff(acc_mag)
        feats["acc_jerk_mean"] = float(np.nanmean(jerk)) if len(jerk)>0 else 0.0
        feats["acc_jerk_std"] = float(np.nanstd(jerk)) if len(jerk)>0 else 0.0

        # gyro magnitude features if present
        if gyr_x is not None and gyr_y is not None and gyr_z is not None:
            gm = np.sqrt(gyr_x**2 + gyr_y**2 + gyr_z**2)
            feats["gyr_mag_mean"] = float(np.nanmean(gm))
            feats["gyr_mag_std"] = float(np.nanstd(gm))
            feats["gyr_mag_ptp"] = float(np.nanmax(gm) - np.nanmin(gm))
        else:
            feats["gyr_mag_mean"] = 0.0
            feats["gyr_mag_std"] = 0.0
            feats["gyr_mag_ptp"] = 0.0

        # small frequency features (dominant freq, power) computed on acc_mag
        try:
            if len(acc_mag) >= 8:
                f, Pxx = signal.welch(acc_mag, nperseg=min(128, len(acc_mag)))
                feats["acc_dom_freq"] = float(f[np.argmax(Pxx)])
                feats["acc_power_sum"] = float(np.sum(Pxx))
            else:
                feats["acc_dom_freq"] = 0.0
                feats["acc_power_sum"] = 0.0
        except Exception:
            feats["acc_dom_freq"] = 0.0
            feats["acc_power_sum"] = 0.0

        feats["n_samples"] = int(len(acc_mag))
        rows.append(feats)
    return rows

def infer_correctness_from_u(u_name):
    if u_name in CORRECT_U_FOLDERS:
        return "correct"
    if u_name in INCORRECT_U_FOLDERS:
        return "incorrect"
    return "unknown"

def main():
    rows = []
    if not ROOT_DIR.exists():
        print("ROOT_DIR not found:", ROOT_DIR)
        return
    for s_path in sorted(ROOT_DIR.glob("s*")):
        subject = s_path.name
        for e_path in sorted(s_path.glob("e*")):
            exercise = e_path.name
            for u_path in sorted(e_path.glob("u*")):
                u_name = u_path.name
                # choose file: prefer template_session, then test, then any file
                file_to_read = None
                for fname in ("template_session","test"):
                    f = u_path / fname
                    if f.exists():
                        file_to_read = f
                        break
                if file_to_read is None:
                    candidate = list(u_path.glob("*"))
                    if candidate:
                        file_to_read = candidate[0]
                if file_to_read is None:
                    continue
                df = safe_read_imu(file_to_read)
                if df is None or df.shape[0] < 8:
                    continue
                window_rows = windowize_and_extract(df, window=WINDOW, step=STEP)
                # attach metadata to each window row
                for w in window_rows:
                    w["subject"] = subject
                    w["exercise"] = exercise
                    w["u_folder"] = u_name
                    w["correctness"] = infer_correctness_from_u(u_name)
                    w["file"] = str(file_to_read)
                    rows.append(w)

    if not rows:
        print("No windows extracted. Check ROOT_DIR path and files.")
        return
    out = pd.DataFrame(rows)
    # add numeric labels
    out["exercise_label"] = out["exercise"].astype("category").cat.codes
    out["correctness_label"] = out["correctness"].astype("category").cat.codes
    out.to_csv(OUT_CSV, index=False)
    print(f"Saved windows dataset with {len(out)} rows to {OUT_CSV}")

if __name__ == "__main__":
    main()
