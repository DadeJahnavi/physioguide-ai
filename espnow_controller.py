# espnow_controller.py
# FINAL STABLE VERSION
# - Reads IMU data from ESP32 Hub
# - Builds EXACT 39-feature vector (38 + dummy)
# - Runs correctness ML model
# - Sends manual + automatic vibration commands
# - Writes imu_status.json for dashboard

import warnings
warnings.filterwarnings("ignore")

import serial
import serial.tools.list_ports
import time
import json
import os
import joblib
import numpy as np
from collections import defaultdict, deque
from scipy.stats import skew, kurtosis
from threading import Thread
from pynput import keyboard

# =========================================================
# CONFIG
# =========================================================
BAUD = 115200
READ_TIMEOUT = 0.2
WINDOW_SIZE = 50
STATUS_FILE = "imu_status.json"

CORR_MODEL_F = "imu_correctness_model.pkl"
CORR_SCALER_F = "imu_correctness_scaler.pkl"

# Exercise-specific thresholds
EXERCISE_CONFIG = {
    "shoulder_flexion_up": dict(movement=1.2, confidence=0.75, cooldown=2.0),
    "shoulder_flexion_down": dict(movement=1.1, confidence=0.75, cooldown=2.0),
    "arm_raise_side": dict(movement=1.3, confidence=0.70, cooldown=2.0),
    "glute_bridge_up": dict(movement=1.0, confidence=0.80, cooldown=2.5),
    "glute_bridge_down": dict(movement=1.0, confidence=0.80, cooldown=2.5),
    "knee_raise_up": dict(movement=1.2, confidence=0.70, cooldown=1.8),
    "knee_raise_down": dict(movement=1.2, confidence=0.70, cooldown=1.8),
}

CURRENT_EXERCISE = "shoulder_flexion_up"

# =========================================================
# GLOBAL STATE
# =========================================================
buffers = defaultdict(lambda: deque(maxlen=WINDOW_SIZE))
last_auto_vibe = defaultdict(lambda: 0.0)
last_manual_vibe = defaultdict(lambda: 0.0)
last_seen = {}

imu_status = {
    "nodes": {},
    "logs": [],
    "current_exercise": {"name": CURRENT_EXERCISE}
}

corr_model = None
corr_scaler = None

# =========================================================
# SERIAL AUTO-DETECT
# =========================================================
def autodetect_serial():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        name = (p.description or "").lower()
        if any(k in name for k in ["usb", "esp", "ch340", "cp210"]):
            print("Auto-detected serial:", p.device)
            return p.device
    return None

# =========================================================
# LOAD MODELS
# =========================================================
def load_models():
    global corr_model, corr_scaler
    corr_model = joblib.load(CORR_MODEL_F)
    corr_scaler = joblib.load(CORR_SCALER_F)
    print("[OK] ML models loaded")

# =========================================================
# SERIAL PARSER (ROBUST)
# =========================================================
def parse_serial_line(raw):
    try:
        line = raw.decode("utf-8", errors="ignore").strip()
    except:
        return None, None

    if not line.startswith("IMU"):
        return None, None

    try:
        head, rest = line.split(":")
        node = int(head.replace("IMU", "").strip())
        acc, gyr = rest.split("|")

        ax, ay, az = [float(x) for x in acc.split(",")]
        gx, gy, gz = [float(x) for x in gyr.split(",")]

        return node, dict(ax=ax, ay=ay, az=az, gx=gx, gy=gy, gz=gz)
    except:
        return None, None

# =========================================================
# FEATURE BUILDER (38 + dummy = 39)
# =========================================================
def build_features(window):
    ax = np.array([s["ax"] for s in window])
    ay = np.array([s["ay"] for s in window])
    az = np.array([s["az"] for s in window])

    def rms(x): return np.sqrt(np.mean(x ** 2))
    def ptp(x): return np.ptp(x)

    feats = []

    for arr in [ax, ay, az]:
        feats += [
            arr.mean(), arr.std(), arr.min(), arr.max(),
            np.median(arr), rms(arr), ptp(arr),
            skew(arr), kurtosis(arr)
        ]

    acc_mag = np.sqrt(ax**2 + ay**2 + az**2)
    acc_jerk = np.diff(acc_mag)

    fft_vals = np.abs(np.fft.rfft(acc_mag))
    dom_freq = float(np.argmax(fft_vals)) if len(fft_vals) else 0.0

    feats += [
        acc_mag.mean(),
        acc_mag.std(),
        ptp(acc_mag),
        acc_jerk.mean() if len(acc_jerk) else 0.0,
        acc_jerk.std() if len(acc_jerk) else 0.0,
        np.sqrt(np.mean(acc_mag**2)),
        acc_mag.std(),
        ptp(acc_mag),
        dom_freq,
        float(np.sum(acc_mag**2)),
        float(len(window))
    ]

    # IMPORTANT: append dummy label to reach 39
    feats.append(0.0)

    X = np.array(feats, dtype=np.float32).reshape(1, -1)
    return X

# =========================================================
# SEND VIBRATION
# =========================================================
def send_vibration(ser, node, manual=False):
    now = time.time()
    cooldown = 1.0 if manual else EXERCISE_CONFIG[CURRENT_EXERCISE]["cooldown"]

    last = last_manual_vibe if manual else last_auto_vibe
    if now - last[node] < cooldown:
        return

    ser.write(f"V{node}\n".encode())
    last[node] = now

    imu_status["logs"].append({
        "time": now,
        "type": "MANUAL" if manual else "AUTO",
        "node": node,
        "message": "Vibration sent"
    })

# =========================================================
# KEYBOARD HANDLER
# =========================================================
def on_key(key, ser):
    try:
        c = key.char
    except:
        return

    if c in "1234":
        send_vibration(ser, int(c), manual=True)
        print(f"[MANUAL] V{c}")

    if c.lower() == "q":
        os._exit(0)

# =========================================================
# MAIN LOOP
# =========================================================
def main():
    port = autodetect_serial()
    if not port:
        print("No ESP32 serial found")
        return

    ser = serial.Serial(port, BAUD, timeout=READ_TIMEOUT)
    load_models()

    Thread(
        target=lambda: keyboard.Listener(
            on_press=lambda k: on_key(k, ser)
        ).run(),
        daemon=True
    ).start()

    print("\n=== ESP-NOW CONTROLLER RUNNING ===")
    print(f"Exercise: {CURRENT_EXERCISE}")
    print("Press 1–4 for manual vibration, Q to quit\n")

    while True:
        raw = ser.readline()
        node, sample = parse_serial_line(raw)
        if node is None:
            continue

        last_seen[node] = time.time()
        buffers[node].append(sample)

        print(f"[IMU{node}] ax={sample['ax']:.2f} ay={sample['ay']:.2f} az={sample['az']:.2f}")

        if len(buffers[node]) < WINDOW_SIZE:
            continue

        X = build_features(buffers[node])
        Xs = corr_scaler.transform(X)

        probs = corr_model.predict_proba(Xs)[0]
        conf = float(max(probs))
        incorrect = int(corr_model.predict(Xs)[0]) == 0

        gyro_mag = np.mean([
            np.sqrt(s["gx"]**2 + s["gy"]**2 + s["gz"]**2)
            for s in buffers[node]
        ])

        cfg = EXERCISE_CONFIG[CURRENT_EXERCISE]
        if incorrect and conf >= cfg["confidence"] and gyro_mag >= cfg["movement"]:
            send_vibration(ser, node, manual=False)
            print(f"❌ AUTO V{node} conf={conf:.2f} gyro={gyro_mag:.2f}")

        imu_status["nodes"][node] = dict(
            last_seen=last_seen[node],
            gyro_mean=gyro_mag,
            confidence=conf,
            is_incorrect=incorrect
        )

        with open(STATUS_FILE, "w") as f:
            json.dump(imu_status, f, indent=2)

# =========================================================
if __name__ == "__main__":
    main()
