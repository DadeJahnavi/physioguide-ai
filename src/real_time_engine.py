# src/real_time_engine.py
"""
Real-Time Inference Engine (simulation mode)
- Streams IMU data (simulation) from data/pt_exercises files
- Buffers windows per node (WINDOW samples)
- Extracts same features as used during training
- Loads saved scalers/models and predicts:
    - imu_exercise_model (exercise_label)
    - imu_correctness_model (correctness_label)
    - pose_model (from MediaPipe landmarks)
- Fuses predictions, displays OpenCV overlay, and prints/simulates haptic commands

Configure:
 - SIMULATE = True (set False to use real serial input)
 - PORT = serial COM port (when hardware connected)
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
import time
import glob
import random
import os
import math
from pathlib import Path
from collections import deque
from scipy import signal
from scipy.spatial.transform import Rotation as R

# --------------------------
# CONFIG
# --------------------------
SIMULATE = True
ROOT_PT = Path("data/pt_exercises")  # folder you copied earlier
WINDOW = 50
STEP = 25
N_NODES = 4  # number of IMU nodes to simulate
FPS = 20     # inference frequency target (Hz)
SERIAL_PORT = "COM5"   # when hardware present
SERIAL_BAUD = 115200

# Target angles per exercise (example; adjust for your exercises)
EXERCISE_TARGETS = {
    # e.g. 'e1': 90.0,
}

# Node mapping (for haptic mapping)
NODE_IDS = [1, 2, 3, 4]  # map simulated node index -> hub node id

# --------------------------
# Utility & feature functions (same logic as parser)
# --------------------------
def safe_read_imu(file_path):
    try:
        df = pd.read_csv(file_path, sep=';', engine='python')
        return df
    except Exception as e:
        print(f"[safe_read_imu] Failed to read {file_path}: {e}")
        return None

def find_col_indices(cols):
    lower = [c.lower().strip() for c in cols]
    def idx_of(prefixes):
        for p in prefixes:
            for i,c in enumerate(lower):
                if c.startswith(p):
                    return i
        return None
    ax_i = idx_of(["acc_x","accx","acc x","accel_x","acc_x;"])
    ay_i = idx_of(["acc_y","accy","acc y","accel_y"])
    az_i = idx_of(["acc_z","accz","acc z","accel_z"])
    gx_i = idx_of(["gyr_x","gyrx","gyr x","gyro_x","gyr_x;"])
    gy_i = idx_of(["gyr_y","gyry","gyr y","gyro_y"])
    gz_i = idx_of(["gyr_z","gyrz","gyr z","gyro_z"])
    return ax_i,ay_i,az_i,gx_i,gy_i,gz_i

def window_features_from_arrays(ax,ay,az,gx=None,gy=None,gz=None):
    feats = {}
    arrs = [("ax",ax),("ay",ay),("az",az)]
    for name, arr in arrs:
        feats[f"{name}_mean"] = float(np.nanmean(arr))
        feats[f"{name}_std"] = float(np.nanstd(arr))
        feats[f"{name}_min"] = float(np.nanmin(arr))
        feats[f"{name}_max"] = float(np.nanmax(arr))
        feats[f"{name}_median"] = float(np.nanmedian(arr))
        feats[f"{name}_rms"] = float(np.sqrt(np.nanmean(np.square(arr))))
        feats[f"{name}_ptp"] = float(np.nanmax(arr) - np.nanmin(arr))
        # shape features
        feats[f"{name}_skew"] = float(pd.Series(arr).skew())
        feats[f"{name}_kurtosis"] = float(pd.Series(arr).kurt())

    # magnitude features
    acc_mag = np.sqrt(ax**2 + ay**2 + az**2)
    feats["acc_mag_mean"] = float(np.nanmean(acc_mag))
    feats["acc_mag_std"] = float(np.nanstd(acc_mag))
    feats["acc_mag_ptp"] = float(np.nanmax(acc_mag) - np.nanmin(acc_mag))
    jerk = np.diff(acc_mag)
    feats["acc_jerk_mean"] = float(np.nanmean(jerk)) if len(jerk)>0 else 0.0
    feats["acc_jerk_std"] = float(np.nanstd(jerk)) if len(jerk)>0 else 0.0

    if gx is not None and gy is not None and gz is not None:
        gm = np.sqrt(gx**2 + gy**2 + gz**2)
        feats["gyr_mag_mean"] = float(np.nanmean(gm))
        feats["gyr_mag_std"] = float(np.nanstd(gm))
        feats["gyr_mag_ptp"] = float(np.nanmax(gm) - np.nanmin(gm))
    else:
        feats["gyr_mag_mean"] = 0.0
        feats["gyr_mag_std"] = 0.0
        feats["gyr_mag_ptp"] = 0.0

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
    feats["nan_frac"] = float(np.isnan(np.concatenate([ax,ay,az])).mean())
    return feats

# Small helper for angle calculation from 3 points (MediaPipe landmarks)
def angle_between_points(A, B, C):
    BA = A - B
    BC = C - B
    nBA = BA / (np.linalg.norm(BA) + 1e-9)
    nBC = BC / (np.linalg.norm(BC) + 1e-9)
    cosang = np.clip(np.dot(nBA, nBC), -1.0, 1.0)
    angle_rad = math.acos(cosang)
    return math.degrees(angle_rad)

# --------------------------
# load models & scalers (fail gently)
# --------------------------
def load_or_die(path):
    if not Path(path).exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return joblib.load(path)

print("Loading models & scalers...")
pose_model = load_or_die("pose_model.pkl")
pose_le = load_or_die("pose_label_encoder.pkl")
imu_ex_model = load_or_die("imu_exercise_model.pkl")
imu_ex_scaler = load_or_die("imu_exercise_scaler.pkl")
imu_cor_model = load_or_die("imu_correctness_model.pkl")
imu_cor_scaler = load_or_die("imu_correctness_scaler.pkl")
print("Models loaded.")

# --------------------------
# Mediapipe setup
# --------------------------
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# --------------------------
# IMU simulation source: choose a set of files to replay
# We'll pick some files from ROOT_PT to feed each simulated node
# --------------------------
def collect_sample_files(root, max_subjects=5):
    files = []
    for s_path in sorted(root.glob("s*")):
        for e_path in sorted(s_path.glob("e*")):
            for u_path in sorted(e_path.glob("u*")):
                # prefer template_session or test
                for fname in ("template_session", "test"):
                    f = u_path / fname
                    if f.exists():
                        files.append(f)
                        break
                else:
                    candidate = list(u_path.glob("*"))
                    if candidate:
                        files.append(candidate[0])
    return files

sim_files = collect_sample_files(ROOT_PT)
if len(sim_files) == 0:
    print("[WARN] No pt_exercises files found in", ROOT_PT)
else:
    random.shuffle(sim_files)
    print(f"Found {len(sim_files)} IMU session files for simulation.")

# --------------------------
# Simple simulator: make N_NODES deques of rows (each row is a dict)
# We'll stream rows (packets) at roughly real-time speed using time index.
# --------------------------
class IMUSimulatorNode:
    def __init__(self, file_path):
        self.file = file_path
        self.df = safe_read_imu(file_path)
        self.idx = 0
        self.n = len(self.df) if self.df is not None else 0

    def next_packet(self):
        if self.df is None or self.idx >= self.n:
            # loop
            self.idx = 0
            if self.df is None:
                return None
        row = self.df.iloc[self.idx]
        self.idx += 1
        return row

sim_nodes = []
if SIMULATE and len(sim_files) > 0:
    # assign each sim node a different file (or rotate)
    for i in range(N_NODES):
        f = sim_files[i % len(sim_files)]
        sim_nodes.append(IMUSimulatorNode(f))
    print(f"Initialized {len(sim_nodes)} simulated IMU nodes.")

# --------------------------
# Buffers for each node: store last WINDOW samples of raw columns
# --------------------------
node_buffers = []
for _ in range(N_NODES):
    node_buffers.append(deque(maxlen=WINDOW))

# --------------------------
# helper: convert mediapipe landmarks to pose feature vector
# We assume training used 33 landmarks and x,y,z ordering → 99 features
# If your pose model used a different order, update this function accordingly.
# --------------------------
def mp_landmarks_to_feature_vector(landmarks, frame_w, frame_h):
    # landmarks: list of mp_landmark, normalized x/y, z roughly in meters
    # produce 99-d vector: x1,y1,z1, x2,y2,z2, ...
    feat = []
    for lm in landmarks:
        # convert normalized xy to pixel coords to match training if needed
        x = lm.x * frame_w
        y = lm.y * frame_h
        z = lm.z * 1000.0  # scaled z — training may expect relative magnitude; adjust if needed
        feat.extend([float(x), float(y), float(z)])
    # if less landmarks, pad zeros
    if len(feat) < 99:
        feat += [0.0] * (99 - len(feat))
    return np.array(feat[:99], dtype=float)

# --------------------------
# Visual & haptic helpers
# --------------------------
def vib_intensity_from_error(error_deg, max_intensity=255, cap_deg=45):
    val = min(abs(error_deg) / cap_deg, 1.0)
    return int(val * max_intensity)

def send_vib_command(node_id, intensity, duration_ms):
    # Simulation: print to console. When hardware is connected, send over serial.
    cmd = f"VIB {node_id} {intensity} {duration_ms}"
    print("[HAPTIC CMD]", cmd)
    # TODO: if SERIAL mode, write to serial port here.

# --------------------------
# Main loop: capture webcam + simulate imu + predict
# --------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Webcam not available. Connect a webcam and retry.")

mp_pose_processor = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

last_time = time.time()
tick_interval = 1.0 / FPS

# simple EMA smoothing for displayed angle
prev_angles = {}

try:
    while True:
        t0 = time.time()

        # 1) Read a webcam frame and get mediapipe keypoints
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam.")
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_pose_processor.process(frame_rgb)

        pose_feat_vector = None
        pose_pred = None
        pose_label = None

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            # prepare feature vector and predict pose class
            frame_h, frame_w = frame.shape[:2]
            lm_list = results.pose_landmarks.landmark
            pose_feat_vector = mp_landmarks_to_feature_vector(lm_list, frame_w, frame_h).reshape(1, -1)
            # NOTE: the pose scaler is not used because earlier we didn't save one.
            # If your pose model expects scaled input, load scaler and transform here.
            try:
                pose_pred_idx = pose_model.predict(pose_feat_vector)[0]
                pose_label = pose_le.inverse_transform([pose_pred_idx])[0]
            except Exception as e:
                pose_label = f"pose_pred_error:{e}"

        # 2) Simulate receiving IMU packets (or read from hardware)
        # We read a packet from each sim node and append to per-node buffer
        if SIMULATE:
            for i, node in enumerate(sim_nodes):
                r = node.next_packet()
                if r is None:
                    continue
                # columns might vary; we try to grab acc_x,acc_y,acc_z,gyr_x,gyr_y,gyr_z
                cols = list(node.df.columns)
                ax_i,ay_i,az_i,gx_i,gy_i,gz_i = find_col_indices(cols)
                if ax_i is None:
                    continue
                # append numeric triple to buffer
                ax = float(r.iloc[ax_i])
                ay = float(r.iloc[ay_i])
                az = float(r.iloc[az_i])
                gx = float(r.iloc[gx_i]) if gx_i is not None else 0.0
                gy = float(r.iloc[gy_i]) if gy_i is not None else 0.0
                gz = float(r.iloc[gz_i]) if gz_i is not None else 0.0
                node_buffers[i].append((ax,ay,az,gx,gy,gz))
        else:
            # Placeholder for real serial/ESP-NOW input logic
            # read serial -> parse -> append to appropriate node_buffers
            pass

        # 3) When buffer full, compute window features and predict for that node
        imu_ex_predictions = []
        imu_cor_predictions = []
        for i in range(N_NODES):
            buf = node_buffers[i]
            if len(buf) >= WINDOW:
                arr = np.array(buf)  # shape (WINDOW, 6)
                ax = arr[:,0]; ay = arr[:,1]; az = arr[:,2]
                gx = arr[:,3]; gy = arr[:,4]; gz = arr[:,5]
                feats = window_features_from_arrays(ax,ay,az,gx,gy,gz)
                # prepare feature vector in same order as training (parser produced these columns)
                feat_df = pd.DataFrame([feats])
                # ensure all columns expected by scaler exist (scaler was fit on columns when training)
                # we assume scaler expects the same numeric columns found in feat_df
                X_vec = feat_df.values
                try:
                    Xs = imu_ex_scaler.transform(X_vec)
                    ex_pred = imu_ex_model.predict(Xs)[0]
                    cor_pred = imu_cor_model.predict(imu_cor_scaler.transform(X_vec))[0]
                except Exception as e:
                    # fallback: try without scaler if mismatch
                    try:
                        ex_pred = imu_ex_model.predict(X_vec)[0]
                        cor_pred = imu_cor_model.predict(X_vec)[0]
                    except Exception as e2:
                        ex_pred = None
                        cor_pred = None
                        print("[WARN] prediction failed:", e, e2)

                imu_ex_predictions.append((i, ex_pred))
                imu_cor_predictions.append((i, cor_pred))

        # Simple fusion: use majority vote across nodes for exercise,
        # use any-incorrect logic for correctness (if any node says incorrect -> incorrect)
        final_ex = None
        final_cor = None
        if imu_ex_predictions:
            vals = [p for (_,p) in imu_ex_predictions if p is not None]
            if vals:
                final_ex_idx = max(set(vals), key=vals.count)
                # decode to label if possible
                try:
                    final_ex = pose_le.inverse_transform([final_ex_idx])[0]
                except Exception:
                    final_ex = f"ex_{final_ex_idx}"
        if imu_cor_predictions:
            vals = [p for (_,p) in imu_cor_predictions if p is not None]
            if vals:
                # correctness_label mapping: 0=incorrect,1=correct (depends on parser)
                final_cor_idx = 1 if sum(vals) >= 0 and np.mean(vals) >= 0.5 else int(round(np.mean(vals)))
                final_cor = "correct" if final_cor_idx == 1 else "incorrect"

        # Display predictions on frame
        label_lines = []
        if pose_label:
            label_lines.append(f"Pose (vision): {pose_label}")
        if final_ex is not None:
            label_lines.append(f"IMU exercise (vote): {final_ex}")
        if final_cor is not None:
            label_lines.append(f"IMU correctness: {final_cor}")

        y0 = 30
        for ln in label_lines:
            cv2.putText(frame, ln, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            y0 += 28

        # Visual guidance: compute a sample joint angle (e.g., right shoulder: landmarks 12-14)
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            # landmark indices: use Mediapipe indices (12 = right shoulder, 14 = right elbow, 16 = right wrist) 
            try:
                A = np.array([lm[12].x*frame.shape[1], lm[12].y*frame.shape[0], lm[12].z])
                B = np.array([lm[14].x*frame.shape[1], lm[14].y*frame.shape[0], lm[14].z])
                C = np.array([lm[16].x*frame.shape[1], lm[16].y*frame.shape[0], lm[16].z])
                sh_angle = angle_between_points(A, B, C)
                # use a target if available
                target = EXERCISE_TARGETS.get("e1", 90.0)
                err = sh_angle - target
                # draw feedback circle on elbow pixel
                ex = int(B[0]); ey = int(B[1])
                abs_err = abs(err)
                if abs_err <= 5:
                    color = (0,255,0)
                elif abs_err <= 15:
                    color = (0,255,255)
                else:
                    color = (0,0,255)
                cv2.circle(frame, (ex,ey), 16, color, -1)
                # arrow text
                if err > 2:
                    cv2.putText(frame, "↑ too low", (ex+20, ey), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                elif err < -2:
                    cv2.putText(frame, "↓ too high", (ex+20, ey), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(frame, f"Err: {err:.1f}°", (ex-60,ey-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                # if severe and persists -> vib simulated
                if abs_err > 10:
                    # which node to vibrate? choose node 1 for right arm for demo
                    node_id = NODE_IDS[0]
                    intensity = vib_intensity_from_error(err)
                    # debounce: only vibrate at state transitions or at limited rate (omitted for brevity)
                    send_vib_command(node_id, intensity, 120)
            except Exception as e:
                pass

        # show frame
        cv2.imshow("Rehab Demo (Sim Mode)", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # ESC
            break

        # throttle to desired FPS
        tspent = time.time() - t0
        to_sleep = max(0.001, (1.0 / FPS) - tspent)
        time.sleep(to_sleep)

finally:
    cap.release()
    cv2.destroyAllWindows()
    mp_pose_processor.close()
    print("Engine stopped.")