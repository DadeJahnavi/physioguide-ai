# dashboard.py
# FINAL STABLE DASHBOARD — PhysioGuide Rehab System

import streamlit as st
import time
import json
import os
import cv2
import threading
import mediapipe as mp

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    layout="wide",
    page_title="PhysioGuide — Rehab Dashboard",
    initial_sidebar_state="collapsed"
)

# =========================================================
# CONSTANTS
# =========================================================
STATUS_FILE = "imu_status.json"

EXERCISES = [
    "shoulder_flexion_up",
    "shoulder_flexion_down",
    "arm_raise_side",
    "glute_bridge_up",
    "glute_bridge_down",
    "knee_raise_up",
    "knee_raise_down"
]

NODE_TO_PART = {
    1: "Right Arm",
    2: "Left Arm",
    3: "Left Leg",
    4: "Right Leg"
}

# =========================================================
# SAFE STATUS LOADER
# =========================================================
def load_status():
    if not os.path.exists(STATUS_FILE):
        return {"nodes": {}, "logs": [], "current_exercise": {}}
    try:
        with open(STATUS_FILE, "r") as f:
            return json.load(f)
    except:
        return {"nodes": {}, "logs": [], "current_exercise": {}}

# =========================================================
# CAMERA THREAD (SINGLE INSTANCE)
# =========================================================
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

@st.cache_resource
class CameraWorker:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.pose = mp_pose.Pose()
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        while self.running:
            ok, img = self.cap.read()
            if not ok:
                time.sleep(0.2)
                continue

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = self.pose.process(rgb)
            out = img.copy()

            if res.pose_landmarks:
                mp_draw.draw_landmarks(
                    out,
                    res.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

            with self.lock:
                self.frame = out

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

camera = CameraWorker()

# =========================================================
# UI HEADER
# =========================================================
st.title("🏥 PhysioGuide — Rehabilitation Dashboard")

status = load_status()
nodes = status.get("nodes", {})
logs = status.get("logs", [])
current_ex = status.get("current_exercise", {}).get("name")

# =========================================================
# EXERCISE SELECTION (SAFE)
# =========================================================
if current_ex not in EXERCISES:
    current_ex = EXERCISES[0]

selected_ex = st.selectbox(
    "🎯 Select Exercise",
    EXERCISES,
    index=EXERCISES.index(current_ex),
    key="exercise_select"
)

# =========================================================
# LAYOUT
# =========================================================
col_cam, col_status = st.columns([3, 2])

# ================= CAMERA =================
with col_cam:
    st.subheader("🎥 Live Posture Feed")
    frame = camera.get_frame()
    if frame is not None:
        st.image(frame[:, :, ::-1], channels="RGB", use_container_width=True)
    else:
        st.info("Waiting for camera...")

    st.markdown("### 🔔 Manual Vibration Control")
    btn_cols = st.columns(4)
    for i in range(4):
        with btn_cols[i]:
            st.button(
                f"Vibrate Node {i+1}",
                key=f"manual_vibe_{i+1}"
            )

# ================= STATUS =================
with col_status:
    st.subheader("📡 Node Status")

    if not nodes:
        st.warning("No IMU nodes detected yet.")
    else:
        for nid in range(1, 5):
            node = nodes.get(str(nid))
            if not node:
                st.info(f"⚪ Node {nid} — {NODE_TO_PART[nid]} (Not connected)")
                continue

            is_bad = node.get("is_incorrect", False)
            conf = node.get("confidence", 0.0)
            gyro = node.get("gyro_mean", 0.0)
            last_seen = node.get("last_seen", 0)

            active = "🟢" if time.time() - last_seen < 2 else "⚪"
            icon = "❌" if is_bad else "✅"

            st.markdown(
                f"""
                **{active} Node {nid} — {NODE_TO_PART[nid]}**  
                Status: {icon}  
                Confidence: `{conf:.2f}`  
                Gyro mean: `{gyro:.2f}`
                """
            )

    st.markdown("---")
    st.subheader("📥 Logs")

    log_text = json.dumps(logs[-300:], indent=2)

    st.download_button(
        label="Download Logs (JSON)",
        data=log_text,
        file_name="session_logs.json",
        mime="application/json",
        key="download_logs_btn"
    )

    if logs:
        st.markdown("### 🧾 Recent Events")
        for e in logs[-10:][::-1]:
            ts = time.strftime("%H:%M:%S", time.localtime(e["time"]))
            st.markdown(
                f"- `{ts}` **{e['type']}** Node {e['node']}: {e['message']}"
            )

# =========================================================
# AUTO REFRESH (SAFE)
# =========================================================
time.sleep(0.2)
st.rerun()
