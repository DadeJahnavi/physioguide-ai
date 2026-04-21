import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time

# ------------------------------------------
#           PHYSIOGUIDE — SETTINGS
# ------------------------------------------
APP_TITLE = "PhysioGuide — Rehab Demo"
EXERCISES = [
    "shoulder_flexion_up",
    "shoulder_flexion_down",
    "arm_raise_side",
    "glute_bridge_up",
    "glute_bridge_down",
    "knee_raise_up",
    "knee_raise_down"
]

# ------------------------------------------
#           MEDIAPIPE INIT
# ------------------------------------------
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

# ------------------------------------------
#         STATE INIT
# ------------------------------------------
if "demo_running" not in st.session_state:
    st.session_state.demo_running = False

# ------------------------------------------
#        RUN THE LIVE WEBCAM LOOP
# ------------------------------------------
def run_demo(exercise, target_angle, simulate_imu=True):

    pose = mp_pose.Pose(min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not access webcam.")
        return

    video_ph = st.empty()

    col1, col2, col3 = st.columns([1, 1, 1])
    status_ph = col1.empty()
    info_ph = col2.empty()

    st.success("Demo running... Press Stop Demo to exit.")

    while st.session_state.demo_running:

        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(img_rgb)

        if res.pose_landmarks:
            mp_draw.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Smaller, fixed video size for screen fit
        video_ph.image(frame, channels="BGR", width=640)

        # Smaller status card
        form_ok = np.random.choice([True, False]) if simulate_imu else False
        status_color = "#4CAF50" if form_ok else "#FF9800"
        status_text = "✔ Perfect Form" if form_ok else "⚠ Adjust Your Form"

        status_ph.markdown(
            f"""
            <div style="
                width: 100%;
                padding: 12px;
                background-color: {status_color};
                color: white;
                text-align: center;
                border-radius: 10px;
                font-size: 16px;
                font-weight: 600;
            ">
                {status_text}
            </div>
            """,
            unsafe_allow_html=True
        )

        info_ph.markdown(
            f"""
            <div style="
                padding: 14px;
                border-radius: 10px;
                background-color: #f6f8fa;
                box-shadow: 0px 2px 6px rgba(0,0,0,0.1);
                font-size: 14px;
            ">
                <b>Exercise:</b> {exercise}<br>
                <b>Target Angle:</b> {target_angle}°<br>
                <b>IMU Simulation:</b> {simulate_imu}<br>
            </div>
            """,
            unsafe_allow_html=True
        )

        time.sleep(0.03)

    cap.release()
    st.success("Demo stopped.")

# ------------------------------------------
#               STREAMLIT UI
# ------------------------------------------
st.set_page_config(page_title="PhysioGuide", layout="wide")

# Smaller sidebar styling
st.markdown("""
    <style>
        .sidebar .stSelectbox, .sidebar .stButton, .sidebar .stSlider {
            font-size: 14px !important;
        }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.subheader("Session Controls")

    selected_ex = st.selectbox("Choose exercise:", EXERCISES)
    target_angle = st.slider("Target angle (degrees)", 0, 160, 90)
    simulate_imu = st.checkbox("Simulate IMU (no hardware)", True)

    st.write(" ")

    start_clicked = st.button("Start Demo", use_container_width=True, key="start_btn")
    stop_clicked = st.button("Stop Demo", use_container_width=True, key="stop_btn")

    st.session_state.selected_ex = selected_ex
    st.session_state.target_angle = target_angle
    st.session_state.simulate_imu = simulate_imu

    if start_clicked:
        if not st.session_state.demo_running:
            st.session_state.demo_running = True
            st.rerun()

    if stop_clicked:
        if st.session_state.demo_running:
            st.session_state.demo_running = False
            st.rerun()

# Smaller main title
st.markdown(
    """
    <div style='text-align: center; padding: 10px;'>
        <h1 style='font-size: 36px; font-weight: 700; margin-bottom: 4px;'>PhysioGuide</h1>
        <p style='font-size:16px; color:gray; margin-top:0;'>
            Welcome to PhysioGuide — Rehab Demo
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Run demo
if st.session_state.demo_running:
    run_demo(
        st.session_state.selected_ex,
        st.session_state.target_angle,
        st.session_state.simulate_imu
    )
else:
    st.markdown(
        """
        <div style='text-align:center; margin-top:40px;'>
            <p style='font-size:16px; color:gray;'>
                Select an exercise and press <b>Start Demo</b> in the sidebar.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
