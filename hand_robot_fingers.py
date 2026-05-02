"""
hand_robot_fingers.py
─────────────────────
Full hand mimicry: wrist + palm + all 5 fingers.

Requirements:
    pip install opencv-python mediapipe pybullet numpy

Model used:
    Shadow Hand URDF bundled with PyBullet:
        pybullet_data / shadow_hand / shadow_hand.urdf
    Falls back to Allegro Hand if Shadow Hand is absent.

MediaPipe landmarks (21 points):
    0  = WRIST
    1-4  = THUMB  (CMC, MCP, IP, TIP)
    5-8  = INDEX  (MCP, PIP, DIP, TIP)
    9-12 = MIDDLE (MCP, PIP, DIP, TIP)
    13-16= RING   (MCP, PIP, DIP, TIP)
    17-20= PINKY  (MCP, PIP, DIP, TIP)
"""

import cv2
import mediapipe as mp
import pybullet as p
import pybullet_data
import numpy as np
import time
import os

# ──────────────────────────────────────────────
# 1.  PYBULLET SETUP
# ──────────────────────────────────────────────
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)
p.resetDebugVisualizerCamera(
    cameraDistance=0.5,
    cameraYaw=45,
    cameraPitch=-30,
    cameraTargetPosition=[0, 0, 0.2]
)

p.loadURDF("plane.urdf")

# ── Choose robot hand model ──────────────────
DATA_PATH = pybullet_data.getDataPath()
SHADOW_PATH   = os.path.join(DATA_PATH, "shadow_hand", "shadow_hand.urdf")
ALLEGRO_PATH  = os.path.join(DATA_PATH, "allegro_hand_description",
                              "allegro_hand_right.urdf")

if os.path.exists(SHADOW_PATH):
    robot = p.loadURDF(SHADOW_PATH, basePosition=[0, 0, 0], useFixedBase=True)
    HAND_MODEL = "shadow"
    print("[INFO] Loaded Shadow Hand")
elif os.path.exists(ALLEGRO_PATH):
    robot = p.loadURDF(ALLEGRO_PATH, basePosition=[0, 0, 0], useFixedBase=True)
    HAND_MODEL = "allegro"
    print("[INFO] Loaded Allegro Hand")
else:
    # Fallback: Kuka arm + gripper so the script still runs
    robot = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)
    HAND_MODEL = "kuka"
    print("[WARN] No dexterous hand URDF found – using Kuka arm as fallback.")
    print("[WARN] Install a Shadow Hand / Allegro Hand URDF for finger control.")

# Collect all controllable (revolute) joints
num_joints = p.getNumJoints(robot)
revolute_joints = []
for j in range(num_joints):
    info = p.getJointInfo(robot, j)
    if info[2] == p.JOINT_REVOLUTE:         # type = REVOLUTE
        revolute_joints.append(j)

print(f"[INFO] {HAND_MODEL} model – {len(revolute_joints)} revolute joints")


# ──────────────────────────────────────────────
# 2.  MEDIAPIPE HAND LANDMARKER SETUP
# ──────────────────────────────────────────────
BaseOptions           = mp.tasks.BaseOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

# Place hand_landmarker.task in the same directory as this script.
TASK_FILE = "hand_landmarker.task"
if not os.path.exists(TASK_FILE):
    raise FileNotFoundError(
        f"Model file '{TASK_FILE}' not found.\n"
        "Download from: https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    )

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=TASK_FILE),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
)
landmarker = HandLandmarker.create_from_options(options)


# ──────────────────────────────────────────────
# 3.  FINGER ANGLE HELPERS
# ──────────────────────────────────────────────

def vec(a, b):
    """3-D vector from landmark a to b."""
    return np.array([b.x - a.x, b.y - a.y, b.z - a.z], dtype=float)


def angle_between(v1, v2):
    """Angle (radians) between two 3-D vectors."""
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos_val = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.arccos(cos_val))


# MediaPipe landmark indices per finger
# Each tuple: (MCP, PIP, DIP, TIP)
FINGER_LANDMARKS = {
    "thumb":  (1,  2,  3,  4),
    "index":  (5,  6,  7,  8),
    "middle": (9,  10, 11, 12),
    "ring":   (13, 14, 15, 16),
    "pinky":  (17, 18, 19, 20),
}

# How many revolute joints each finger occupies in the robot model.
# Shadow Hand layout (adjust if you use a different model):
SHADOW_FINGER_JOINTS = {
    "thumb":  4,   # J1-J4
    "index":  4,
    "middle": 4,
    "ring":   4,
    "pinky":  5,   # pinky has one extra abduction joint
}
ALLEGRO_FINGER_JOINTS = {
    "thumb":  4,
    "index":  4,
    "middle": 4,
    "ring":   4,
    "pinky":  0,   # Allegro has 4 fingers (no separate pinky)
}
KUKA_JOINT_COUNT = 7  # just the arm joints for fallback


def compute_finger_angles(lm, finger_name):
    """
    Returns a list of joint angles (rad) for one finger.
    We compute:
        joint[0] = MCP flex  (knuckle)
        joint[1] = PIP flex  (middle joint)
        joint[2] = DIP flex  (tip joint)
    Thumb gets an extra abduction angle at index 0.
    """
    ids = FINGER_LANDMARKS[finger_name]
    wrist = lm[0]

    mcp = lm[ids[0]]
    pip = lm[ids[1]]
    dip = lm[ids[2]]
    tip = lm[ids[3]]

    # Flexion angles (bend)
    mcp_flex = angle_between(vec(wrist, mcp), vec(mcp, pip))
    pip_flex = angle_between(vec(mcp,  pip),  vec(pip, dip))
    dip_flex = angle_between(vec(pip,  dip),  vec(dip, tip))

    if finger_name == "thumb":
        # Thumb abduction: lateral spread from palm
        index_mcp = lm[5]
        abduction = angle_between(vec(wrist, mcp), vec(wrist, index_mcp))
        return [abduction, mcp_flex, pip_flex, dip_flex]

    return [mcp_flex, pip_flex, dip_flex]


def wrist_pose_from_landmarks(lm):
    """
    Derive a 3-D target position + orientation for the wrist.
    Uses landmarks 0 (wrist), 5 (index MCP), 17 (pinky MCP).
    Returns (position [3], quaternion [4]).
    """
    w  = np.array([lm[0].x,  lm[0].y,  lm[0].z])
    im = np.array([lm[5].x,  lm[5].y,  lm[5].z])
    pm = np.array([lm[17].x, lm[17].y, lm[17].z])

    # Palm normal (z-axis of hand frame)
    palm_right = im - pm
    palm_up    = im - w
    palm_norm  = np.cross(palm_right, palm_up)

    def safe_norm(v):
        n = np.linalg.norm(v)
        return v / n if n > 1e-6 else v

    z_axis = safe_norm(palm_norm)
    y_axis = safe_norm(palm_up)
    x_axis = np.cross(y_axis, z_axis)

    # Build rotation matrix → quaternion
    R = np.column_stack([x_axis, y_axis, z_axis])

    # Convert 3x3 rotation matrix to quaternion (w, x, y, z) then to (x, y, z, w)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    # Scale to robot workspace
    pos = np.array([
         (w[0] - 0.5) * 0.4,   # x: -0.2 .. 0.2
         (0.5 - w[1]) * 0.4,   # y: flip because image Y is down
         0.15 + w[2] * 0.3     # z: keep hand elevated
    ])
    quat = [qx, qy, qz, qw]    # PyBullet format
    return pos.tolist(), quat


# ──────────────────────────────────────────────
# 4.  ROBOT CONTROL  (model-aware)
# ──────────────────────────────────────────────

def apply_finger_angles_to_robot(finger_angles_dict):
    """
    Distribute per-finger angles to the robot's revolute joints.
    Joint assignment depends on which model is loaded.
    """
    if HAND_MODEL == "shadow":
        joint_cursor = 0
        for finger, n_joints in SHADOW_FINGER_JOINTS.items():
            angles = finger_angles_dict.get(finger, [0.0] * n_joints)
            for k in range(n_joints):
                if joint_cursor >= len(revolute_joints):
                    break
                target = angles[k] if k < len(angles) else 0.0
                p.setJointMotorControl2(
                    robot,
                    revolute_joints[joint_cursor],
                    p.POSITION_CONTROL,
                    targetPosition=target,
                    force=1.0,
                    maxVelocity=2.0,
                )
                joint_cursor += 1

    elif HAND_MODEL == "allegro":
        joint_cursor = 0
        for finger, n_joints in ALLEGRO_FINGER_JOINTS.items():
            if n_joints == 0:
                continue
            angles = finger_angles_dict.get(finger, [0.0] * n_joints)
            for k in range(n_joints):
                if joint_cursor >= len(revolute_joints):
                    break
                target = angles[k] if k < len(angles) else 0.0
                p.setJointMotorControl2(
                    robot,
                    revolute_joints[joint_cursor],
                    p.POSITION_CONTROL,
                    targetPosition=target,
                    force=1.0,
                    maxVelocity=2.0,
                )
                joint_cursor += 1

    else:  # kuka fallback – move arm with wrist position only
        pass


def apply_wrist_pose(pos, quat):
    """
    For dexterous hand models: set the base position/orientation.
    For Kuka: use IK to track the wrist.
    """
    if HAND_MODEL in ("shadow", "allegro"):
        p.resetBasePositionAndOrientation(robot, pos, quat)
    else:
        # Kuka IK fallback
        end_effector = 6
        joint_poses = p.calculateInverseKinematics(robot, end_effector, pos, quat)
        for i in range(min(7, len(joint_poses))):
            p.setJointMotorControl2(
                robot, i, p.POSITION_CONTROL,
                targetPosition=joint_poses[i],
                force=500, maxVelocity=2.0
            )


# ──────────────────────────────────────────────
# 5.  SMOOTHING BUFFER
# ──────────────────────────────────────────────
ALPHA = 0.25   # EMA smoothing factor  (higher = more responsive, more jitter)

smooth_state = {}   # will be filled on first detection


def ema(key, new_val, shape=None):
    """Exponential moving average per named key."""
    arr = np.array(new_val, dtype=float)
    if key not in smooth_state:
        smooth_state[key] = arr.copy()
    smooth_state[key] = ALPHA * arr + (1 - ALPHA) * smooth_state[key]
    return smooth_state[key].tolist()


# ──────────────────────────────────────────────
# 6.  DRAW OVERLAY HELPERS
# ──────────────────────────────────────────────
MP_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),           # thumb
    (0,5),(5,6),(6,7),(7,8),           # index
    (0,9),(9,10),(10,11),(11,12),       # middle
    (0,13),(13,14),(14,15),(15,16),    # ring
    (0,17),(17,18),(18,19),(19,20),    # pinky
    (5,9),(9,13),(13,17),              # palm knuckle row
]

FINGER_COLORS = {
    "thumb":  (255, 100, 100),
    "index":  (100, 255, 100),
    "middle": (100, 200, 255),
    "ring":   (255, 180,  50),
    "pinky":  (200, 100, 255),
}
LANDMARK_FINGER = {}
for fn, ids in FINGER_LANDMARKS.items():
    for lid in ids:
        LANDMARK_FINGER[lid] = fn

def draw_hand(frame, lm_list):
    H, W = frame.shape[:2]
    pts = [(int(l.x * W), int(l.y * H)) for l in lm_list]

    # connections
    for a, b in MP_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (200, 200, 200), 2)

    # landmark dots
    for i, (x, y) in enumerate(pts):
        color = FINGER_COLORS.get(LANDMARK_FINGER.get(i, ""), (255, 255, 255))
        cv2.circle(frame, (x, y), 6, color, -1)
        cv2.circle(frame, (x, y), 6, (0, 0, 0), 1)

    # labels for key points
    labels = {0: "WRIST", 4: "THUMB", 8: "INDEX", 12: "MID", 16: "RING", 20: "PINKY"}
    for lid, txt in labels.items():
        cv2.putText(frame, txt, (pts[lid][0]+6, pts[lid][1]-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)


def draw_finger_bars(frame, finger_angles_dict):
    """Mini bar chart of per-finger bend angles (bottom-left corner)."""
    y0, x0, bar_w, bar_h = 400, 10, 18, 80
    for i, (fname, angles) in enumerate(finger_angles_dict.items()):
        avg_angle = float(np.mean(angles)) if angles else 0.0
        fill = int(np.clip(avg_angle / 1.8, 0, 1) * bar_h)
        color = FINGER_COLORS[fname]
        bx = x0 + i * (bar_w + 4)
        cv2.rectangle(frame, (bx, y0 - bar_h), (bx + bar_w, y0), (50, 50, 50), -1)
        cv2.rectangle(frame, (bx, y0 - fill), (bx + bar_w, y0), color, -1)
        cv2.putText(frame, fname[0].upper(), (bx + 4, y0 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


# ──────────────────────────────────────────────
# 7.  CAMERA & MAIN LOOP
# ──────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera.")

print("\n[INFO] Running. Press  Q  to quit.\n")

prev_ts = -1

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (640, 480))
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Strictly increasing timestamps required by VIDEO mode
        ts = int(time.time() * 1000)
        if ts <= prev_ts:
            ts = prev_ts + 1
        prev_ts = ts

        result = landmarker.detect_for_video(mp_image, ts)

        if result.hand_landmarks:
            lm = result.hand_landmarks[0]   # first hand

            # ── Wrist pose ───────────────────────────
            pos, quat = wrist_pose_from_landmarks(lm)
            pos  = ema("wrist_pos",  pos)
            quat = ema("wrist_quat", quat)
            apply_wrist_pose(pos, quat)

            # ── Finger angles ────────────────────────
            finger_angles = {}
            for fname in FINGER_LANDMARKS:
                raw   = compute_finger_angles(lm, fname)
                smooth = ema(f"finger_{fname}", raw)
                finger_angles[fname] = smooth

            apply_finger_angles_to_robot(finger_angles)

            # ── Draw overlay ─────────────────────────
            draw_hand(frame, lm)
            draw_finger_bars(frame, finger_angles)

            # ── Console log (optional – comment out for speed) ──
            # for fname, angles in finger_angles.items():
            #     print(f"{fname:6s}: " +
            #           "  ".join(f"{np.degrees(a):5.1f}°" for a in angles))

        # ── Status text ──────────────────────────────
        status = "HAND DETECTED" if result.hand_landmarks else "No hand"
        color  = (0, 255, 80) if result.hand_landmarks else (0, 80, 255)
        cv2.putText(frame, status, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Model: {HAND_MODEL}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        p.stepSimulation()

        cv2.imshow("Hand Robot Mimic", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    p.disconnect()
    print("[INFO] Closed cleanly.")