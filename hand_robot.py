import cv2
import mediapipe as mp
import pybullet as p
import pybullet_data
import time

# ==================================================
# PYBULLET SETUP
# ==================================================
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

plane = p.loadURDF("plane.urdf")
robot = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)
#robot = p.loadURDF("humanoid/humanoid.urdf", [0,0,1])
endEffectorIndex = 6

# ==================================================
# MEDIAPIPE NEW VERSION SETUP
# ==================================================
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

landmarker = HandLandmarker.create_from_options(options)

# ==================================================
# CAMERA START
# ==================================================
cap = cv2.VideoCapture(0)

# ==================================================
# MAIN LOOP
# ==================================================
while True:

    ret, frame = cap.read()
    if not ret:
        break

    # mirror image
    frame = cv2.flip(frame, 1)

    # resize
    frame = cv2.resize(frame, (640, 480))

    # BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    # timestamp required for VIDEO mode
    timestamp = int(time.time() * 1000)

    # detect hand
    result = landmarker.detect_for_video(mp_image, timestamp)

    # ==================================================
    # IF HAND FOUND
    # ==================================================
    if result.hand_landmarks:

        for hand in result.hand_landmarks:

            # wrist landmark = ID 0
            wrist = hand[0]

            x = int(wrist.x * 640)
            y = int(wrist.y * 480)

            # draw wrist point
            cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)

            # ------------------------------------------
            # Convert camera coordinate -> robot coord
            # ------------------------------------------
            robot_x = (x - 320) / 500
            robot_y = (240 - y) / 500
            robot_z = 0.5

            targetPos = [robot_x, robot_y, robot_z]

            # ------------------------------------------
            # INVERSE KINEMATICS
            # ------------------------------------------
            jointPoses = p.calculateInverseKinematics(
                robot,
                endEffectorIndex,
                targetPos
            )

            # move robot joints
            for i in range(7):
                p.setJointMotorControl2(
                    robot,
                    i,
                    p.POSITION_CONTROL,
                    targetPosition=jointPoses[i]
                )

            # print coordinates
            print("Hand X:", x, "Hand Y:", y)

    # step simulation
    p.stepSimulation()

    # show camera
    cv2.imshow("Hand Tracking", frame)

    # press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==================================================
# CLOSE EVERYTHING
# ==================================================
cap.release()
cv2.destroyAllWindows()
p.disconnect()