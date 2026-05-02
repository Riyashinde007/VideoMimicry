import pybullet as p
import pybullet_data
import time

# Connect GUI
p.connect(p.GUI)

# Load assets
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Gravity
p.setGravity(0, 0, -9.8)

# Load plane
plane = p.loadURDF("plane.urdf")

# Load robot
robot = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)

# End effector link index
endEffectorIndex = 6

# Main loop
while True:

    # Example target position
    targetPos = [0.4, 0.2, 0.5]

    # IK Calculation
    jointPoses = p.calculateInverseKinematics(
        robot,
        endEffectorIndex,
        targetPos
    )

    # Apply joint angles
    for i in range(7):
        p.setJointMotorControl2(
            robot,
            i,
            p.POSITION_CONTROL,
            jointPoses[i]
        )

    p.stepSimulation()
    time.sleep(1/240)

