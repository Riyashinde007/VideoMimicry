# # import cv2
# # import mediapipe as mp

# # # start camera
# # cap = cv2.VideoCapture(0)

# # # initialize mediapipe
# # mp_hands = mp.solutions.hands
# # mp_draw = mp.solutions.drawing_utils
# # hands = mp_hands.Hands()
# import cv2
# import mediapipe as mp

# print(mp.__file__)   # debug line

# mp_hands = mp.solutions.hands
# mp_draw = mp.solutions.drawing_utils
# hands = mp_hands.Hands()

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Step 2.1 Flip (mirror)
#     frame = cv2.flip(frame, 1)

#     # Step 2.2 Resize
#     frame = cv2.resize(frame, (640, 480))

#     # Step 2.3 Convert BGR to RGB
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # Step 3 Hand detection
#     results = hands.process(rgb_frame)

#     # Draw landmarks
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             mp_draw.draw_landmarks(
#                 frame,
#                 hand_landmarks,
#                 mp_hands.HAND_CONNECTIONS
#             )

#     cv2.imshow("Video Feed", frame)

#     # press q to exit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

# import cv2
# import mediapipe as mp # type: ignore

# # Initialize mediapipe
# mp_hands = mp.solutions.hands
# mp_draw = mp.solutions.drawing_utils
# hands = mp_hands.Hands()

# # Start camera  ← ADD HERE
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Flip
#     frame = cv2.flip(frame, 1)

#     # Resize
#     frame = cv2.resize(frame, (640, 480))

#     # Convert to RGB
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # Hand detection
#     results = hands.process(rgb_frame)

#     # Draw landmarks
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             mp_draw.draw_landmarks(
#                 frame,
#                 hand_landmarks,
#                 mp_hands.HAND_CONNECTIONS
#             )

#     cv2.imshow("Video Feed", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Use the new MediaPipe API
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Download model from:
# https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

cap = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (640, 480))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect hands
        import time

        timestamp = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp)
        # Draw landmarks manually
        # Draw landmarks + extract coordinates
        if result.hand_landmarks:
            for hand in result.hand_landmarks:

                for id, landmark in enumerate(hand):
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])

            # draw point
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            # Step 4: print coordinates
                    print("ID:", id, "X:", x, "Y:", y)

            cv2.imshow("Video Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()


