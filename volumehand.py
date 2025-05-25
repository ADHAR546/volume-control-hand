import cv2
import mediapipe as mp
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import math

# Initialize MediaPipe and OpenCV
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Pycaw initialization for volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Get volume range
vol_min, vol_max = volume.GetVolumeRange()[:2]

# OpenCV Video Capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    image = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Hand detection
    results = hands.process(img_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the coordinates of the thumb and index finger
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            h, w, _ = image.shape
            thumb_pos = (int(thumb_tip.x * w), int(thumb_tip.y * h))
            index_pos = (int(index_tip.x * w), int(index_tip.y * h))

            # Draw circles on fingertips
            cv2.circle(image, thumb_pos, 5, (0, 255, 0), cv2.FILLED)
            cv2.circle(image, index_pos, 5, (0, 255, 0), cv2.FILLED)

            # Draw line between thumb and index
            cv2.line(image, thumb_pos, index_pos, (255, 0, 0), 2)

            # Calculate distance
            length = math.hypot(index_pos[0] - thumb_pos[0], index_pos[1] - thumb_pos[1])

            # Map the length to volume range
            vol = np.interp(length, [20, 200], [vol_min, vol_max])
            volume.SetMasterVolumeLevel(vol, None)

            # Display volume level
            cv2.putText(image, f'Volume control by tyagi ji: {int(np.interp(length, [20, 200], [0, 100]))}%',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the frame
    cv2.imshow('Volume Control by Tyagi ji ', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
