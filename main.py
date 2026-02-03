import cv2
import mediapipe as mp
import numpy as np
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from math import hypot
import screen_brightness_control as sbc

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Initialize Pycaw for volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None
)
volume = cast(interface, POINTER(IAudioEndpointVolume))
vol_range = volume.GetVolumeRange()
min_vol = vol_range[0]
max_vol = vol_range[1]

# Webcam
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_lms in result.multi_hand_landmarks:
            lm_list = []

            for id, lm in enumerate(hand_lms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((id, cx, cy))

            # Thumb tip (4) & Index finger tip (8)
            x1, y1 = lm_list[4][1], lm_list[4][2]
            x2, y2 = lm_list[8][1], lm_list[8][2]

            cv2.circle(frame, (x1, y1), 8, (255, 0, 0), -1)
            cv2.circle(frame, (x2, y2), 8, (255, 0, 0), -1)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            length = hypot(x2 - x1, y2 - y1)

            # Map distance to volume
            vol = np.interp(length, [30, 200], [min_vol, max_vol])
            volume.SetMasterVolumeLevel(vol, None)

            # Map distance to brightness
            bright = np.interp(length, [30, 200], [0, 100])
            sbc.set_brightness(int(bright))

            # Display values
            cv2.putText(frame, f'Volume', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'Brightness', (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            mp_draw.draw_landmarks(
                frame, hand_lms, mp_hands.HAND_CONNECTIONS
            )

    cv2.imshow("Volume & Brightness Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()