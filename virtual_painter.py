import cv2
import mediapipe as mp
import numpy as np
import time
import os

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
time.sleep(2)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

canvas = None
prev_x, prev_y = None, None
colors = [(255,0,255), (255,0,0), (0,255,0), (0,0,255)]
color_index = 0
brush_thickness = 8
eraser_thickness = 35

clear_time = None

prev_time = 0

os.makedirs("drawings", exist_ok=True)

def finger_up(lm, tip, pip, h):
    return lm[tip].y * h < lm[pip].y * h

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    if canvas is None:
        canvas = np.zeros_like(frame)

    curr_time = time.time()
    fps = int(1 / (curr_time - prev_time)) if prev_time else 0
    prev_time = curr_time

    header_height = 70
    cv2.rectangle(frame, (0,0), (w, header_height), (40,40,40), -1)

    for i, col in enumerate(colors):
        cv2.rectangle(frame, (10 + i*70, 10), (60 + i*70, 60), col, -1)
        if i == color_index:
            cv2.rectangle(frame, (10 + i*70, 10), (60 + i*70, 60), (255,255,255), 2)

    cv2.putText(frame, "Ring+Pinky: Erase | All Fingers: Clear | S: Save",
                (300, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        lm = result.multi_hand_landmarks[0].landmark

        index = finger_up(lm, 8, 6, h)
        middle = finger_up(lm, 12, 10, h)
        ring = finger_up(lm, 16, 14, h)
        pinky = finger_up(lm, 20, 18, h)

        x = int(lm[8].x * w)
        y = int(lm[8].y * h)

        if y < header_height and index:
            for i in range(len(colors)):
                if 10 + i*70 < x < 60 + i*70:
                    color_index = i
                    prev_x, prev_y = None, None
                    time.sleep(0.25)

        elif ring and pinky and not index and not middle:
            if prev_x is None:
                prev_x, prev_y = x, y
            cv2.line(canvas, (prev_x, prev_y), (x, y), (0,0,0), eraser_thickness)
            prev_x, prev_y = x, y

        elif index and middle and ring and pinky:
            if clear_time is None:
                clear_time = time.time()
            elif time.time() - clear_time > 1:
                canvas = np.zeros_like(frame)
        else:
            clear_time = None

        if index and not middle and y > header_height:
            if prev_x is None:
                prev_x, prev_y = x, y
            cv2.line(canvas, (prev_x, prev_y), (x, y), colors[color_index], brush_thickness)
            prev_x, prev_y = x, y
        else:
            prev_x, prev_y = None, None

        mp_draw.draw_landmarks(frame, result.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

    frame = cv2.add(frame, canvas)

    cv2.putText(frame, f"FPS: {fps}", (w-120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Virtual Painter", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        filename = f"drawings/drawing_{int(time.time())}.png"
        cv2.imwrite(filename, canvas)
        print(f"Saved: {filename}")

hands.close()
cap.release()

cv2.destroyAllWindows()

