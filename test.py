import os
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

mp.hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp.drawing_style = mp.solutions.drawing_styles

hands = mp.hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_))[:1]:
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        results = hands.process(img_rgb)
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img_rgb,hand_landmarks, mp.hands.HAND_CONNECTIONS)
        
        plt.figure()
        plt.imshow(img_rgb)

plt.show()
