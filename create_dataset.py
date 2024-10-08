import os
import pickle
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands

# Initialize the MediaPipe Hands model
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(dir_path):
        continue

    for video_path in os.listdir(dir_path):
        video_full_path = os.path.join(dir_path, video_path)
        cap = cv2.VideoCapture(video_full_path)

        if not cap.isOpened():
            print(f"Warning: Unable to open video {video_full_path}. Skipping...")
            continue

        while True:
            ret, frame = cap.read()

            if not ret:
                break  # End of video

            # Convert the frame to RGB
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    data_aux = []
                    x_ = []
                    y_ = []

                    # Extract landmarks
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        x_.append(x)
                        y_.append(y)

                    # Normalize landmarks
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                    data.append(data_aux)
                    labels.append(dir_)

        cap.release()

# Save the extracted data and labels to a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Data processing complete and saved to 'data.pickle'.")
