import os
import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe hands and drawing utilities
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize the MediaPipe Hands solution
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# Define the color for the peach-black background (RGB format)
peach_black_color = (20, 20, 20)  # Adjust to a darker or lighter shade as needed

DATA_DIR = './data'

for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(dir_path):
        continue 

    for video_path in os.listdir(dir_path):  # Process each video[:1]
        video_full_path = os.path.join(dir_path, video_path)
        cap = cv2.VideoCapture(video_full_path)

        if not cap.isOpened():
            print(f"Warning: Unable to open video {video_full_path}. Skipping...")
            continue

        while True:
            ret, frame = cap.read()

            if not ret:
                break  # Exit loop when the video ends

            # Convert the frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame to detect hands
            results = hands.process(frame_rgb)

            # Create a completely black or peach-black frame (same size as the original frame)
            masked_frame = np.zeros_like(frame)
            masked_frame[:] = peach_black_color  # Fill with peach-black color

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw only the hand landmarks and connections on the peach-black frame
                    mp_draw.draw_landmarks(
                        masked_frame,  # Drawing on the peach-black frame
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

            # Display the masked frame with only hand landmarks visible
            cv2.imshow('Hand Landmarks on Peach-Black Background', masked_frame)

            # Press 'q' to exit the video playback early
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()  # Release the video capture object

cv2.destroyAllWindows()
hands.close()  # Close the MediaPipe hands object
