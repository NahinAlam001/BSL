import os
import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe hands and drawing utilities
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Define the color for the peach-black background (RGB format)
peach_black_color = (40, 40, 40)  # Adjust to a lighter shade if needed

DATA_DIR = './data'

for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(dir_path):
        continue 

    if dir_ != 'three':  # Optional: Only process folder '0'
        continue

    for video_path in os.listdir(dir_path):
        video_full_path = os.path.join(dir_path, video_path)
        cap = cv2.VideoCapture(video_full_path)

        if not cap.isOpened():
            print(f"Warning: Unable to open video {video_full_path}. Skipping...")
            continue

        # Initialize the MediaPipe Hands solution for each video
        hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

        while True:
            ret, frame = cap.read()
            if not ret:
                break  # Exit loop when the video ends

            # Convert the frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame to detect hands
            results = hands.process(frame_rgb)

            # Create a completely black or peach-black frame
            masked_frame = np.zeros_like(frame)
            masked_frame[:] = peach_black_color  # Fill with peach-black color

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks and connections
                    mp_draw.draw_landmarks(
                        masked_frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

            # Display the masked frame
            cv2.imshow('Hand Landmarks on Peach-Black Background', masked_frame)

            # Press 'q' to exit the video playback early
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()  # Release video capture
        hands.close()  # Close the MediaPipe hands object for the current video

cv2.destroyAllWindows()
