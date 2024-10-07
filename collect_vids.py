import pygame
import time
import os
import cv2

# Initialize pygame mixer for playing sounds
pygame.mixer.init()

# Define a function to play the notification sound
def play_audio_notification():
    pygame.mixer.music.load("click_sound.wav")  # Ensure you have this sound file
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():  # Wait until sound finishes playing
        time.sleep(0.1)

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 1
dataset_size = 2  # Total number of videos per class
video_length = 3  # Length of each video in seconds
fps = 20  # Frames per second

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera not accessible.")
    exit()

for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {j}')

    done = False
    while not done:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        cv2.putText(frame, 'Ready? Press "Q" to start recording!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            done = True

    counter = 0
    while counter < dataset_size:
        print(f'Recording video {counter+1}/{dataset_size} for class {j}')
        
        # Change codec to MJPG for Mac compatibility
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        video_filename = os.path.join(class_dir, f'{counter}.avi')
        out = cv2.VideoWriter(video_filename, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

        total_frames = video_length * fps
        frame_count = 0

        while frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            out.write(frame)
            cv2.imshow('frame', frame)
            frame_count += 1
            cv2.waitKey(1)

        out.release()
        print("Recording finished for this video.")

        # Play audio notification after recording finishes
        play_audio_notification()

        # Wait for 1 second before starting the next video
        time.sleep(0.5)

        counter += 1

cap.release()
cv2.destroyAllWindows()
