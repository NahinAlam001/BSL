import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 3
dataset_size = 100

# Use the default camera index if unsure about the correct one
cap = cv2.VideoCapture(0)  # Try 0 if 2 doesn't work

if not cap.isOpened():
    print("Error: Camera not accessible.")
    exit()

for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {j}')

    # Display a prompt for the user
    done = False
    while not done:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            done = True

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        cv2.imshow('frame', frame)
        cv2.waitKey(1)  # Short delay to display the frame

        # Save the frame to file
        filename = os.path.join(class_dir, f'{counter}.jpg')
        cv2.imwrite(filename, frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()
