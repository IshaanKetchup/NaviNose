import mediapipe as mp
import cv2
import pyautogui

# Initialize MediaPipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

print('________START__________')

# Get screen size
screen_width, screen_height = pyautogui.size()

# Move the mouse cursor to the center of the screen
pyautogui.moveTo(screen_width // 2, screen_height // 2)

capture = cv2.VideoCapture(0)

# Create a Hands object
with mp_hands.Hands(max_num_hands=1) as hands:
    while capture.isOpened():
        ret, frame = capture.read()
        frame = cv2.flip(frame, 1)  # Flip the frame horizontally
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB

        # Process the image and get hand landmarks
        results = hands.process(image)

        # Check if any hands are found
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get the index finger tip landmark
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                # Get the coordinates of the fingertip
                h, w, _ = frame.shape
                index_tip_x = int(index_finger_tip.x * w)
                index_tip_y = int(index_finger_tip.y * h)

                # Draw the fingertip
                cv2.circle(frame, (index_tip_x, index_tip_y), 10, (255, 0, 0), -1)  # Blue for index finger

                # Map the fingertip position to the screen
                screen_x = int(index_tip_x * (screen_width / w))
                screen_y = int(index_tip_y * (screen_height / h))

                # Move the mouse cursor to the calculated position
                pyautogui.moveTo(screen_x, screen_y)

                # Optionally, draw the hand landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Show the frame with detected fingertip
        cv2.imshow('Hand Tracking', frame)

        if cv2.waitKey(10) & 0xFF == ord('x'):
            break

capture.release()
cv2.destroyAllWindows()

print('------------------END-----------------')
