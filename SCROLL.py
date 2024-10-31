import mediapipe as mp
import cv2
import pyautogui

# Initialize MediaPipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

print('________START__________')
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
                # Get the index and middle finger landmarks
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                index_knuckle = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
                middle_knuckle = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

                # Get the coordinates of the fingertips and knuckles
                h, w, _ = frame.shape
                index_tip_y = int(index_finger_tip.y * h)
                middle_tip_y = int(middle_finger_tip.y * h)
                index_knuckle_y = int(index_knuckle.y * h)
                middle_knuckle_y = int(middle_knuckle.y * h)

                # Draw the fingertips
                cv2.circle(frame, (int(index_finger_tip.x * w), index_tip_y), 10, (255, 0, 0), -1)  # Blue for index finger
                cv2.circle(frame, (int(middle_finger_tip.x * w), middle_tip_y), 10, (0, 255, 0), -1)  # Green for middle finger

                # Scrolling logic
                if index_tip_y > index_knuckle_y and middle_tip_y > middle_knuckle_y:
                    print("Scrolling Down")
                    pyautogui.scroll(-40)  # Scroll down
                elif index_tip_y < index_knuckle_y and middle_tip_y < middle_knuckle_y:
                    print("Scrolling Up")
                    pyautogui.scroll(40)  # Scroll up

                # Optionally, draw the hand landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Show the frame with detected fingertips
        cv2.imshow('Hand Tracking', frame)

        if cv2.waitKey(10) & 0xFF == ord('x'):
            break

capture.release()
cv2.destroyAllWindows()

print('------------------END-----------------')
