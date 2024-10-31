import mediapipe as mp
import cv2

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

                # Get the coordinates of the fingertips
                h, w, _ = frame.shape
                index_tip_coords = (int(index_finger_tip.x * w), int(index_finger_tip.y * h))
                middle_tip_coords = (int(middle_finger_tip.x * w), int(middle_finger_tip.y * h))

                # Draw the fingertips
                cv2.circle(frame, index_tip_coords, 10, (255, 0, 0), -1)  # Blue for index finger
                cv2.circle(frame, middle_tip_coords, 10, (0, 255, 0), -1)  # Green for middle finger

                # Optionally, draw the hand landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Show the frame with detected fingertips
        cv2.imshow('Hand Tracking', frame)
        
        if cv2.waitKey(10) & 0xFF == ord('x'):
            break

capture.release()
cv2.destroyAllWindows()

print('------------------END-----------------')
