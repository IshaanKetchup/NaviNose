import mediapipe as mp
import cv2
import pyautogui
import math

# Initialize MediaPipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

print('________START__________')

# Get screen size
screen_width, screen_height = pyautogui.size()

# Define the margin from the edges of the screen
margin = 20  # Set this value as per your requirement

# Move the mouse cursor to the center of the screen
pyautogui.moveTo(screen_width // 2, screen_height // 2)

capture = cv2.VideoCapture(0)

# Smaller size for the calibration box (small square)
box_size = 30  # Size of the square box
click_box_size = 40  # Size of the click detection box

# Create a Hands object
with mp_hands.Hands(max_num_hands=1) as hands:
    while capture.isOpened():
        ret, frame = capture.read()
        frame = cv2.flip(frame, 1)  # Flip the frame horizontally
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB

        # Get frame dimensions
        h, w, _ = frame.shape
        
        # Process the image and get hand landmarks
        results = hands.process(image)

        # Check if any hands are found
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get the position of the index finger middle joint (landmark 6)
                index_finger_mid_joint = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
                mid_joint_x = int(index_finger_mid_joint.x * w)
                mid_joint_y = int(index_finger_mid_joint.y * h)

                # Calculate the square's coordinates based on the middle joint
                box_x_min = mid_joint_x - box_size // 2 - 20  # Shift left by 20 pixels
                box_x_max = mid_joint_x + box_size // 2 - 20
                box_y_min = mid_joint_y - box_size // 2
                box_y_max = mid_joint_y + box_size // 2

                # Draw the calibration box (small square)
                cv2.rectangle(frame, (box_x_min, box_y_min), (box_x_max, box_y_max), (0, 255, 0), 2)  # Green square

                # Get the position of the index finger tip
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_tip_x = int(index_finger_tip.x * w)
                index_tip_y = int(index_finger_tip.y * h)
                cv2.circle(frame, (index_tip_x, index_tip_y), 10, (255, 0, 0), -1)  # Blue for index fingertip

                # Calculate the center of the box
                box_center_x = (box_x_min + box_x_max) // 2
                box_center_y = (box_y_min + box_y_max) // 2

                # Calculate the difference between the fingertip and the box center
                delta_x = index_tip_x - box_center_x
                delta_y = index_tip_y - box_center_y

                # Calculate the distance from the fingertip to the box center
                distance = math.sqrt(delta_x**2 + delta_y**2)

                # Define base sensitivity factor for mouse movement
                base_sensitivity = 2.5  # Increase this value for faster movement

                # Adjust mouse movement sensitivity based on distance
                sensitivity = base_sensitivity * (distance / 50)  # Increase sensitivity with distance
                sensitivity = min(sensitivity, 10)  # Cap the sensitivity to prevent too fast movement

                # Calculate new mouse position
                new_mouse_x = int(pyautogui.position()[0] + delta_x * sensitivity)
                new_mouse_y = int(pyautogui.position()[1] + delta_y * sensitivity)

                # Ensure the mouse position stays within screen boundaries with margin
                new_mouse_x = max(margin, min(screen_width - margin - 1, new_mouse_x))
                new_mouse_y = max(margin, min(screen_height - margin - 1, new_mouse_y))

                # Move the mouse relative to the calculated delta
                if abs(delta_x) > box_size // 2 or abs(delta_y) > box_size // 2:
                    pyautogui.moveTo(new_mouse_x, new_mouse_y)

                # Get the position of the base of the ring finger (landmark 13)
                ring_finger_base = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]  # Change to landmark 13
                ring_base_x = int(ring_finger_base.x * w)
                ring_base_y = int(ring_finger_base.y * h)

                # Draw the click box centered at the base of the ring finger
                click_box_x_min = ring_base_x - click_box_size // 2
                click_box_x_max = ring_base_x + click_box_size // 2
                click_box_y_min = ring_base_y - click_box_size // 2
                click_box_y_max = ring_base_y + click_box_size // 2
                cv2.rectangle(frame, (click_box_x_min, click_box_y_min), (click_box_x_max, click_box_y_max), (255, 0, 0), 2)  # Blue box for clicking

                # Get the position of the thumb tip
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                thumb_tip_x = int(thumb_tip.x * w)
                thumb_tip_y = int(thumb_tip.y * h)

                # Check if the index fingertip is inside the movement box
                index_inside_box = (box_x_min <= index_tip_x <= box_x_max) and (box_y_min <= index_tip_y <= box_y_max)

                # Check if the thumb tip is inside the click box
                thumb_inside_click_box = (click_box_x_min <= thumb_tip_x <= click_box_x_max) and (click_box_y_min <= thumb_tip_y <= click_box_y_max)

                # Trigger a click only if both conditions are met
                if index_inside_box and thumb_inside_click_box:
                    pyautogui.click()  # Trigger a click

                # Optionally, draw the hand landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Show the frame with detected fingertip and calibration box
        cv2.imshow('Hand Tracking', frame)

        if cv2.waitKey(10) & 0xFF == ord('x'):
            break

capture.release()
cv2.destroyAllWindows()

print('------------------END-----------------')
