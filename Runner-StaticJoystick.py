import cv2
import mediapipe as mp
import pyautogui
import time
import math

# Initialize webcam, face mesh, and hand tracking
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
hands = mp.solutions.hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Define screen dimensions
screen_width, screen_height = pyautogui.size()
tracking_circle_radius = 20

# Initialize mouse position for dampening effect
prev_mouse_x, prev_mouse_y = screen_width // 2, screen_height // 2
joystick_center = None  # Center of the joystick circle

# Initialize click tracking variables
last_click_time = 0
click_start_time = None
double_click_interval = 0.75  # 500 milliseconds for double-click
long_press_threshold = 1  # 1 second for long press
clicking_dist = 25  # Distance threshold for registering a click
click_state = 'none'

while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)  # Flip the frame horizontally
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output_face = face_mesh.process(rgb_frame)
    output_hands = hands.process(rgb_frame)  # Process the RGB frame for hand landmarks

    if output_face.multi_face_landmarks:
        landmarks_face = output_face.multi_face_landmarks[0].landmark
        nose = landmarks_face[1]  # Nose tip landmark
        nose_x = int(nose.x * frame.shape[1])  # Nose x position
        nose_y = int(nose.y * frame.shape[0])  # Nose y position

        # Initialize joystick center at the nose position
        if joystick_center is None:
            joystick_center = (nose_x, nose_y)
        dx = nose_x - joystick_center[0]
        dy = nose_y - joystick_center[1]
        distance = (dx**2 + dy**2) ** 0.5

        # If the nose is outside the circle, constrain it to the circle's edge
        if distance > tracking_circle_radius:
            # Normalize the offset and scale it to the circle radius
            nose_x = int(joystick_center[0] + (dx / distance) * tracking_circle_radius)
            nose_y = int(joystick_center[1] + (dy / distance) * tracking_circle_radius)

        # Draw the static joystick circle centered at the nose
        cv2.circle(frame, joystick_center, tracking_circle_radius, (0, 255, 0), 2)
        cv2.circle(frame, (nose_x, nose_y), 5, (255, 0, 0), -1)
        # Calculate proportional mouse position using nose position as center
        mouse_x = int((dx / tracking_circle_radius) * (screen_width / 2) + (screen_width / 2))
        mouse_y = int((dy / tracking_circle_radius) * (screen_height / 2) + (screen_height / 2))

        # Prevent the mouse from leaving the screen
        mouse_x = max(0, min(mouse_x, screen_width - 1))
        mouse_y = max(0, min(mouse_y, screen_height - 1))

        # Apply dampening effect
        mouse_x = int(prev_mouse_x + (mouse_x - prev_mouse_x) * 0.2)  # 0.2 is the dampening factor
        mouse_y = int(prev_mouse_y + (mouse_y - prev_mouse_y) * 0.2)

        # Move the mouse
        pyautogui.moveTo(mouse_x, mouse_y)

        # Update previous mouse position
        prev_mouse_x, prev_mouse_y = mouse_x, mouse_y

        # Draw nose position

    # Hand tracking code
    if output_hands.multi_hand_landmarks:
        for hand_landmarks in output_hands.multi_hand_landmarks:
            # Get index finger tip and thumb tip positions
            index_finger_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
            middle_finger_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
            index_knuckle = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP]  # Index finger knuckle
            middle_knuckle = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_MCP]  # Middle finger knuckle
            thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]

            index_y = int(index_finger_tip.y * frame.shape[0])
            middle_x = int(middle_finger_tip.x * frame.shape[1])
            middle_y = int(middle_finger_tip.y * frame.shape[0])
            index_knuckle_y = int(index_knuckle.y * frame.shape[0])
            middle_knuckle_y = int(middle_knuckle.y * frame.shape[0])

            index_x = int(index_finger_tip.x * frame.shape[1])
            index_y = int(index_finger_tip.y * frame.shape[0])
            thumb_x = int(thumb_tip.x * frame.shape[1])
            thumb_y = int(thumb_tip.y * frame.shape[0])
            mi_dist = math.sqrt((index_x - middle_x) ** 2 + (index_y - middle_y) ** 2)
            print(mi_dist)
            if index_y < index_knuckle_y and middle_y < middle_knuckle_y and mi_dist<31:  # Both fingers above knuckles
                pyautogui.scroll(50)  # Scroll up
                cv2.putText(frame, "Scrolling Up", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            elif index_y > index_knuckle_y and middle_y > middle_knuckle_y and mi_dist<31:  # Both fingers below knuckles
                pyautogui.scroll(-50)  # Scroll down
                cv2.putText(frame, "Scrolling Down", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.circle(frame, (index_x, index_y), 5, (0, 0, 255), -1)  # Draw index finger tip
            cv2.circle(frame, (thumb_x, thumb_y), 5, (0, 255, 255), -1)  # Draw thumb tip
            cv2.circle(frame, (middle_x, middle_y), 5, (0, 0, 255), -1)  # Draw index finger tip
            # Calculate the distance between the thumb and index finger tips
            distance_between_fingers = math.sqrt((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2)

            # Check if the fingers are pinching
            if distance_between_fingers < clicking_dist:
                # Start counting for long press if not already counting
                if click_start_time is None:
                    click_start_time = time.time()

                elapsed_click_time = time.time() - click_start_time

                if elapsed_click_time >= long_press_threshold:
                    if click_state != 'long':
                        pyautogui.mouseDown()  # Perform a long press
                        click_state = 'long'
                        cv2.putText(frame, "Long Pressing...", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                else:
                    # Handle single and double click logic only if not in long press state
                    if click_state == 'none':
                        if elapsed_click_time < double_click_interval:
                            # Check if within double click interval
                            if time.time() - last_click_time <= double_click_interval:
                                pyautogui.doubleClick()
                                cv2.putText(frame, "Double Click!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
                                click_state = 'double'
                            else:
                                pyautogui.click()
                                cv2.putText(frame, "Clicked!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                                last_click_time = time.time()
                                click_state = 'single'
            else:
                # Reset click tracking when fingers are released
                if click_start_time is not None:
                    if click_state == 'long':
                        pyautogui.mouseUp()  # Release long press
                    click_start_time = None  # Reset click start time
                    click_state = 'none'  # Reset state to none

    cv2.imshow('Nose Controlled Mouse and Hand Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
