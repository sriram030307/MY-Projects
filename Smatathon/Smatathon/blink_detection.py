import cv2
import mediapipe as mp
import time
import numpy as np
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
mp_draw = mp.solutions.drawing_utils
BLINK_THRESHOLD = 0.25  
CLOSED_EYE_FRAMES = 2
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
blink_count = 0
frame_count = 0
def eye_aspect_ratio(eye_points, landmarks):#eye aspect ratio=ear 
    A = np.linalg.norm(np.array(landmarks[eye_points[1]]) - np.array(landmarks[eye_points[5]]))
    B = np.linalg.norm(np.array(landmarks[eye_points[2]]) - np.array(landmarks[eye_points[4]]))
    C = np.linalg.norm(np.array(landmarks[eye_points[0]]) - np.array(landmarks[eye_points[3]]))
    ear = (A + B) / (2.0 * C)
    return ear
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot access webcam.")
    exit()
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = [(int(l.x * w), int(l.y * h)) for l in face_landmarks.landmark]
            left_ear = eye_aspect_ratio(LEFT_EYE, landmarks)
            right_ear = eye_aspect_ratio(RIGHT_EYE, landmarks)
            avg_ear = (left_ear + right_ear) / 2.0
            if avg_ear < BLINK_THRESHOLD:
                frame_count += 1
            else:
                if frame_count >= CLOSED_EYE_FRAMES:
                    blink_count += 1
                    print(f"Blinks Detected: {blink_count}")
                frame_count = 0
            for point in LEFT_EYE + RIGHT_EYE:
                cv2.circle(frame, landmarks[point], 2, (0, 255, 0), -1)
    cv2.putText(frame, f"Blinks: {blink_count}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Eye Blink Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()

