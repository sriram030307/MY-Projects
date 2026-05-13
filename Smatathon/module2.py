import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Landmark indices for eye corners
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263

# Approximate real interpupillary distance (IPD) in cm
REAL_IPD_CM = 6.3  

# Estimated focal length (To be adjusted based on test results)
FOCAL_LENGTH = 650  

def calculate_distance(landmarks):
    """Estimate distance from camera using eye corner distance."""
    if landmarks is None:
        return None

    try:
        # Get eye corner positions
        left_eye_corner = np.array(landmarks[LEFT_EYE_OUTER])
        right_eye_corner = np.array(landmarks[RIGHT_EYE_OUTER])
        
        # Compute pixel distance
        pixel_distance = np.linalg.norm(left_eye_corner - right_eye_corner)
        
        if pixel_distance == 0:
            return None  # Avoid division errors
        
        # Compute real-world distance
        distance_cm = (REAL_IPD_CM * FOCAL_LENGTH) / pixel_distance

        print(f"Pixel Distance: {pixel_distance:.2f}, Estimated Distance: {distance_cm:.2f} cm")
        
        return distance_cm
    except Exception as e:
        print("Error in distance calculation:", e)
        return None

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = {i: (int(l.x * w), int(l.y * h)) for i, l in enumerate(face_landmarks.landmark)}

            # Calculate and display distance
            distance = calculate_distance(landmarks)
            if distance:
                cv2.putText(frame, f"Distance: {distance:.2f} cm", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("Distance Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

