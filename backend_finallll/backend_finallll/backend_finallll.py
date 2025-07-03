import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, Response

app = Flask(__name__)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

def aspect_ratio(eye_landmarks):
    vertical1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    vertical2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    horizontal = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    return (vertical1 + vertical2) / (2.0 * horizontal)

def detect_blinks():
    cap = cv2.VideoCapture(0)
    blink_count = 0
    eye_closed_frames = 0
    EYE_AR_THRESHOLD = 0.2
    FRAME_THRESHOLD = 3
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_eye = np.array([[face_landmarks.landmark[i].x, face_landmarks.landmark[i].y] for i in LEFT_EYE_INDICES])
                right_eye = np.array([[face_landmarks.landmark[i].x, face_landmarks.landmark[i].y] for i in RIGHT_EYE_INDICES])
                
                left_eye_ar = aspect_ratio(left_eye)
                right_eye_ar = aspect_ratio(right_eye)
                avg_eye_ar = (left_eye_ar + right_eye_ar) / 2
                
                if avg_eye_ar < EYE_AR_THRESHOLD:
                    eye_closed_frames += 1
                else:
                    if eye_closed_frames >= FRAME_THRESHOLD:
                        blink_count += 1
                    eye_closed_frames = 0
                
                cv2.putText(frame, f'Blinks: {blink_count}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Blink Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

@app.route('/start_blink_detection', methods=['GET'])
def start_blink_detection():
    detect_blinks()
    return Response("Blink detection started.", status=200)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
