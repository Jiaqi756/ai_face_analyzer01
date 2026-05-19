import mediapipe as mp
import cv2

mp_face = mp.solutions.face_detection

detector = mp_face.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)

def detect_faces(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb)

    faces = []

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            faces.append(bbox)

    return faces
