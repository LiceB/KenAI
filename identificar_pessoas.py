import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            print("Ignoring No Video in Camera frame")
            continue
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = pose.process(image)
        
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245,117,66), thickness=4, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245,66,230), thickness=4)
                                  )
        
        