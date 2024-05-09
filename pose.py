import mediapipe as mp
import cv2
import math

def inicia_mediapipe():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    return pose, mp_pose

def analyze_pose(landmarks, mp_pose, posture_history):
    nose_y = landmarks[mp_pose.PoseLandmark.NOSE.value].y
    left_foot_y = landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y
    right_foot_y = landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y
    left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
    right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
    left_hip_x = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x
    left_hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
    right_hip_x = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x
    right_hip_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
    left_knee_x = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x
    right_knee_x = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x
    left_knee_y = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y
    right_knee_y = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y

    # Distância vertical entre os ombros e os quadris
    shoulder_hip_distance = abs(left_shoulder_y - left_hip_y)

    # Distância vertical entre os quadris e os joelhos
    hip_knee_distance = abs(left_hip_y - left_knee_y)

    # Distância vertical entre os joelhos e os pés
    knee_foot_distance = abs(left_knee_y - left_foot_y)

    # Distância horizontal entre os joelhos
    knee_distance = abs(left_knee_x - right_knee_x)

    # Comprimento da linha do quadril ao joelho
    hip_knee_line_length = math.sqrt((left_hip_y - left_knee_y) ** 2 + (left_hip_x - left_knee_x) ** 2)

    # Determinando a postura atual
    if (hip_knee_line_length < knee_foot_distance and
        shoulder_hip_distance > 0.1 and
        nose_y < min(left_shoulder_y, right_shoulder_y) and
        abs(left_shoulder_y - right_shoulder_y) < 0.1 and
        abs(left_foot_y - right_foot_y) < 0.1):
        posture = 'Sentado'
    elif (hip_knee_distance > 0.1 and
          shoulder_hip_distance > 0.1 and
          nose_y < min(left_shoulder_y, right_shoulder_y) and
          abs(left_shoulder_y - right_shoulder_y) < 0.1 and
          abs(left_foot_y - right_foot_y) < 0.1):
        posture = 'Em Pe'
    else:
        posture = 'Indeterminado'

    # Suavização dos dados
    posture_history.append(posture)
    N = 5  # Número de classificações recentes a considerar
    smoothed_posture = max(set(posture_history[-N:]), key=posture_history.count)

    return smoothed_posture

def draw_landmarks(image, landmarks, mp_pose):
    upper_body_landmarks = [
        mp_pose.PoseLandmark.NOSE,
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_ELBOW,
        mp_pose.PoseLandmark.RIGHT_ELBOW,
        mp_pose.PoseLandmark.LEFT_WRIST,
        mp_pose.PoseLandmark.RIGHT_WRIST,
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.RIGHT_HIP,
        mp_pose.PoseLandmark.LEFT_KNEE,
        mp_pose.PoseLandmark.RIGHT_KNEE,
        mp_pose.PoseLandmark.LEFT_ANKLE,
        mp_pose.PoseLandmark.RIGHT_ANKLE
    ]
    
    for landmark in upper_body_landmarks:
        landmark_point = landmarks[landmark.value]
        cv2.circle(image, 
                   (int(landmark_point.x * image.shape[1]), int(landmark_point.y * image.shape[0])), 
                   5, (255, 0, 0), -1)



def process_frame(image, pose, mp_pose, posture_history):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    
    results = pose.process(image)
    
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if results.pose_landmarks:
        position = analyze_pose(results.pose_landmarks.landmark, mp_pose, posture_history)
        cv2.putText(image, f'Position: {position}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        draw_landmarks(image, results.pose_landmarks.landmark, mp_pose)

    return image

def main():
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("./videokenai.mp4")
    pose, mp_pose = inicia_mediapipe()
    posture_history = []  # Inicializando o histórico de posturas

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = process_frame(frame, pose, mp_pose, posture_history)
        
        cv2.imshow('Pose', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
