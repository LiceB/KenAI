import mediapipe as mp
import cv2

def inicia_mediapipe():
    # Inicialização do MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    return pose, mp_pose

# Função para analisar os pontos chave e determinar a posição
def analyze_pose(landmarks, mp_pose):
    # Obtenha a altura de vários pontos chave
    nose_y = landmarks[mp_pose.PoseLandmark.NOSE.value].y
    left_foot_y = landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y
    right_foot_y = landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y
    left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
    right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
    left_hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
    right_hip_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y

    # Calcular a média das alturas dos ombros e dos quadris
    shoulder_avg_y = (left_shoulder_y + right_shoulder_y) / 2
    hip_avg_y = (left_hip_y + right_hip_y) / 2

    # Determinar a postura baseada na posição relativa de várias partes do corpo
    if abs(left_foot_y - right_foot_y) < 0.1 and nose_y < shoulder_avg_y and abs(shoulder_avg_y - hip_avg_y) < 0.2:
        return 'Em Pé'
    elif left_foot_y > hip_avg_y and right_foot_y > hip_avg_y and abs(nose_y - shoulder_avg_y) < 0.2:
        return 'Deitada'
    elif (left_foot_y < hip_avg_y and right_foot_y < hip_avg_y) and (nose_y > shoulder_avg_y or abs(nose_y - shoulder_avg_y) < 0.2):
        return 'Sentada'
    else:
        return 'Indeterminado'


# Função para processar cada frame do vídeo
def process_frame(image, pose, mp_pose):
    # Converta a imagem para RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    
    # Processamento de detecção de pose
    results = pose.process(image)
    
    # Converta a imagem de volta para BGR para exibição
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if results.pose_landmarks:
        position = analyze_pose(results.pose_landmarks.landmark, mp_pose)
        cv2.putText(image, f'Position: {position}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return image

def main():
    cap = cv2.VideoCapture(0)
    pose, mp_pose = inicia_mediapipe()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Processar o frame
        frame = process_frame(frame, pose, mp_pose)
        
        # Exibir o frame
        cv2.imshow('Pose', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
