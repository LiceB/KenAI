import mediapipe as mp
import cv2
import math

# Inicializa o MediaPipe para a detecção de poses
def inicia_mediapipe():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    return pose, mp_pose

# Analisa a pose da pessoa com base nos landmarks detectados
def analyze_pose(landmarks, mp_pose, posture_history):
    # Captura a posição vertical (y) de diversos pontos de interesse
    left_foot_index_y = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y
    right_foot_index_y = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y
    left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
    right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
    left_hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
    right_hip_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
    left_knee_y = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y
    right_knee_y = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y

    # Captura a posição horizontal (x) de diversos pontos de interesse
    left_foot_index_x = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x
    right_foot_index_x = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x
    left_shoulder_x = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x
    right_shoulder_x = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x
    left_hip_x = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x
    right_hip_x = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x

    # Verifica se a pessoa está sentada 
    if (left_hip_y > left_shoulder_y and right_hip_y > right_shoulder_y and
        left_knee_y < left_hip_y and right_knee_y < right_hip_y and
        left_foot_index_y >= left_knee_y and right_foot_index_y >= right_knee_y):
        posture = 'Sentado'
        # Critérios:
        # 1. Os quadris estão abaixo dos ombros (left_hip_y > left_shoulder_y e right_hip_y > right_shoulder_y).
        # 2. Os joelhos estão mais altos que os quadris (left_knee_y < left_hip_y e right_knee_y < right_hip_y).
        # 3. Os pés estão no mesmo nível ou abaixo dos joelhos (left_foot_index_y >= left_knee_y e right_foot_index_y >= right_knee_y).
    
    # Verifica se a pessoa está em pé
    elif (left_shoulder_y < left_hip_y < left_foot_index_y and
          right_shoulder_y < right_hip_y < right_foot_index_y and
          abs(left_foot_index_y - right_foot_index_y) < 0.1 and
          abs(left_shoulder_y - right_shoulder_y) < 0.1 and
          abs(left_hip_y - right_hip_y) < 0.1):
        posture = 'Em Pe'
        # Critérios:
        # 1. Os ombros estão acima dos quadris e os quadris acima dos pés (left_shoulder_y < left_hip_y < left_foot_index_y e right_shoulder_y < right_hip_y < right_foot_index_y).
        # 2. Os pés estão no mesmo nível vertical (abs(left_foot_index_y - right_foot_index_y) < 0.1).
        # 3. Os ombros estão no mesmo nível (abs(left_shoulder_y - right_shoulder_y) < 0.1).
        # 4. Os quadris estão no mesmo nível (abs(left_hip_y - right_hip_y) < 0.1).

    else:
        posture = 'Deitado'
        # Caso nenhuma das condições anteriores seja satisfeita, assume-se que a pessoa está deitada.

    # Suavização dos dados para evitar oscilações bruscas
    posture_history.append(posture)
    N = 5  # Número de classificações recentes a considerar
    smoothed_posture = max(set(posture_history[-N:]), key=posture_history.count)

    return smoothed_posture

# Desenha os pontos de interesse no frame
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
        mp_pose.PoseLandmark.RIGHT_ANKLE,
        mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
        mp_pose.PoseLandmark.RIGHT_FOOT_INDEX
    ]
    
    for landmark in upper_body_landmarks:
        landmark_point = landmarks[landmark.value]
        cv2.circle(image, 
                   (int(landmark_point.x * image.shape[1]), int(landmark_point.y * image.shape[0])), 
                   5, (255, 0, 0), -1)

# Processa cada frame para analisar a pose e desenhar os landmarks
def process_frame(image, pose, mp_pose, posture_history):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Converte a imagem para RGB
    image.flags.writeable = False  # Torna a imagem não-editável para otimização
    
    results = pose.process(image)  # Processa a imagem para detectar os landmarks
    
    image.flags.writeable = True  # Torna a imagem editável novamente
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Converte a imagem de volta para BGR
    
    if results.pose_landmarks:
        position = analyze_pose(results.pose_landmarks.landmark, mp_pose, posture_history)  # Analisa a pose da pessoa
        cv2.putText(image, f'Position: {position}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)  # Escreve a posição na imagem
        draw_landmarks(image, results.pose_landmarks.landmark, mp_pose)  # Desenha os landmarks na imagem

    return image

def main(): 
    # cap = cv2.VideoCapture(0)  # Use a câmera ao vivo
    # cap = cv2.VideoCapture("./videokenai.mp4")  # Use um arquivo de vídeo
    # cap = cv2.VideoCapture("./videokenaisentado.mp4")  # Use um arquivo de vídeo
    cap = cv2.VideoCapture("./kenai5.mp4")  # Use um arquivo de vídeo
    pose, mp_pose = inicia_mediapipe()  # Inicializa o MediaPipe Pose
    posture_history = []  # Inicializa o histórico de posturas

    while cap.isOpened():
        success, frame = cap.read()  # Lê um frame do vídeo
        if not success:
            break

        # frame = cv2.resize(frame, (1200, 800))  # Redimensiona o frame, se necessário
        frame = process_frame(frame, pose, mp_pose, posture_history)  # Processa o frame
        
        cv2.imshow('Pose', frame)  # Mostra o frame processado
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Sai do loop se a tecla 'q' for pressionada
            break

    cap.release()  # Libera a captura de vídeo
    cv2.destroyAllWindows()  # Fecha todas as janelas do OpenCV

if __name__ == "__main__":
    main()  # Executa a função principal
