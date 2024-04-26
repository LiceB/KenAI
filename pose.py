import mediapipe as mp
import cv2
import time

# Variáveis globais para armazenar o estado anterior
prev_nose_y = 0
prev_time = 0
fall_detected_time = 0
fall_display_time = 30  # Tempo em quadros para exibir o texto de queda

def inicia_mediapipe():
    # Inicialização do MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    return pose, mp_pose

def analyze_pose(landmarks, mp_pose):
    # Obtenha a altura do ponto do nariz
    nose_y = landmarks[mp_pose.PoseLandmark.NOSE.value].y
    return nose_y

def detect_fall(nose_y):
    global prev_nose_y, prev_time, fall_detected_time

    # Calcula o tempo decorrido desde o último quadro
    current_time = time.time()
    time_elapsed = current_time - prev_time

    # Calcula a velocidade vertical
    if time_elapsed > 0:
        velocity = (nose_y - prev_nose_y) / time_elapsed
    else:
        velocity = 0

    # Define o limite de velocidade para detecção de queda
    fall_velocity_limit = -1  # Defina o limite de velocidade de queda conforme necessário

    # Verifica se a velocidade excede o limite
    if velocity < fall_velocity_limit:
        fall_detected_time = 0  # Reinicia o contador de tempo de queda
        return True
    else:
        fall_detected_time += 1
        return False

def process_frame(image, pose, mp_pose):
    global prev_nose_y, prev_time, fall_detected_time

    # Converta a imagem para RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Processamento de detecção de pose
    results = pose.process(image)

    # Converta a imagem de volta para BGR para exibição
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        nose_y = analyze_pose(landmarks, mp_pose)

        # Determina a posição da pessoa
        position = analyze_position(landmarks, mp_pose)

        # Desenha a posição da pessoa na imagem
        cv2.putText(image, f'Position: {position}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Verifica queda
        if detect_fall(nose_y):
            cv2.putText(image, 'Queda detectada!', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Exibe o texto de queda por um número fixo de quadros
        if fall_detected_time < fall_display_time:
            cv2.putText(image, 'Queda detectada!', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Atualiza as variáveis globais
        prev_nose_y = nose_y
        prev_time = time.time()

    return image

def analyze_position(landmarks, mp_pose):
    # Obter as coordenadas y dos pontos chave relevantes
    nose_y = landmarks[mp_pose.PoseLandmark.NOSE.value].y
    left_foot_y = landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y
    right_foot_y = landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y
    left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
    right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
    left_hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
    right_hip_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y

    # Calcular a média das alturas dos ombros
    shoulder_avg_y = (left_shoulder_y + right_shoulder_y) / 2
    hip_avg_y = (left_hip_y + right_hip_y) / 2

    # Determinar a postura baseada na posição relativa de várias partes do corpo
    if abs(left_foot_y - right_foot_y) < 0.1 and nose_y < shoulder_avg_y:
        return 'Em Pe'
    elif left_foot_y > hip_avg_y and right_foot_y > hip_avg_y and abs(nose_y - shoulder_avg_y) < 0.2:
        return 'Deitada'
    else:
        return 'Indeterminado'

def main():
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("videos/Vídeo.mp4")
    pose, mp_pose = inicia_mediapipe()
    # inicia o yolo
    # pega o código que inicia o yolo

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Processar o frame
        frame = process_frame(frame, pose, mp_pose)
        # processa frame yolo
        # entrada do yolo ser a saída do mediapipe ou vise versa
        # varificar se a classe do yolo é o que queremos detectar, assim temos que pegar o bonderbox da cama e da pessoa para
        # ver se a pessoa está em cima deitada.

        # Exibir o frame
        cv2.imshow('Pose', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
