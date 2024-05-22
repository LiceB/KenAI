import cv2
import time
import mediapipe as mp
import pyttsx3
import threading

# Inicializa a captura do vídeo
video_path = "videos/video.mp4"
cap = cv2.VideoCapture(video_path)

# Verifica se o vídeo está aberto corretamente
if not cap.isOpened():
    print("Erro ao abrir o vídeo")
    exit()

# Define a duração que a mensagem "Queda!" deve ser exibida (em segundos)
message_duration = 3

# Inicializa o tempo da última detecção de queda e o tempo da última repetição do áudio
last_fall_time = None
last_audio_time = None

# Inicializa o mecanismo de text-to-speech
engine = pyttsx3.init()

# Função para falar "Queda!"
def falar_queda():
    engine.say("Queda!")
    engine.runAndWait()

# Função para iniciar a fala em um thread separado
def falar_queda_thread():
    thread = threading.Thread(target=falar_queda)
    thread.start()

# Inicializa o detector de pose do MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Inicializa variáveis para armazenar a posição Y do nariz nos frames anteriores
prev_nose_y = None
prev_prev_nose_y = None

# Variável para controlar a primeira detecção
first_fall_detected = False

while True:
    # Captura frame-by-frame
    ret, frame = cap.read()
    
    # Verifica se o frame foi capturado corretamente
    if not ret:
        print("Erro ao capturar o frame")
        break

    # Detecta a pose no frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(frame_rgb)

    # Verifica se a pose foi detectada
    if result.pose_landmarks:
        # Desenha os landmarks no frame
        mp_drawing = mp.solutions.drawing_utils 
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Obtém a posição do nariz
        nose_landmark = result.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        nose_y = nose_landmark.y * frame.shape[0]  # Converte a posição Y normalizada para pixels
        
        # Obtém a posição dos pés
        left_foot_landmark = result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL]
        left_foot_y = left_foot_landmark.y * frame.shape[0]
        
        right_foot_landmark = result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL]
        right_foot_y = right_foot_landmark.y * frame.shape[0]
        
        # Define um limite de proximidade para considerar uma queda
        proximity_limit = 60  # pixels
        
        # Verifica se o nariz está próximo ao nível do pé
        if abs(nose_y - left_foot_y) < proximity_limit or abs(nose_y - right_foot_y) < proximity_limit:
            current_time = time.time()
            if not first_fall_detected:
                last_fall_time = current_time
                last_audio_time = current_time
                first_fall_detected = True
                falar_queda_thread()  # Fala "Queda!" na primeira detecção em um thread separado
            elif (current_time - last_audio_time) >= 5:
                last_audio_time = current_time
                falar_queda_thread()  # Fala "Queda!" novamente após 5 segundos em um thread separado

    # Verifica se a mensagem de queda deve ser exibida
    if first_fall_detected and (time.time() - last_fall_time) < message_duration:
        cv2.putText(frame, "Queda!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Mostra o frame
    cv2.imshow('Detecção de queda', frame)
 
    # Verifica se a tecla 'q' foi pressionada para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a captura do vídeo e fecha todas as janelas
cap.release()
cv2.destroyAllWindows()