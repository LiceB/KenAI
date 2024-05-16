import cv2
import time
import mediapipe as mp

# Inicializa a captura do vídeo
video_path = "videos/Vídeo.mp4"
cap = cv2.VideoCapture(video_path)

# Verifica se o vídeo está aberto corretamente
if not cap.isOpened():
    print("Erro ao abrir o vídeo")
    exit()

# Define a taxa de atualização da flag em segundos
update_interval = 2

# Inicializa o tempo de referência
start_time = time.time()

# Inicializa o detector de pose do MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Inicializa variáveis para armazenar a posição Y do nariz nos frames anteriores
prev_nose_y = None
prev_prev_nose_y = None

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
        # Obtém a posição do nariz
        nose_landmark = result.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        nose_y = nose_landmark.y * frame.shape[0]  # Converte a posição Y normalizada para pixels
        
        # Calcula a velocidade da mudança da posição Y do nariz
        if prev_nose_y is not None and prev_prev_nose_y is not None:
            velocity = prev_prev_nose_y - prev_nose_y
            if velocity < -20:  # Limite para considerar uma queda brusca
                print("Queda brusca na posição Y do nariz:", velocity)

        # Armazena a posição Y do nariz para comparação no próximo frame
        prev_prev_nose_y = prev_nose_y
        prev_nose_y = nose_y

    # Calcula o tempo decorrido
    elapsed_time = time.time() - start_time
    
    # Atualiza a flag a cada 'update_interval' segundos
    if elapsed_time > update_interval:
        cv2.putText(frame, "Evento acionado!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        # Reinicializa o tempo de referência
        start_time = time.time()

    # Mostra o frame
    cv2.imshow('Vídeo', frame)
 
    # Verifica se a tecla 'q' foi pressionada para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a captura do vídeo e fecha todas as janelas
cap.release()
cv2.destroyAllWindows()
