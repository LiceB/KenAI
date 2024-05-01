import mediapipe as mp
import cv2

def inicia_mediapipe():
    # Inicialização do MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    return pose, mp_pose

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
        landmarks = results.pose_landmarks.landmark
        
        for landmark in landmarks:
            x = int(landmark.x * image.shape[1])
            y = int(landmark.y * image.shape[0])
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

    return image

def main():
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("videos/video.mp4")
    pose, mp_pose = inicia_mediapipe()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        # Redimensiona o frame
        frame = cv2.resize(frame, (640, 480))

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