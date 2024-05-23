import cv2
import face_recognition as fr

# Inicializar a captura de vídeo do arquivo
videoPessoa = cv2.VideoCapture('videos/Larissa.mp4')

# Lista para armazenar codificações de rostos do vídeo
video_encodings = []

# Processar todos os frames do vídeo e armazenar as codificações faciais
while True:
    ret, frame_video = videoPessoa.read()
    if not ret:
        break
    
    # Converter o frame do vídeo para RGB
    imgPessoa = cv2.cvtColor(frame_video, cv2.COLOR_BGR2RGB)
    
    # Encontrar as localizações dos rostos no frame
    face_locations_video = fr.face_locations(imgPessoa)
    
    # Codificar os rostos no frame
    encodingsVideo = fr.face_encodings(imgPessoa, face_locations_video)
    
    # Armazenar as codificações dos rostos
    video_encodings.extend(encodingsVideo)

# Liberar o vídeo
videoPessoa.release()

# Inicializar a captura de vídeo da webcam
webcam = cv2.VideoCapture(0)

while True:
    # Capturar um frame da webcam
    ret, frame_webcam = webcam.read()
    if not ret:
        break
    
    # Converter o frame da webcam para RGB
    imgWebcam = cv2.cvtColor(frame_webcam, cv2.COLOR_BGR2RGB)
    
    # Encontrar as localizações dos rostos no frame da webcam
    face_locations_webcam = fr.face_locations(imgWebcam)
    
    # Codificar os rostos no frame da webcam
    encodingsWebcam = fr.face_encodings(imgWebcam, face_locations_webcam)
    
    # Comparar cada rosto encontrado no frame da webcam com os rostos do vídeo
    for (top, right, bottom, left), encoding in zip(face_locations_webcam, encodingsWebcam):
        matches = fr.compare_faces(video_encodings, encoding)
        
        if any(matches):
            label = "Larissa"
            color = (0, 255, 0)  # Verde para correspondência
        else:
            label = "Outra pessoa"
            color = (0, 0, 255)  # Vermelho para não correspondência
        
        # Desenhar o retângulo ao redor do rosto detectado na webcam
        cv2.rectangle(frame_webcam, (left, top), (right, bottom), color, 2)
        cv2.putText(frame_webcam, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Redimensionar o frame da webcam para uma proporção menor
    scale_percent = 50  # Percentual do tamanho original
    width = int(frame_webcam.shape[1] * scale_percent / 100)
    height = int(frame_webcam.shape[0] * scale_percent / 100)
    resized_frame = cv2.resize(frame_webcam, (width, height), interpolation=cv2.INTER_AREA)
    
    # Exibir o frame redimensionado da webcam
    cv2.imshow('Detecção de Rosto', resized_frame)
    # cv2.imshow('Detecção de Rosto', frame_webcam)
    
    # Parar o vídeo ao pressionar a tecla 'ESC'
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Liberar a webcam e fechar todas as janelas
webcam.release()
cv2.destroyAllWindows()
