import cv2
from ultralytics import YOLO

# Caminho para salvar o vídeo
video_path = "C:/Users/Esther/OneDrive/Faculdade/4SIR - 2024/KenAI Help/output.mp4"

# Carregar o modelo YOLOv8
model = YOLO('yolov8s')  # Escolha o modelo desejado, por exemplo, 'yolov8s'

# Configurar a captura de vídeo da webcam
cap = cv2.VideoCapture(0)

# Inicializar o objeto de escrita de vídeo
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

# Lista de objetos detectados e suas coordenadas
detected_objects = {}

# Processar cada quadro
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Aplicar YOLOv8 para detectar objetos
    results = model.predict(frame)

    # Limpar a lista de objetos detectados
    detected_objects.clear()

    # Coletar as coordenadas das detecções
    for result in results:
        for det in result.boxes:
            xyxy = det.xyxy.tolist()[0]
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            detected_objects[det.cls.item()] = (x1, y1, x2 - x1, y2 - y1)  # Formato: (x, y, w, h)

    # Desenhar as caixas delimitadoras (bounding boxes) das detecções
    for obj_id, bbox in detected_objects.items():
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"Object {obj_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Exibir o quadro
    cv2.imshow('Detecção e Rastreamento de Objetos', frame)

    # Escrever o quadro no vídeo
    video_writer.write(frame)
    
    # Parar o loop ao pressionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a captura de vídeo e o objeto de escrita de vídeo
cap.release()
video_writer.release()
cv2.destroyAllWindows()
