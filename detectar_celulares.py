# Inicialmente, simulei o pedido de detecção do celular. será feito por comando de voz, mas aqui usamos um botão para ativar o filtro de detecção.

import cv2
from ultralytics import YOLO

# Caminho para salvar o vídeo
video_path = "C:/Users/Esther/OneDrive/Faculdade/4SIR - 2024/KenAI Help/output.mp4"

# Carregar o modelo YOLOv8
model = YOLO('yolov8s')  # Escolha o modelo desejado, por exemplo, 'yolov8s'

# Classe "cell phone" na lista COCO é a de índice 67
target_class = 67

# Configurar a captura de vídeo da webcam
cap = cv2.VideoCapture(0)

# Configurar o objeto de escrita de vídeo
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

# Variável de estado para alternar filtros
apply_filters = False

# Processar cada quadro
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Verificar se os filtros devem ser aplicados
    if apply_filters:
        # Aplicar YOLOv8 para detectar objetos
        results = model.predict(frame)

        # Desenhar as detecções no quadro
        for result in results:
            for det in result.boxes:
                # Verificar se a detecção é da classe "cell phone"
                if det.cls.item() == target_class:
                    # Converter coordenadas de tensor para escalares usando .item()
                    xyxy = det.xyxy.tolist()[0]
                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    conf = det.conf.item()
                    label = f"{model.names[target_class]}: {conf:.2f}"
                    
                    # Desenhar retângulo e rótulo
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Exibir o quadro
    cv2.imshow('Detecção de Pessoas e Objetos com YOLOv8', frame)

    # Escrever o quadro no vídeo
    video_writer.write(frame)
    
    # Ler a tecla pressionada
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        apply_filters = True
    elif key == ord('e'):
        apply_filters = False
    elif key == ord('q'):
        break

# Liberar a captura de vídeo e o objeto de escrita de vídeo
cap.release()
video_writer.release()
cv2.destroyAllWindows()
