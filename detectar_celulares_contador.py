import cv2
import time
from ultralytics import YOLO
import pyttsx3  # Importa a biblioteca pyttsx3 para texto para fala

# Inicializa o motor de texto para fala
engine = pyttsx3.init()

# Caminho para salvar o vídeo
video_path = "C:/Users/Esther/OneDrive/Faculdade/4SIR - 2024/KenAI Help/output.mp4"

# Carregar o modelo YOLOv8
model = YOLO('yolov8s')  # Escolha o modelo desejado, por exemplo, 'yolov8s'

# Classe "cell phone" na lista COCO é a de índice 67
target_class = 67

# Configurar a captura de vídeo da webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro ao abrir a câmera")
    exit()

# Configurar o objeto de escrita de vídeo
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

# Variável de estado para alternar filtros
apply_filters = False
detection_start_time = None
cell_phone_count = 0

def calculate_iou(box1, box2):
    """
    Calcula a Interseção sobre União (IoU) de duas caixas delimitadoras.
    """
    x1, y1, x2, y2 = box1
    x1_b, y1_b, x2_b, y2_b = box2

    inter_rect_x1 = max(x1, x1_b)
    inter_rect_y1 = max(y1, y1_b)
    inter_rect_x2 = min(x2, x2_b)
    inter_rect_y2 = min(y2, y2_b)

    inter_area = max(0, inter_rect_x2 - inter_rect_x1 + 1) * max(0, inter_rect_y2 - inter_rect_y1 + 1)
    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x2_b - x1_b + 1) * (y2_b - y1_b + 1)

    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

print("Iniciando o loop principal...")
# Processar cada quadro
while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar o quadro")
        break
    
    # Verificar se os filtros devem ser aplicados
    if apply_filters:
        # Verificar se 5 segundos se passaram desde o início da detecção
        elapsed_time = time.time() - detection_start_time
        if elapsed_time > 20:  # Muda para 5 segundos
            print(f"Tempo de detecção terminado. Celulares detectados: {cell_phone_count}")
            apply_filters = False

            # Fala a contagem de celulares detectados
            engine.say(f"{cell_phone_count} celulares detectados")
            engine.runAndWait()

            # Exibir a contagem na tela por alguns segundos
            display_count_time = time.time() + 2  # Exibir por 2 segundos
            while time.time() < display_count_time:
                count_frame = frame.copy()
                cv2.putText(count_frame, f'Celulares detectados: {cell_phone_count}', (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Detecção de Pessoas e Objetos com YOLOv8', count_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cell_phone_count = 0  # Resetar a contagem após exibir
        else:
            # Aplicar YOLOv8 para detectar objetos
            results = model.predict(frame)

            # Verificar se algum resultado foi detectado
            print(f"Detecções encontradas: {len(results)}")
            
            detected_boxes = []
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
                        
                        # Verificar se a caixa atual é única (usando IoU)
                        unique = True
                        for box in detected_boxes:
                            iou = calculate_iou((x1, y1, x2, y2), box)
                            if iou > 0.5:  # Limite para considerar a mesma caixa
                                unique = False
                                break
                        if unique:
                            detected_boxes.append((x1, y1, x2, y2))
                            cell_phone_count += 1  # Incrementar a contagem de celulares

                        # Desenhar retângulo e rótulo
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        print(f"Desenhando: {label} em ({x1}, {y1}, {x2}, {y2})")

    # Exibir o quadro
    cv2.imshow('Detecção de Pessoas e Objetos com YOLOv8', frame)

    # Escrever o quadro no vídeo
    video_writer.write(frame)
    
    # Ler a tecla pressionada
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        print("Detecção iniciada.")
        apply_filters = True
        detection_start_time = time.time()
        cell_phone_count = 0  # Resetar a contagem no início da detecção
    elif key == ord('e'):
        apply_filters = False
        cell_phone_count = 0  # Resetar a contagem se a detecção for interrompida
    elif key == ord('q'):
        break

# Liberar a captura de vídeo e o objeto de escrita de vídeo
cap.release()
video_writer.release()
cv2.destroyAllWindows()
print("Loop principal encerrado.")
