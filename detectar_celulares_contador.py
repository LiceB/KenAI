import cv2  # Importa a biblioteca OpenCV para processamento de vídeo e imagem
import time  # Importa a biblioteca time para manipulação de tempo
from ultralytics import YOLO  # Importa a biblioteca YOLO da ultralytics para detecção de objetos
import pyttsx3  # Importa a biblioteca pyttsx3 para conversão de texto em fala

# Inicializa o motor de texto para fala
engine = pyttsx3.init()

# Define o caminho para salvar o vídeo capturado
video_path = "C:/Users/Esther/OneDrive/Faculdade/4SIR - 2024/KenAI Help/output.mp4"

# Carrega o modelo YOLOv8 para detecção de objetos
model = YOLO('yolov8s')  # Escolhe o modelo desejado, por exemplo, 'yolov8s'

# Define o índice da classe "cell phone" na lista COCO (67)
target_class = 67

# Configura a captura de vídeo da webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():  # Verifica se a webcam foi aberta corretamente
    print("Erro ao abrir a câmera")
    exit()

# Configura o objeto de escrita de vídeo com a largura e altura do quadro capturado
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

# Variável de estado para alternar filtros
apply_filters = False  # Inicialmente, os filtros não são aplicados
detection_start_time = None  # Armazena o tempo de início da detecção
cell_phone_count = 0  # Contador de celulares detectados

def calculate_iou(box1, box2):
    """
    Calcula a Interseção sobre União (IoU) de duas caixas delimitadoras.
    """
    x1, y1, x2, y2 = box1
    x1_b, y1_b, x2_b, y2_b = box2

    # Calcula as coordenadas da área de interseção
    inter_rect_x1 = max(x1, x1_b)
    inter_rect_y1 = max(y1, y1_b)
    inter_rect_x2 = min(x2, x2_b)
    inter_rect_y2 = min(y2, y2_b)

    # Calcula a área de interseção
    inter_area = max(0, inter_rect_x2 - inter_rect_x1 + 1) * max(0, inter_rect_y2 - inter_rect_y1 + 1)
    # Calcula a área das caixas delimitadoras
    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x2_b - x1_b + 1) * (y2_b - y1_b + 1)

    # Calcula a razão da área de interseção pela área de união
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

print("Iniciando o loop principal...")  # Mensagem de início do loop principal
# Processar cada quadro
while True:
    ret, frame = cap.read()  # Lê um quadro da webcam
    if not ret:  # Verifica se a captura do quadro foi bem-sucedida
        print("Erro ao capturar o quadro")
        break
    
    # Verificar se os filtros devem ser aplicados
    if apply_filters:
        # Verificar se 20 segundos se passaram desde o início da detecção
        elapsed_time = time.time() - detection_start_time
        if elapsed_time > 20:  # Se mais de 20 segundos se passaram
            print(f"Tempo de detecção terminado. Celulares detectados: {cell_phone_count}")
            apply_filters = False

            # Fala a contagem de celulares detectados
            engine.say(f"{cell_phone_count} celulares detectados")
            engine.runAndWait()

            # Exibir a contagem na tela por alguns segundos
            display_count_time = time.time() + 2  # Exibir por 2 segundos
            while time.time() < display_count_time:
                count_frame = frame.copy()  # Copia o quadro atual
                cv2.putText(count_frame, f'Celulares detectados: {cell_phone_count}', (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Escreve o texto no quadro
                cv2.imshow('Detecção de Pessoas e Objetos com YOLOv8', count_frame)  # Mostra o quadro com a contagem
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cell_phone_count = 0  # Reseta a contagem após exibir
        else:
            # Aplicar YOLOv8 para detectar objetos
            results = model.predict(frame)  # Realiza a predição no quadro

            # Verificar se algum resultado foi detectado
            print(f"Detecções encontradas: {len(results)}")
            
            detected_boxes = []  # Lista para armazenar caixas detectadas
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
                            cell_phone_count += 1  # Incrementa a contagem de celulares

                        # Desenhar retângulo e rótulo
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Desenha o retângulo em torno do objeto
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Coloca o texto
                        print(f"Desenhando: {label} em ({x1}, {y1}, {x2}, {y2})")

    # Exibir o quadro
    cv2.imshow('Detecção de Pessoas e Objetos com YOLOv8', frame)

    # Escrever o quadro no vídeo
    video_writer.write(frame)
    
    # Ler a tecla pressionada
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):  # Se a tecla 'c' for pressionada, inicia a detecção
        print("Detecção iniciada.")
        apply_filters = True
        detection_start_time = time.time()
        cell_phone_count = 0  # Reseta a contagem no início da detecção
    elif key == ord('e'):  # Se a tecla 'e' for pressionada, para a detecção
        apply_filters = False
        cell_phone_count = 0  # Reseta a contagem se a detecção for interrompida
    elif key == ord('q'):  # Se a tecla 'q' for pressionada, sai do loop
        break

# Liberar a captura de vídeo e o objeto de escrita de vídeo
cap.release()  # Libera a captura de vídeo
video_writer.release()  # Libera o objeto de escrita de vídeo
cv2.destroyAllWindows()  # Fecha todas as janelas abertas do OpenCV
print("Loop principal encerrado.")  # Mensagem indicando que o loop principal foi encerrado
