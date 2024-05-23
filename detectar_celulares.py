import cv2  # Importa a biblioteca OpenCV para manipulação de vídeo e imagem
import time  # Importa a biblioteca time para manipulação de tempo
from ultralytics import YOLO  # Importa a classe YOLO da biblioteca ultralytics para detecção de objetos

# Caminho para salvar o vídeo
video_path = "C:/Users/Esther/OneDrive/Faculdade/4SIR - 2024/KenAI Help/output.mp4"

# Carregar o modelo YOLOv8
model = YOLO('yolov8s')  # Carrega o modelo YOLOv8, usando a versão 'yolov8s'

# Classe "cell phone" na lista COCO é a de índice 67
target_class = 67  # Define a classe alvo (cell phone) com o índice 67

# Configurar a captura de vídeo da webcam
cap = cv2.VideoCapture(0)  # Inicia a captura de vídeo da webcam (índice 0)

# Configurar o objeto de escrita de vídeo
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Obtém a largura do quadro
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Obtém a altura do quadro
# Configura o objeto de escrita de vídeo com o caminho, codec, FPS e tamanho do quadro
video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

# Variável de estado para alternar filtros
apply_filters = False  # Inicializa o estado do filtro como False
detection_start_time = None  # Inicializa a variável para armazenar o tempo de início da detecção

# Processar cada quadro
while True:
    ret, frame = cap.read()  # Lê um quadro da webcam
    if not ret:  # Se a leitura falhar, sai do loop
        break
    
    # Verificar se os filtros devem ser aplicados
    if apply_filters:
        # Verificar se 5 segundos se passaram desde o início da detecção
        if time.time() - detection_start_time > 5:  # Verifica se passaram mais de 5 segundos
            apply_filters = False  # Desativa os filtros após 5 segundos
        else:
            # Aplicar YOLOv8 para detectar objetos
            results = model.predict(frame)  # Faz a previsão de objetos no quadro

            # Desenhar as detecções no quadro
            for result in results:
                for det in result.boxes:
                    # Verificar se a detecção é da classe "cell phone"
                    if det.cls.item() == target_class:  # Verifica se a classe detectada é "cell phone"
                        # Converter coordenadas de tensor para escalares usando .item()
                        xyxy = det.xyxy.tolist()[0]
                        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                        conf = det.conf.item()  # Obtém a confiança da detecção
                        label = f"{model.names[target_class]}: {conf:.2f}"  # Cria o rótulo com o nome e a confiança
                        
                        # Desenhar retângulo e rótulo
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Desenha o retângulo ao redor do objeto
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Desenha o rótulo

    # Exibir o quadro
    cv2.imshow('Detecção de Pessoas e Objetos com YOLOv8', frame)  # Exibe o quadro com as detecções

    # Escrever o quadro no vídeo
    video_writer.write(frame)  # Escreve o quadro no vídeo de saída
    
    # Ler a tecla pressionada
    key = cv2.waitKey(1) & 0xFF  # Lê a tecla pressionada
    if key == ord('c'):  # Se a tecla 'c' for pressionada
        apply_filters = True  # Ativa os filtros
        detection_start_time = time.time()  # Armazena o tempo de início da detecção
    elif key == ord('e'):  # Se a tecla 'e' for pressionada
        apply_filters = False  # Desativa os filtros
    elif key == ord('q'):  # Se a tecla 'q' for pressionada
        break  # Sai do loop

# Liberar a captura de vídeo e o objeto de escrita de vídeo
cap.release()  # Libera a captura de vídeo
video_writer.release()  # Libera o objeto de escrita de vídeo
cv2.destroyAllWindows()  # Fecha todas as janelas abertas pelo OpenCV
