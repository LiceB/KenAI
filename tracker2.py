import cv2
import numpy as np

# Carregar o modelo YOLOv5 ONNX
net = cv2.dnn.readNet('yolov5m.onnx')

# Inicializar o rastreador
tracker = None

x1, y1, w1, h1 = 100, 100, 50, 50

class_ids = [0, 1, 2]
confidences = [0.8, 0.7, 0.6]
boxes = [(x1, y1, w1, h1)]

# Carregar os nomes das classes
classes_file = "coco.names"
classes = None
with open(classes_file, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Inicializar a detecção e o rastreamento
def init_detection_and_tracking(frame):
    global tracker
    global net

    # Obter as dimensões do quadro
    frame_height, frame_width = frame.shape[:2]

    # Pré-processamento da imagem para entrada na rede YOLOv5
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), [0, 0, 0], 1, crop=False)
    net.setInput(blob)

    # Obter as saídas da rede YOLOv5
    outputs = net.forward()

    # Encontrar o objeto com a maior confiança para rastreamento
    best_confidence = 0
    best_box = None
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id in class_ids:
                cx, cy, w, h = (detection[0:4] * np.array([frame_width, frame_height, frame_width, frame_height])).astype('int')
                left = int(cx - w / 2)
                top = int(cy - h / 2)
                best_confidence = confidence
                best_box = (left, top, int(w), int(h))

    # Verificar se uma caixa delimitadora válida foi encontrada
    if best_box is not None:
        print("Caixa delimitadora encontrada:", best_box)

        # Inicializar o rastreador com o melhor objeto encontrado
        tracker = cv2.TrackerKCF_create()  # Use o rastreador KCF
        init_success = tracker.init(frame, best_box)

        # Verificar se o rastreador foi inicializado com sucesso
        if init_success:
            print("Rastreador inicializado com sucesso")
        else:
            print("Falha na inicialização do rastreador")
    else:
        print("Nenhum objeto detectado para rastreamento")

# Atualizar a detecção e o rastreamento
def update_detection_and_tracking(frame):
    global tracker

    if tracker is not None:
        success, box = tracker.update(frame)
        if success:
            left, top, width, height = [int(v) for v in box]
            cv2.rectangle(frame, (left, top), (left + width, top + height), (0, 255, 0), 2)
            
            # Loop sobre as classes e mostrar os nomes dos objetos detectados
            for class_id, confidence, box in zip(class_ids, confidences, boxes):
                class_name = classes[class_id]
                print(f"Object: {class_name}, Confidence: {confidence}")
            
                # Adicionar código para usar as classes detectadas
                print("Objeto detectado:", class_name)
        else:
            print("Falha no rastreamento do objeto")
    else:
        print("Rastreador não inicializado")

# Inicializar a captura de vídeo
cap = cv2.VideoCapture('video.mp4')
ret, frame = cap.read()

# Inicializar detecção e rastreamento
init_detection_and_tracking(frame)

# Loop de captura e processamento de vídeo
while True:
    ret, frame = cap.read()
    if not ret:
        break

    update_detection_and_tracking(frame)

    cv2.imshow('YOLO Object Detection and Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
