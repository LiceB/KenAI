import cv2

# Classificador de rostos
pessoa = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicializa a webcam
cap = cv2.VideoCapture(0)

# Loop para capturar frames da webcam
while True:
    # Captura frame a frame
    ret, frame = cap.read()

    # Converte para a escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecta rostos na imagem em escala de cinza
    rosto = pessoa.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Desenha retângulos ao redor dos rostos detectados
    for (x, y, w, h) in rosto:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (153, 51, 153), 2)

    # Mostra os rostos com os retângulos ao redor
    cv2.imshow('Detecção de Rosto', frame)

    # Condição para sair do loop (pressionar 'esc' para sair)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
# Libera a webcam e fecha a janela
cap.release()
cv2.destroyAllWindows()
