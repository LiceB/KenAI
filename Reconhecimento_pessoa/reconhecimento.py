import cv2

# Inicializa a webcam
cap = cv2.VideoCapture(0)git status

# Loop para capturar frames da webcam
while True:
    # Captura frame a frame
    ret, frame = cap.read()

    # Mostra o frame na janela
    cv2.imshow('Webcam', frame)

    # Condição para sair do loop (pressionar 'esc' para sair)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
# Libera a webcam e fecha a janela
cap.release()
cv2.destroyAllWindows()
