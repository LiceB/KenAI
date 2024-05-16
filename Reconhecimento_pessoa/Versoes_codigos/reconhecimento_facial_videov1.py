import cv2
import face_recognition as fr

videoPessoa = cv2.VideoCapture('videos/mulher.mp4')
listaFrames = []

while True:
    ret, frame = videoPessoa.read()
    if not ret:
        break
    imgPessoa = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    encodingPessoa = fr.face_encodings(imgPessoa)[0]
    listaFrames.append((frame, encodingPessoa))

# Inicializar o classificador de cascata para detecção de rostos
pessoa = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicializar a captura de vídeo da webcam
cap = cv2.VideoCapture(0)

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

        # Extrai a região do rosto detectado para comparar com a imagem da pessoa (Miguel)
        rosto_enc = frame[y:y+h, x:x+w]
        rosto_enc = cv2.cvtColor(rosto_enc, cv2.COLOR_BGR2RGB)

        # Calcula a codificação facial do rosto detectado
        encodingsRosto = fr.face_encodings(rosto_enc)

        # Compara as codificações faciais para reconhecimento
        if len(encodingsRosto) > 0:
            comparacao = fr.compare_faces([encodingPessoa], encodingsRosto[0])
            distancia = fr.face_distance([encodingPessoa], encodingsRosto[0])

            # Exibe o resultado do reconhecimento na imagem da webcam
            if comparacao[0]:
                cv2.putText(frame, "Mikhail", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            else:
                cv2.putText(frame, "Outra Pessoa", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Mostra os rostos com os retângulos ao redor
    cv2.imshow('Detecção de Rosto', frame)

    # Condição para sair do loop (pressionar 'esc' para sair)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

# Libera a webcam e fecha a janela
cap.release()
cv2.destroyAllWindows()
