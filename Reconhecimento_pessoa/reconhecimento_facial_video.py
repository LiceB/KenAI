import cv2
import face_recognition as fr

# Carregar a imagem da pessoa conhecida (Miguel) uma imagem
imgPessoa = fr.load_image_file('images/miguel.png')
imgPessoa2 = cv2.cvtColor(imgPessoa, cv2.COLOR_BGR2RGB)
encodingsMiguel = fr.face_encodings(imgPessoa2)[0]


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
        rosto_enc = cv2.resize(rosto_enc, (imgPessoa2.shape[1], imgPessoa2.shape[0]))

        # Calcula a codificação facial do rosto detectado
        encodingsRosto = fr.face_encodings(rosto_enc)

        # Compara as codificações faciais para reconhecimento
        if len(encodingsRosto) > 0:
            comparacao = fr.compare_faces([encodingsMiguel], encodingsRosto[0])
            distancia = fr.face_distance([encodingsMiguel], encodingsRosto[0])

            # Exibe o resultado do reconhecimento na imagem da webcam
            if comparacao[0]:
                cv2.putText(frame, "Miguel", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
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
