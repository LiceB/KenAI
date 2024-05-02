import cv2
import face_recognition as fr

# Carregar a imagem da pessoa conhecida (Miguel) uma imagem
imgPessoa = fr.load_image_file('Reconhecimento_pessoa/images/larissa.jpg')
imgPessoa2 = cv2.cvtColor(imgPessoa, cv2.COLOR_BGR2RGB)
encodingsPessoa = fr.face_encodings(imgPessoa2)[0]

# Inicializar o classificador de cascata para detecção de rostos
pessoa = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
olhosPessoa = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
# sorrisoPessoa = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

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
        
        
    olhos = olhosPessoa.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))   
    
    for (x, y, w, h) in olhos:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # sorriso = sorrisoPessoa.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)) 
        
    # for (x, y, w, h) in sorriso:
    #     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
	
	
    # Mostra os rostos com os retângulos ao redor
    cv2.imshow('Detecção de Rosto', frame)

    # Condição para sair do loop (pressionar 'esc' para sair)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

# Libera a webcam e fecha a janela
cap.release()
cv2.destroyAllWindows()
