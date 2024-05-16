import os
import cv2
import numpy as np
import face_recognition
from sklearn.neighbors import KNeighborsClassifier

def carregar_imagens(pasta):
    imagens = []
    for arquivo in os.listdir(pasta):
        imagem = face_recognition.load_image_file(os.path.join(pasta, arquivo))
        encoding = face_recognition.face_encodings(imagem)[0]  
        imagens.append(encoding)
    return imagens

pasta_pessoa1 = "D:/FIAP/2024/1_Semestre/AI ENGENEERING, COGNITIVE AND SEMANTIC COMPUTATION & IOT/CP/KenAI/Reconhecimento_pessoa/images/Larissa"
pasta_pessoa2 = "D:/FIAP/2024/1_Semestre/AI ENGENEERING, COGNITIVE AND SEMANTIC COMPUTATION & IOT/CP/KenAI/Reconhecimento_pessoa/images/Miguel"

encodings_pessoa1 = carregar_imagens(pasta_pessoa1)
encodings_pessoa2 = carregar_imagens(pasta_pessoa2)

X_treino = encodings_pessoa1 + encodings_pessoa2
y_treino = ["Pessoa1"] * len(encodings_pessoa1) + ["Pessoa2"] * len(encodings_pessoa2)



knn = KNeighborsClassifier(n_neighbors=3)   
knn.fit(X_treino, y_treino)

 
def reconhecimento_facil(modelo):
    video_capture = cv2.VideoCapture(0) 

    while True:
 
        ret, frame = video_capture.read()

        rgb_frame = frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_frame)
        if len(face_locations) > 0:
    
            encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]   
     
            pessoa_predita = modelo.predict([encoding])[0]
    
            top, right, bottom, left = face_locations[0]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, pessoa_predita, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

      
        cv2.imshow('Video', frame)

     
        if cv2.waitKey(1) & 0xFF == ord('esc'):
            break

 
    video_capture.release()
    cv2.destroyAllWindows()

reconhecimento_facil(knn)
