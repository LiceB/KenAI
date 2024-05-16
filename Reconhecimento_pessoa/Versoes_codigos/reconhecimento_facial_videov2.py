import cv2
import face_recognition as fr

videoPessoa = cv2.VideoCapture('videos/mulher.mp4')
videoPessoa2 = fr.load_image_file('images/mulher2.mp4')
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


faceLoc = fr.face_locations(imgPessoa2)[0]
faceLoc2 = fr.face_locations(imgPessoaComp2)[0]
cv2.rectangle(imgPessoa2,(faceLoc[3],faceLoc[0]),(faceLoc[1], faceLoc[2]),(0,255,0),2) 
cv2.putText(imgPessoa2, "Miguel", (faceLoc[3],faceLoc[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

encodingsMiguel = fr.face_encodings(imgPessoa2)[0]
encodingsMiguelTest = fr.face_encodings(imgPessoaComp2)[0]

comparacao = fr.compare_faces([encodingsMiguel], encodingsMiguelTest)
distancia = fr.face_distance([encodingsMiguel], encodingsMiguelTest)    

if comparacao[0]:
    cv2.rectangle(imgPessoaComp2,(faceLoc2[3],faceLoc2[0]),(faceLoc2[1], faceLoc2[2]),(0,255,0),2) 
    cv2.putText(imgPessoaComp2, "Miguel", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
else:
    cv2.rectangle(imgPessoaComp2,(faceLoc2[3],faceLoc2[0]),(faceLoc2[1], faceLoc2[2]),(0,255,0),2) 
    cv2.putText(imgPessoaComp2, "OutraPessoa", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
