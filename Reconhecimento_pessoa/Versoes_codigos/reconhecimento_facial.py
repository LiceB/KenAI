import cv2
import face_recognition as fr

imgPessoa = fr.load_image_file('images/miguel.png')
imgPessoa2 = cv2.cvtColor(imgPessoa,cv2.COLOR_BGR2RGB)

imgPessoaComp = fr.load_image_file('images/larissa.jpg')
imgPessoaComp2 = cv2.cvtColor(imgPessoaComp,cv2.COLOR_BGR2RGB)

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


cv2.imshow('Miguel', imgPessoa2)
cv2.imshow('Miguel2', imgPessoaComp2)
cv2.waitKey(0)

