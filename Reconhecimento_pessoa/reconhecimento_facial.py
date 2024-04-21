import cv2
import face_recognition as fr

imgMiguel = fr.load_image_file('images/miguel.png')
imgMiguel2 = cv2.cvtColor(imgMiguel,cv2.COLOR_BGR2RGB)

imgMiguelTest = fr.load_image_file('images/miguel4.jpg')
imgMiguelTest2 = cv2.cvtColor(imgMiguelTest,cv2.COLOR_BGR2RGB)

faceLoc = fr.face_locations(imgMiguel2)[0]
cv2.rectangle(imgMiguel2,(faceLoc[3],faceLoc[0]),(faceLoc[1], faceLoc[2]),(0,255,0),2) 
cv2.putText(imgMiguel2, "Miguel", (faceLoc[3],faceLoc[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

encodingsMiguel = fr.face_encodings(imgMiguel2)[0]
encodingsMiguelTest = fr.face_encodings(imgMiguelTest2)[0]

comparacao = fr.compare_faces([encodingsMiguel], encodingsMiguelTest)
distancia = fr.face_distance([encodingsMiguel], encodingsMiguelTest)    
print(comparacao, distancia)

cv2.imshow('Miguel', imgMiguel2)
cv2.imshow('Miguel2', imgMiguelTest2)
cv2.waitKey(0)