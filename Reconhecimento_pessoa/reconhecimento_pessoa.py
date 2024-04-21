import cv2
import face_recognition as fr

imgMiguel = fr.load_image_file('images/miguel.jpg')
imgMiguel = cv2.transpose(imgMiguel)
imgMiguel = cv2.cvtColor(imgMiguel,cv2.COLOR_BGR2RGB)

imgMiguelTest = fr.load_image_file('images/miguelTest.jpg')
imgMiguelTest = cv2.transpose(imgMiguelTest)
imgMiguelTest = cv2.cvtColor(imgMiguelTest,cv2.COLOR_BGR2RGB)

faceLoc = fr.face_locations(imgMiguel)[0]
cv2.rectangle(imgMiguel,(faceLoc[3],faceLoc[0]),(faceLoc[1], faceLoc[2]),(0,255,0),2) 

encodeMiguel = fr.face_encodings(imgMiguel)[0]
encodeMiguelTest = fr.face_encodings(imgMiguelTest)[0]

comparar = fr.compare_faces([encodeMiguel],[encodeMiguelTest])

print(comparar)

cv2.imshow('Miguel', imgMiguel)
cv2.waitKey(0)