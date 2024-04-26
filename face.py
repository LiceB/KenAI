import threading 
import cv2
from deepface import DeepFace

#cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap = cv2.VideoCapture("Video.mp4")
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0

reference_images = ["morgan1.jpg", "morgan2.png", "morgan3.png"]

def check_face(frame):
    global reference_images
    for ref_image in reference_images:
        reference_img = cv2.imread(ref_image)
        try:
            if DeepFace.verify(frame, reference_img.copy())['verified']:
                return True
        except ValueError:
            pass
    return False

while True:
    ret, frame = cap.read()

    if ret:
        if counter % 30 == 0:
            try:
                threading.Thread(target=check_face, args=(frame.copy(),)).start()
            except ValueError:
                pass
        counter += 1

        if check_face(frame):
            cv2.putText(frame, "Match", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "Sem Match!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    
    cv2.imshow("video", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
