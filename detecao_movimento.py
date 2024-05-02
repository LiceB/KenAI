import cv2
import sys
from random import randint

cap = cv2.VideoCapture("videos/video5.mp4")

ok, frame = cap.read()

if not ok:
    print("Cannot read video file")
    sys.exit(1)
    
bboxes = []
colors = []

while True:
    bbox = cv2.selectROI('Tracker', frame)
    bboxes.append(bbox)
    colors.append((randint(0,255), randint(0,255), randint(0,255)))
    print("Precione Q para sair ou qualquer outra para continuar próximo objeto")
    k = cv2.waitKey(0) & 0xFF
    if (k == 113):
        break
print(bboxes)

tracker =  cv2.legacy.TrackerCSRT.create()	
multitracker = cv2.legacy.MultiTracker.create()

for bbox in bboxes:
    multitracker.add(tracker, frame, bbox)

while cap.isOpened:
    ok, frame = cap.read()
    if not ok:
        break
    
    ok , boxes = multitracker.update(frame)
    
    for i, newbox in enumerate(boxes):
        (x,y,w,h) = [int(v) for v in newbox]
        cv2.rectangle(frame, (x,y), (x+w, y+h), colors[i], 1, 1)

    cv2.imshow("video", frame)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break
