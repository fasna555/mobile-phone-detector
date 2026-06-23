#GOAL: Detect when a phone appears in front of the webcam.

import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture(0)
model=YOLO("yolov8n.pt")


while True:
    success,frame = cap.read()
    if not success:
        break
    results = model(frame)
    annotated = results[0].plot()

    phone_found = False
    for box in results[0].boxes:
        class_id = int(box.cls)
        name = model.names[class_id]
        if name == "cell phone":
            phone_found = True
    if phone_found:
        cv2.putText(annotated,"cell phone detected!",(50,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    cv2.imshow("phone_detector",annotated)

    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()