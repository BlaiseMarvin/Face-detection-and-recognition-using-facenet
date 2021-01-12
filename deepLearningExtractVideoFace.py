import cv2
from mtcnn.mtcnn import MTCNN

cap=cv2.VideoCapture(0)
detector=MTCNN()

while(True):
    _,frame=cap.read()

    faces=detector.detect_faces(frame)
    for face in faces:
        x,y,width,height=face['box']
        x2,y2=x+width,y+height

        cv2.rectangle(frame,(x,y),(x2,y2),(0,0,255),4)

    cv2.imshow("Frame",frame)
    k=cv2.waitKey(1) & 0xFF
    if k==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

