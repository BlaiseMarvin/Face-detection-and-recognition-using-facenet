import cv2

cap=cv2.VideoCapture(0)
classifier=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while(True):
    _,image=cap.read()

    bboxes=classifier.detectMultiScale(image,1.05,15)

    for box in bboxes:
        x,y,width,height=box
        x2,y2=x+width,y+height

        cv2.rectangle(image,(x,y),(x2,y2),(0,0,255),3)

    cv2.imshow("Frame",image)
    k=cv2.waitKey(1) & 0xFF
    if k==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
