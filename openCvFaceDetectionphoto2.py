import cv2

pixels=cv2.imread("test2.jpg")

classifier=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

bboxes=classifier.detectMultiScale(pixels,1.05,8)

for box in bboxes:
    x,y,width,height=box
    x2,y2=x+width,y+height

    cv2.rectangle(pixels,(x,y),(x2,y2),(0,0,255),2)

cv2.imshow('faces',pixels)

cv2.waitKey(0)
cv2.destroyAllWindows()