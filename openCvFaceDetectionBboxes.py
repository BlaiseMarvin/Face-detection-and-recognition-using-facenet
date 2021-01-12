import cv2

pixels=cv2.imread('test1.jpg')

classifier=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

bboxes=classifier.detectMultiScale(pixels)

for box in bboxes:
    x,y,width,height=box
    x2,y2=x+width,y+height

    #Draw rectangle over the pixels
    cv2.rectangle(pixels,(x,y),(x2,y2),(0,0,255),5)

cv2.imshow('face detection',pixels)

cv2.waitKey(0)
cv2.destroyAllWindows()