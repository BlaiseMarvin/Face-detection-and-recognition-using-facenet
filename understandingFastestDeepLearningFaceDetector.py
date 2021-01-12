import numpy as np
import cv2

#cap = cv2.VideoCapture(0)
caffeModel = "C:/Users/LENOVO/Desktop/KMC Internship/releaseTheKraken/res10_300x300_ssd_iter_140000.caffemodel"
prototextPath = "C:/Users/LENOVO/Desktop/KMC Internship/releaseTheKraken/deploy.prototxt.txt"

#Load the model
net=cv2.dnn.readNetFromCaffe(prototextPath,caffeModel)
img=cv2.imread('test2.jpg')
(h,w)=img.shape[:2]
#img=cv2.resize(img,(00,300))

blob=cv2.dnn.blobFromImage(img,1.0,(486,640),(104.0, 177.0, 123.0))
net.setInput(blob)
detections=net.forward()

for i in range(0,detections.shape[2]):
    confidence=detections[0,0,i,2]
    if confidence > 0.4:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        cv2.rectangle(img, (startX, startY), (endX, endY),
                      (0, 0, 255), 2)


cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()






