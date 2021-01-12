import cv2
from mtcnn.mtcnn import MTCNN
from matplotlib import pyplot

#img=cv2.imread('test2.jpg')
img=pyplot.imread('test2.jpg')

detector=MTCNN()
faces=detector.detect_faces(img)
img4=cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
print(len(faces))
count=10
for face in faces:

    x,y,width,height=face['box']
    x2,y2=x+width,y+height



    img1=img4[y:y2,x:x2]
    filepath="E:/Oh Faces/extractedFaces/ef" +str(count) +'.jpg'
    count+=1
    cv2.imwrite(filepath,img1)

    cv2.rectangle(img4,(x,y),(x2,y2),(0,0,255),4)


    cv2.imshow("Frame",img4)

cv2.waitKey(0)

cv2.destroyAllWindows()

