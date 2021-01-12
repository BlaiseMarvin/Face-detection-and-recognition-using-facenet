import numpy as np
import pickle
import cv2
import os
from numpy import expand_dims
protopath="E:/FaceRecognitionFaceNet2.0/face_detection_model/deploy.prototxt.txt"
modelpath="E:/FaceRecognitionFaceNet2.0/face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"

detector=cv2.dnn.readNetFromCaffe(protopath,modelpath)

#load embedding model
embpath="C:/Users/LENOVO/Downloads/openface.nn4.small2.v1.t7"
embedder=cv2.dnn.readNetFromTorch(embpath)

knownEmbeddings=[]
knownNames=[]

total=0
datasetpath="E:/FaceRecognitionFaceNet2.0/dataset/"

def embeddings_and_names(datasetpath):
    embeddings=[]
    names=[]
    for directory in os.listdir(datasetpath):

        newpath=datasetpath+directory+'/'
        for filename in os.listdir(newpath):
            #print(filename)
            imgpath=newpath+filename
            image=cv2.imread(imgpath)
            #image=cv2.resize(image,(600,600))
            image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

            (h,w)=image.shape[:2]

            imageBlob = cv2.dnn.blobFromImage(
                cv2.resize(image, (300, 300)), 1.0, (300, 300),
                (104.0, 177.0, 123.0), swapRB=False, crop=False)

            detector.setInput(imageBlob)
            detections=detector.forward()

            if len(detections)>0:
                i = np.argmax(detections[0, 0, :, 2])
                confidence = detections[0, 0, i, 2]


                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    face = image[startY:endY, startX:endX]
                    face=cv2.resize(face,(160,160))
                    face=face.astype('float32')



                    (fH, fW) = face.shape[:2]

                    if fW < 20 or fH < 20:
                        continue

                    faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                                     (96, 96), (0, 0, 0), swapRB=True, crop=False)

                    embedder.setInput(faceBlob)
                    vec=embedder.forward()

                    embeddings.append(vec.flatten())
                    names.append(directory)

    return np.asarray(embeddings),np.asarray(names)
                    #knownEmbeddings.append(vec.flatten())
                    #knownNames.append(directory)


trainX,trainy=embeddings_and_names("E:/FaceRecognitionFaceNet2.0/train/")
testX,testy=embeddings_and_names("E:/FaceRecognitionFaceNet2.0/test/")

print("Train X shape: ",trainX.shape)
print("Train y shape: ",trainy.shape)

print("Test X shape: ",testX.shape)
print("Test y shape: ",testy.shape)






from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC

#load dataset




#normalize input vectors
in_encoder=Normalizer(norm='l2')
trainX=in_encoder.transform(trainX)
testX=in_encoder.transform(testX)

#label encode targets
out_encoder=LabelEncoder()
out_encoder.fit(trainy)
trainy=out_encoder.transform(trainy)
testy=out_encoder.transform(testy)

#fit model
model=SVC(C=1.0,kernel='linear',probability=True)
model.fit(trainX,trainy)

#predict
yhat_train=model.predict(trainX)
yhat_test=model.predict(testX)

#score
score_train=accuracy_score(trainy,yhat_train)

score_test=accuracy_score(testy,yhat_test)

#summarize
print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))



#Testing an Image


def extract_face(filename):
    image = cv2.imread(filename)
    # image=cv2.resize(image,(600,600))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    (h, w) = image.shape[:2]

    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    detector.setInput(imageBlob)
    detections = detector.forward()

    if len(detections) > 0:
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = image[startY:endY, startX:endX]
            face = cv2.resize(face, (160, 160))
            face = face.astype('float32')

            (fH, fW) = face.shape[:2]



            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                             (96, 96), (0, 0, 0), swapRB=True, crop=False)

            embedder.setInput(faceBlob)
            vec = embedder.forward()
            vec1=vec.flatten()
            xg=vec1.reshape(1,-1)
            return xg


zx=model.predict(extract_face("E:/FaceRecognitionFaceNet2.0/test/unknown/80dbbafa223654a99bd6d7e79dfdf6fa.jpg"))
print(zx)


