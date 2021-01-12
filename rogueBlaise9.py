# develop a classifier for the 5 Celebrity Faces Dataset
from random import choice
from numpy import load
from numpy import expand_dims
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
import face_recognition
import numpy as np
# load faces

import pickle
model1=pickle.load(open('finalized_model.sav','rb'))
# test model on a random example from the test dataset

from PIL import Image
from mtcnn.mtcnn import MTCNN
from numpy import asarray
import cv2
caffeModel = "C:/Users/LENOVO/Desktop/KMC Internship/releaseTheKraken/res10_300x300_ssd_iter_140000.caffemodel"
prototextPath = "C:/Users/LENOVO/Desktop/KMC Internship/releaseTheKraken/deploy.prototxt.txt"

net=cv2.dnn.readNetFromCaffe(prototextPath,caffeModel)

def extract_face(filename, required_size=(160, 160)):
    z=[]
    # load image from file
    image = Image.open(filename)
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = asarray(image)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    for result in results:
        x1, y1, width, height = result['box']
        # bug fix
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        # extract the face
        face = pixels[y1:y2, x1:x2]
        # resize pixels to the model size
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = asarray(image)
        z.append(face_array)
    return z
def get_embedding(model, face_pixels):

	# scale pixel values
	face_pixels = face_pixels.astype('float32')
	# standardize pixel values across channels (global)
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	# transform face into one sample
	samples = expand_dims(face_pixels, axis=0)
	# make prediction to get embedding
	yhat = model.predict(samples)
	return yhat[0]

from imutils.video import VideoStream
from keras.models import load_model
model=load_model('facenet_keras.h5')

from imutils.video import FPS
import time


cap=cv2.VideoCapture(0)

while(True):
    _,frame=cap.read()
    y=extract_face(frame)

    #face=cv2.cvtColor(y,cv2.COLOR_BGR2RGB)
    emb = get_embedding(model, y)

    in_encoder = Normalizer(norm='l2')
    emb = in_encoder.transform(emb.reshape(1, -1))

    prediction = model1.predict(emb.reshape(1,-1))
    print("prediction: ", prediction)

    # drawing the bounding box
    text = "{}".format(prediction)

    cv2.putText(frame, text, (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    #fps.update()
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

#fps.stop()
cap.release()

cv2.destroyAllWindows()















#print(emb.shape)

#cv2.imshow('face',face)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#prez=model1.predict(emb.reshape(1,-1))
#print(prez)


