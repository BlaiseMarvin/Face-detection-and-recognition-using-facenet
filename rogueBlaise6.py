#develop a classifier
from random import choice
from numpy import load
from numpy import expand_dims
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from matplotlib import pyplot

#load faces that you are going to test on
data=load('rogue_blaise.npz')
testX_faces=data['arr_2']

#each image from the above set is of shape: (160,160,3)
#print(testX_faces[0].shape)
#load face embeddings
#facenet generated embeddings, which I stored in a compressed npz file, it's these embeddings that act as inputs to my classifier
data=load('rogue_blaise-embeddings.npz')
trainX,trainy,testX,testy=data['arr_0'],data['arr_1'],data['arr_2'],data['arr_3']
#the above are basically embeddings and names
#print(trainX[0].shape)
print(testX[0])
#embeddings of images are basically of shape(128,), meaning one row with 128 columns

#Normalize the data to the classifier
in_encoder=Normalizer(norm='l2')
trainX=in_encoder.transform(trainX)
testX=in_encoder.transform(testX)

#print(trainX[0].shape)
#print(testX[0])
#data is normalized but still of shape (128,)

#label the encode targets
from sklearn.preprocessing import LabelEncoder
out_encoder=LabelEncoder()
print(testy)
out_encoder.fit(trainy)
out_encoder.transform(testy)

#fit the model on these normalized inputs and encoded outputs
model=SVC(kernel='linear',probability=True)
model.fit(trainX,trainy)

#test the trained model, i.e. model trained on encoded and normalized stuff on
#random example from the test dataset

selection = choice([i for i in range(testX.shape[0])])
#theres 10 test images = testX.shape[0]
#selection is helping us get a random number from 0 to 9

random_face_pixels=testX_faces[selection] #random face pixels is an image of shape 160,160,3
random_face_emb=testX[selection] #this is the corresponding normalized embedding of the random image above

random_face_class=testy[selection] #class corresponding to the random selected image
random_face_name=out_encoder.inverse_transform([random_face_class]) #getting name corresponding to the class, since class is encoded and is in numerical form

#prediction for the face

#expand_dims of the normalized face embeddings
samples=expand_dims(random_face_emb,axis=0)
yhat_class=model.predict(samples)

yhat_prob=model.predict_proba(samples)

class_index=yhat_class[0] #index from prediction
class_probability = yhat_prob[0,class_index] * 100
predict_names = out_encoder.inverse_transform(yhat_class)

print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
print('Expected: %s' % random_face_name[0])
# plot for fun
pyplot.imshow(random_face_pixels)
title = '%s (%.3f)' % (predict_names[0], class_probability)
pyplot.title(title)
pyplot.show()






