#Create face embeddings using the FaceNet model
#converting all faces in the dataset into embeddings

from numpy import load
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed
from keras.models import load_model


#get the face embeddings for one face
def get_embedding(model,face_pixels):
    #scale the pixels
    face_pixels=face_pixels.astype('float32')

    #standardize the pixels across channels
    mean,std=face_pixels.mean(),face_pixels.std()

    face_pixels=(face_pixels-mean)/std
    #transform the face into one sample
    samples=expand_dims(face_pixels,axis=0)

    #make prediction to get embedding
    yhat=model.predict(samples)
    return yhat[0]

#load the face dataset
data=load('5-celebrity-faces-dataset.npz')
trainX,trainy,testX,testy=data['arr_0'],data['arr_1'],data['arr_2'],data['arr_3']

print('Loaded: ',trainX.shape,trainy.shape,testX.shape,testy.shape)

#load the facenet model
model=load_model('facenet_keras.h5')
print('Loaded model')

#convert each face in the train set to get an embedding
newTrainX=list()

for face_pixels in trainX:
    embedding=get_embedding(model,face_pixels)
    newTrainX.append(embedding)

newTrainX=asarray(newTrainX)

print("newTrainX shape: ",newTrainX.shape)
print(trainy.shape)
#convert each face in the test set to get an embedding
newTestX=list()
for face_pixels in testX:
    embedding=get_embedding(model,face_pixels)
    newTestX.append(embedding)

newTestX=asarray(newTestX)

print("New test x shape: ",newTestX.shape)
print(testy.shape)

#save the arrays into one file in the compressed format
savez_compressed('5-celebrity-faces-embeddings',newTrainX,trainy,newTestX,testy)



