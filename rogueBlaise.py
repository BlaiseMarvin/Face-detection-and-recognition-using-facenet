# extract the faces from folder

from os import listdir
from PIL import Image
from numpy import asarray
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN

#extract a single face from a given photograph

def extract_face(filename,required_size=(160,160)):
    #load image from file
    image=Image.open(filename)
    #convert the image to RGB
    image=image.convert('RGB')
    #convert the image to an array
    pixels=asarray(image)

    #create a detector using default weights
    detector=MTCNN()

    #detect faces in the image
    results=detector.detect_faces(pixels)

    #extract the bounding box from the first face
    x1,y1,width,height=results[0]['box']

    #bug fix
    x1,y1=abs(x1),abs(y1)
    x2,y2=x1+width,y1+height

    #extract the face
    face=pixels[y1:y2,x1:x2]

    #resize the pixels to the model size
    image=Image.fromarray(face)
    image=image.resize(required_size)
    face_array=asarray(image)

    return face_array


folder="E:/FaceRecognitionFaceNet2.0/train/blaise/"

i=1
#enumerate files
for filename in listdir(folder):
    #path
    path=folder+filename
    #get face
    face=extract_face(path)
    print(i,face.shape)

    #plot
    pyplot.subplot(4,6,i)
    pyplot.axis('off')
    pyplot.imshow(face)

    i+=1

pyplot.show()