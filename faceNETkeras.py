from keras.models import load_model

#load the model
model=load_model('facenet_keras.h5')

#summarize the input and output shapes
print(model.inputs)
print(model.outputs)

#The model henceforth expects a colored image of shape 160*160*3 and outputs a 128 face embedding
# function for face detection with mtcnn
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN

# extract a single face from a given photograph

# function for face detection with mtcnn
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN




#The function below inputs an image, it expects a single face in the image. After this it extracts this isngle face fro
def extract_face(filename, required_size=(160, 160)):

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
	x1, y1, width, height = results[0]['box']
	# bug fix
	x1, y1 = abs(x1), abs(y1)
	x2, y2 = x1 + width, y1 + height
	# extract the face
	face = pixels[y1:y2, x1:x2]
	# resize pixels to the model size
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array

# load the photo and extract the face
