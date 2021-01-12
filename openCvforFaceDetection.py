import cv2

pixels=cv2.imread('test1.jpg')

#Load the pretrained model
classifier=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Perform face detection
bboxes=classifier.detectMultiScale(pixels)

#Print bounding boxes for each detected face
for box in bboxes:
    print(box)

#the returned coordinates by detectMultiScale function basically return the x and y coordinates for the bottom left hand corner of the bounding box as well as the height and width

#We can update this example to plot the photograph and draw each bounding box
#This can be achieved by drawing a rectangle for each box directly over the pixels of the loaded image using the rectangle function



