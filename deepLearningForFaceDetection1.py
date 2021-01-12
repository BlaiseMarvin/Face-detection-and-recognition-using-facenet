from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
from matplotlib.patches import Rectangle

#Draw an image with detected object results
def draw_image_with_boxes(filename,result_list):
    #load the image
    data=pyplot.imread(filename)

    #plot the image
    pyplot.imshow(data)

    #get the context for drawing boxes
    ax=pyplot.gca()

    #plot each box
    for result in result_list:
        # get coordinates
        x,y,width,height=result['box']

        #create the shape
        rect=Rectangle((x,y),width,height,fill=False,color='red')

        #draw the box
        ax.add_patch(rect)

    #show the plot
    pyplot.show()


filename='test1.jpg'

#load the image from the file
pixels=pyplot.imread(filename)

detector=MTCNN()

faces=detector.detect_faces(pixels)

draw_image_with_boxes(filename,faces)

