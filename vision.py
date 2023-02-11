from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np
from pipeline import *
#from pipeline.example import ExamplePipeline
from pipeline.dualpipeline import DualPipeline

####IMPORTANT NOTES####

#under dualpipe.process(image, {}), the {} determines what game piece you search for, 0 for cone, anything else for cube

#######################




# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)

# Initiate the pipeline
#pipe = ExamplePipeline()
dualpipecube = DualPipeline()
dualpipecone = DualPipeline()

while True:
    
    ########################## THE REAL IMPORTANT STUFF ###########################
    # Grab the webcamera's image.
    ret, image = camera.read()

    # Show the image in a window
    cv2.imshow("Plain Image", image)
    
    dualpipecone.process(image, 0)
    dualpipecube.process(image, 1)

    #if dualpipecone.extract_condata_0_output[6] != None & dualpipecube.extract_condata_1_output[6] != None & dualpipecone.extract_condata_0_output[6] <= dualpipecube.extract_condata_1_output[6]:
    if dualpipecube.extract_condata_1_output != None:
        print(dualpipecube.extract_condata_1_output)
    #elif dualpipecone.extract_condata_0_output[6] != None & dualpipecube.extract_condata_1_output[6] != None & dualpipecone.extract_condata_0_output[6] > dualpipecube.extract_condata_1_output[6]:
    if dualpipecone.extract_condata_0_output != None:    
        print(dualpipecone.extract_condata_0_output)
    ###############################################################################

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()