from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np
from pipeline import *
#from pipeline.example import ExamplePipeline
from pipeline.dualpipeline import DualPipeline

####IMPORTANT NOTES####
#make sure to check example pipeline to see what changes that need to be made to make dual pipeline work.

#under dualpipe.process(image, {}), the {} determines what game piece you search for, 0 for cone, anything else for cube

#TO DO: CHANGE ALL PIPELINES TO JAVA, NETWORK TABLES ARE WEIRD IN PYTHON. (if it can still keep up with only using python)
#ignore that to do if pipelines are easier than I thought.
#######################




# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)

# Initiate the pipeline
#pipe = ExamplePipeline()
dualpipe = DualPipeline()

while True:
    # Grab the webcamera's image.
    ret, image = camera.read()

    # Show the image in a window
    cv2.imshow("Plain Image", image)
    
    #pipe.process(reimage)
    #print(pipe.filter_contours_0_output)
    dualpipe.process(image, 0)
    
    if dualpipe.find_contours_0_output != None:
        print(dualpipe.find_contours_0_output)

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()