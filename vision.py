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
dualcalibratecube = DualPipeline()
dualcalibratecone = DualPipeline()

focal_lengthcube = None
focal_lengthcone = None

KNOWN_WIDTHCUBE = 512 #########################FIND THIS!!!##########################
KNOWN_WIDTHCONE = 512 #this too 
KNOWN_DISTANCE = 2



calibratecone = cv2.imread("calibration\calibratecone.PNG")
calibratecube = cv2.imread("calibration\calibratecube.PNG")

######################### FIND FOCAL LENGTH ########################
dualcalibratecube.process(source0=calibratecube, gametype = 1, focal_length=focal_lengthcube)

if dualcalibratecube.extract_condata_1_output != None:
    calibratewidthcube = float(dualcalibratecube.extract_condata_1_output[4])
    focal_lengthcube = (calibratewidthcube * KNOWN_DISTANCE)/KNOWN_WIDTHCUBE

dualcalibratecone.process(source0=calibratecone, gametype = 0, focal_length=focal_lengthcone)

if dualcalibratecone.extract_condata_0_output != None:
    calibratewidthcone = float(dualcalibratecone.extract_condata_0_output[4])
    focal_lengthcone = (calibratewidthcone * KNOWN_DISTANCE)/KNOWN_WIDTHCONE
###################################################################

########
while True:
    
    ########################## THE REAL IMPORTANT STUFF ###########################
    # Grab the webcamera's image.
    ret, image = camera.read()
    

    # Show the image in a window
    cv2.imshow("Plain Image", image)
    

    dualpipecone.process(source0=image, gametype=0, focal_length=focal_lengthcone)
    dualpipecube.process(source0=image, gametype=1, focal_length=focal_lengthcube)

    #if dualpipecone.extract_condata_0_output[6] != None & dualpipecube.extract_condata_1_output[6] != None & dualpipecone.extract_condata_0_output[6] <= dualpipecube.extract_condata_1_output[6]:
    if dualpipecube.find_distance_1_output != None:
        print(dualpipecube.find_distance_1_output)
    #elif dualpipecone.extract_condata_0_output[6] != None & dualpipecube.extract_condata_1_output[6] != None & dualpipecone.extract_condata_0_output[6] > dualpipecube.extract_condata_1_output[6]:
    if dualpipecone.find_distance_0_output != None:    
        print(dualpipecone.find_distance_0_output)
    ###############################################################################

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()