from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np
from pipeline import *
#from pipeline.example import ExamplePipeline
from pipeline.dualpipeline import DualPipeline

####IMPORTANT NOTES####

#under dualpipe.process(image, {}), the {} determines what game piece you search for, 0 for cone, anything else for cube

#get good reference images :D

#KNOWN_WIDTH is currently a random number because i dont have access to a cube or cone rn

#also i know there is still the model data in the no tensor branch, i dont really care to change that. The skeleton of this program was made with tensorflow anyways.
#######################

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)

# Initiate the pipeline and calibration
dualpipecube = DualPipeline()
dualpipecone = DualPipeline()
dualcalibratecube = DualPipeline()
dualcalibratecone = DualPipeline()

# Initiate focal length and distance values
focal_lengthcube = None
focal_lengthcone = None

#width in pixels, distance in feet
KNOWN_WIDTHCUBE = 512 #########################FIND THIS!!!##########################
KNOWN_WIDTHCONE = 512 #this too 
KNOWN_DISTANCE = 2 #feet


######################### FIND FOCAL LENGTH ########################
#this sequence will run once every time the program is ran and will calibrate the focal length.
calibratecone = cv2.imread("calibration\calibratecone.PNG")
calibratecube = cv2.imread("calibration\calibratecube.PNG")

dualcalibratecube.process(source0=calibratecube, gametype = 1, focal_length=focal_lengthcube)

#the if statements are technically useless if you have good reference images, but i dont :D
#in a perfect world, focal_lengthcube and focal_lengthcone would be the same value, but sadly that doesnt happen D:
if dualcalibratecube.extract_condata_1_output != None:
    calibratewidthcube = float(dualcalibratecube.extract_condata_1_output[4])
    focal_lengthcube = (calibratewidthcube * KNOWN_DISTANCE)/KNOWN_WIDTHCUBE

dualcalibratecone.process(source0=calibratecone, gametype = 0, focal_length=focal_lengthcone)

if dualcalibratecone.extract_condata_0_output != None:
    calibratewidthcone = float(dualcalibratecone.extract_condata_0_output[4])
    focal_lengthcone = (calibratewidthcone * KNOWN_DISTANCE)/KNOWN_WIDTHCONE
###################################################################

while True:
    
    ########################## THE REAL IMPORTANT STUFF ###########################
    # Grab the webcamera's image.
    ret, image = camera.read()
    

    # Show the image in a window
    cv2.imshow("Plain Image", image)
    

    # Run the dual pipeline with both the cube and cone (gametype 0 is cone and vise versa)
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