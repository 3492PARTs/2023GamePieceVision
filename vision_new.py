import cv2 as cv
import numpy as np
from pipeline import Pipeline
import ntcore as networktables
import os
from os.path import basename
import math


########################### NETWORK TABLES #########################

instance = networktables.NetworkTableInstance.getDefault()

identity = f"{basename(__file__)}-{os.getpid()}"
instance.startClient4(identity)

instance.setServer("10.34.92.2", networktables.NetworkTableInstance.kDefaultPort4)

table = instance.getTable("vision")
distance = table.getFloatTopic("distance")
pixels = table.getFloatTopic("pixels")
angle = table.getFloatTopic("angle")

###################################################################

# ew scientific notation
np.set_printoptions(suppress=True)

height, width = 720, 1280 #pixels
horzAngle = 60 #degrees

camera = cv.VideoCapture(0)

KNOWN_VALUES = [205, 2] # cube width, distance in ft ##MIGHT NEED RECALIBRATING##
CALIBRATE_IMAGE = cv.imread("calibratecube.PNG")


# Might only need calibrate here but we love redundancy :D
pipeCube = Pipeline()
calibrateCube = Pipeline()

##### CALIBRATES THE FOCAL LENGTH FOR DISTANCE ESTIMATION ######
calibrateCube.process(source0=CALIBRATE_IMAGE, focalLength=None)


calibratedWidth = float(calibrateCube.extract_condata_1_output[4])
focalLength = (calibratedWidth * KNOWN_VALUES[1]) / KNOWN_VALUES[0]


################################################################

def calculateAngle(differenceInPixels: float) -> float:
    horizontalPixels = horzAngle / width
    angle = horizontalPixels * differenceInPixels
    print(angle)
    return angle

def findDistanceAndPixels():
    
    if pipeCube.find_distance_1_output != None and pipeCube.extract_condata_1_output != None:
        centerw = pipeCube.extract_condata_1_output[1]
        difference_in_pix_x = centerw - 640
        table.putNumber("distance", float(pipeCube.find_distance_1_output))
        table.putNumber("pixels", float(difference_in_pix_x))
        table.putNumber("angle", float(calculateAngle(differenceInPixels=difference_in_pix_x)))
    

while True:
    ret, image = camera.read()

    ################# IMPORTANT STUFF #################
    pipeCube.process(source0=image, focalLength=focalLength)

    findDistanceAndPixels()
    ###################################################

    keyboardInput = cv.waitKey(1)
    # 27 = ascii code for escape.
    if keyboardInput == 27:
        break

camera.release()
networktables.NetworkTableInstance.destroy()
cv.destroyAllWindows()
