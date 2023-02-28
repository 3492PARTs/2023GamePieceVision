import cv2 as cv
import numpy as np
from pipeline import Pipeline
import ntcore as networktables
import os
from os.path import basename
import math
#uncomment this for the test server
#import argparse


########################### NETWORK TABLES #########################



instance = networktables.NetworkTableInstance.getDefault()

identity = f"{basename(__file__)}-{os.getpid()}"
instance.startClient4(identity)


#### TEST SERVER ####
#parser = argparse.ArgumentParser()
#parser.add_argument("ip", type=str, help="IP address to connect to")
#args = parser.parse_args()
#instance.setServer(server_name=args.ip, port=networktables.NetworkTableInstance.kDefaultPort4)
#####################

# Comment this out for the test server
#instance.setServerTeam(team=3492, port=networktables.NetworkTableInstance.kDefaultPort4)

#instance.startDSClient()
instance.setServer("10.34.92.2", networktables.NetworkTableInstance.kDefaultPort4)

table = instance.getTable("vision")
distance = table.getFloatTopic("distance")
pixels = table.getFloatTopic("pixels")
angle = table.getFloatTopic("angle")

###################################################################

# ew scientific notation
np.set_printoptions(suppress=True)

classNames = [0, 1]

height, width = 480, 640
fov = 68.5

camera = cv.VideoCapture(0)

KNOWN_VALUES = [205, 229, 2] # cube width, cone width, distance in ft
CALIBRATE_IMAGES = [cv.imread("calibratecone.PNG"), cv.imread("calibratecube.PNG")]


# Might only need calibrate here but we love redundancy :D
pipeCube = Pipeline()
pipeCone = Pipeline()
calibrateCube = Pipeline()
calibrateCone = Pipeline()

##### CALIBRATES THE FOCAL LENGTH FOR DISTANCE ESTIMATION ######
calibrateCone.process(source0=CALIBRATE_IMAGES[0], gametype=0, focalLength=None)
calibrateCube.process(source0=CALIBRATE_IMAGES[1], gametype=1, focalLength=None)

focalLengths = []

def calibrateWidthAndFocalLength(gamePieceType: int) -> None:
    calibratedWidth = 0
    if gamePieceType == 0:
        calibratedWidth = float(calibrateCone.extract_condata_0_output[4])
    else:
        calibratedWidth = float(calibrateCube.extract_condata_1_output[4])

    focalLengths.append((calibratedWidth * KNOWN_VALUES[2]) / KNOWN_VALUES[gamePieceType])

calibrateWidthAndFocalLength(0)
calibrateWidthAndFocalLength(1)
################################################################

def calculateAngle(differenceInPixels: float) -> float:
    diagonalPixels = math.sqrt(math.pow(height, 2) + math.pow(width, 2))
    degreePerPixel = fov / diagonalPixels
    angle = degreePerPixel * differenceInPixels
    return angle

def findDistanceAndPixels():
    if pipeCone.find_distance_0_output != None and pipeCube.find_distance_1_output != None:
        if pipeCone.find_distance_0_output >= pipeCube.find_distance_1_output:
            if pipeCube.extract_condata_1_output != None:
                centerw = pipeCube.extract_condata_1_output[1]
                difference_in_pix_x = centerw - 320
                table.putNumber("distance", float(pipeCube.find_distance_1_output))
                table.putNumber("pixels", float(difference_in_pix_x))
                table.putNumber("angle", float(calculateAngle(differenceInPixels=difference_in_pix_x)))

        else:
            if pipeCone.extract_condata_0_output != None and pipeCone.extract_condata_0_output[5] > pipeCone.extract_condata_0_output[4]:
                centerw = pipeCone.extract_condata_0_output[1]
                difference_in_pix_x = centerw - 320
                table.putNumber("distance", float(pipeCube.find_distance_1_output))
                table.putNumber("pixels", float(difference_in_pix_x))
                table.putNumber("angle", float(calculateAngle(differenceInPixels=difference_in_pix_x)))
    
    if pipeCube.find_distance_1_output != None and pipeCone.find_contours_0_output == None:
        if pipeCube.extract_condata_1_output != None:
            centerw = pipeCube.extract_condata_1_output[1]
            difference_in_pix_x = centerw - 320
            table.putNumber("distance", float(pipeCube.find_distance_1_output))
            table.putNumber("pixels", float(difference_in_pix_x))
            table.putNumber("angle", float(calculateAngle(differenceInPixels=difference_in_pix_x)))
    
    if pipeCone.find_distance_0_output != None and pipeCube.find_distance_1_output == None:    
        if pipeCone.extract_condata_0_output != None and pipeCone.extract_condata_0_output[5] > pipeCone.extract_condata_0_output[4]:
            centerw = pipeCone.extract_condata_0_output[1]
            difference_in_pix_x = centerw - 320
            table.putNumber("distance", float(pipeCube.find_distance_1_output))
            table.putNumber("pixels", float(difference_in_pix_x))
            table.putNumber("angle", float(calculateAngle(differenceInPixels=difference_in_pix_x)))

while True:
    ret, image = camera.read()

    ################# IMPORTANT STUFF #################
    pipeCone.process(source0=image, gametype=0, focalLength=focalLengths[0])
    pipeCube.process(source0=image, gametype=1, focalLength=focalLengths[1])

    findDistanceAndPixels()
    ###################################################

    keyboardInput = cv.waitKey(1)
    # 27 = ascii code for escape.
    if keyboardInput == 27:
        break

camera.release()
networktables.NetworkTableInstance.destroy()
cv.destroyAllWindows()
