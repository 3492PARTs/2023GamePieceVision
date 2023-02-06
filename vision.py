from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np
from pipeline import *
from pipeline.example import ExamplePipeline

####IMPORTANT NOTES####
#index of 0 = shape
#index of 1 = notshape
#using index for if statements is much easier than using class_name[] or class_names[]
#as class_names[] has both a number and a name while class_name[] is just weird. 
#######################




# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("modeldata\keras_model.h5", compile=False)

# Load the labels
class_names = open("modeldata\labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)

while True:
    # Grab the webcamera's image.
    ret, image = camera.read()

    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Show the image in a window
    cv2.imshow("Webcam Image", image)

    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predicts the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Testing class_name usage
    if index == 1: #nonshape detected
        ExamplePipeline.calledexample(camera)
    elif index == 0: #shape destected
        print("sadge", end="\n")
    else: #error occured
        print("prediction or camera failed.", end="\n")

    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()