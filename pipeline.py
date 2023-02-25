import cv2
import numpy as np
import math
from enum import Enum

class Pipeline:
    """
    An OpenCV pipeline generated by GRIP.
    """
    
    def __init__(self):
        """initializes all values to presets or None if need to be set
        """

        self.__resize_image_width = 640.0
        self.__resize_image_height = 480.0
        self.__resize_image_interpolation = cv2.INTER_CUBIC

        self.resize_image_output = None

        self.__hsv_threshold_0_input = self.resize_image_output
        self.__hsv_threshold_0_hue = [13, 58]
        self.__hsv_threshold_0_saturation = [78, 255]
        self.__hsv_threshold_0_value = [29, 255]

        self.hsv_threshold_0_output = None

        self.__hsv_threshold_1_input = self.resize_image_output
        self.__hsv_threshold_1_hue = [90.3050847457627, 155.94727592267134]
        self.__hsv_threshold_1_saturation = [45.62146892655367, 255.05448154657293]
        self.__hsv_threshold_1_value = [15.75011172432892, 254.04041188418918]

        self.hsv_threshold_1_output = None

        self.__cv_erode_0_src = self.hsv_threshold_0_output
        self.__cv_erode_0_kernel = None
        self.__cv_erode_0_anchor = (-1, -1)
        self.__cv_erode_0_iterations = 1.0
        self.__cv_erode_0_bordertype = cv2.BORDER_CONSTANT
        self.__cv_erode_0_bordervalue = (-1)

        self.cv_erode_0_output = None

        self.__cv_erode_1_src = self.hsv_threshold_1_output
        self.__cv_erode_1_kernel = None
        self.__cv_erode_1_anchor = (-1, -1)
        self.__cv_erode_1_iterations = 2.0
        self.__cv_erode_1_bordertype = cv2.BORDER_CONSTANT
        self.__cv_erode_1_bordervalue = (-1)

        self.cv_erode_1_output = None

        self.__find_contours_0_input = self.cv_erode_0_output
        self.__find_contours_0_external_only = False

        self.find_contours_0_output = None

        self.__find_contours_1_input = self.cv_erode_1_output
        self.__find_contours_1_external_only = False

        self.find_contours_1_output = None

        self.__filter_contours_0_contours = self.find_contours_0_output
        self.__filter_contours_0_min_area = 100.0
        self.__filter_contours_0_min_perimeter = 0.0
        self.__filter_contours_0_min_width = 0.0
        self.__filter_contours_0_max_width = 1000.0
        self.__filter_contours_0_min_height = 0.0
        self.__filter_contours_0_max_height = 1000.0
        self.__filter_contours_0_solidity = [0, 100]
        self.__filter_contours_0_max_vertices = 1000000.0
        self.__filter_contours_0_min_vertices = 0.0
        self.__filter_contours_0_min_ratio = 0.0
        self.__filter_contours_0_max_ratio = 1000.0

        self.filter_contours_0_output = None

        self.__filter_contours_1_contours = self.find_contours_1_output
        self.__filter_contours_1_min_area = 700.0
        self.__filter_contours_1_min_perimeter = 0
        self.__filter_contours_1_min_width = 0
        self.__filter_contours_1_max_width = 1000
        self.__filter_contours_1_min_height = 0
        self.__filter_contours_1_max_height = 1000
        self.__filter_contours_1_solidity = [0, 100]
        self.__filter_contours_1_max_vertices = 1000000
        self.__filter_contours_1_min_vertices = 0
        self.__filter_contours_1_min_ratio = 0
        self.__filter_contours_1_max_ratio = 1000

        self.filter_contours_1_output = None

        self.extract_condata_0_output = None
        self.extract_condata_1_output = None
        self.find_distance_1_output = None
        self.find_distance_0_output = None

        ###################################
        #distance is in feet, width in pixels
        self.known_widthcube = 205 ##################FIND THIS!##################
        self.known_widthcone = 229
        ###################################


    def process(self, source0, gametype, focalLength):
        """
        Runs the pipeline and sets all outputs to new values.
        """
        
        self.__resize_image_input = source0
        (self.resize_image_output) = self.__resize_image(self.__resize_image_input, self.__resize_image_width, self.__resize_image_height, self.__resize_image_interpolation)

        if gametype == 0:
            # Step HSV_Threshold0:
            self.__hsv_threshold_0_input = self.resize_image_output
            (self.hsv_threshold_0_output) = self.__hsv_threshold(self.__hsv_threshold_0_input, self.__hsv_threshold_0_hue, self.__hsv_threshold_0_saturation, self.__hsv_threshold_0_value)

            # Step CV_erode0:
            self.__cv_erode_0_src = self.hsv_threshold_0_output
            (self.cv_erode_0_output) = self.__cv_erode(self.__cv_erode_0_src, self.__cv_erode_0_kernel, self.__cv_erode_0_anchor, self.__cv_erode_0_iterations, self.__cv_erode_0_bordertype, self.__cv_erode_0_bordervalue)

            # Step Find_Contours0:
            self.__find_contours_0_input = self.cv_erode_0_output
            (self.find_contours_0_output) = self.__find_contours(self.__find_contours_0_input, self.__find_contours_0_external_only)
            
            # Step Filter_Contours0:
            self.__filter_contours_0_contours = self.find_contours_0_output
            (self.filter_contours_0_output) = self.__filter_contours(self.__filter_contours_0_contours, self.__filter_contours_0_min_area, self.__filter_contours_0_min_perimeter, self.__filter_contours_0_min_width, self.__filter_contours_0_max_width, self.__filter_contours_0_min_height, self.__filter_contours_0_max_height, self.__filter_contours_0_solidity, self.__filter_contours_0_max_vertices, self.__filter_contours_0_min_vertices, self.__filter_contours_0_min_ratio, self.__filter_contours_0_max_ratio)
        
            # Step Extract_ConData0:
            self.__extract_condata_0_input = self.filter_contours_0_output
            (self.extract_condata_0_output) = self.__extract_condata(self.__extract_condata_0_input, self.__find_contours_0_input)

            # Step Find_Distance0:
            if focalLength != None:
                self.__find_distance_0_input = self.extract_condata_0_output
                if self.__find_distance_0_input != None:
                    (self.find_distance_0_output) = self.__find_distance(self.known_widthcone, focalLength, self.__find_distance_0_input[4])

        else:
            # Step HSV_Threshold1:
            self.__hsv_threshold_1_input = self.resize_image_output
            (self.hsv_threshold_1_output) = self.__hsv_threshold(self.__hsv_threshold_1_input, self.__hsv_threshold_1_hue, self.__hsv_threshold_1_saturation, self.__hsv_threshold_1_value)

            # Step CV_erode1:
            self.__cv_erode_1_src = self.hsv_threshold_1_output
            (self.cv_erode_1_output) = self.__cv_erode(self.__cv_erode_1_src, self.__cv_erode_1_kernel, self.__cv_erode_1_anchor, self.__cv_erode_1_iterations, self.__cv_erode_1_bordertype, self.__cv_erode_1_bordervalue)

            # Step Find_Contours1:
            self.__find_contours_1_input = self.cv_erode_1_output
            (self.find_contours_1_output) = self.__find_contours(self.__find_contours_1_input, self.__find_contours_1_external_only)

            # Step Filter_Contours1:
            self.__filter_contours_1_contours = self.find_contours_1_output
            (self.filter_contours_1_output) = self.__filter_contours(self.__filter_contours_1_contours, self.__filter_contours_1_min_area, self.__filter_contours_1_min_perimeter, self.__filter_contours_1_min_width, self.__filter_contours_1_max_width, self.__filter_contours_1_min_height, self.__filter_contours_1_max_height, self.__filter_contours_1_solidity, self.__filter_contours_1_max_vertices, self.__filter_contours_1_min_vertices, self.__filter_contours_1_min_ratio, self.__filter_contours_1_max_ratio)

            # Step Extract_ConData1:
            self.__extract_condata_1_input = self.filter_contours_1_output
            (self.extract_condata_1_output) = self.__extract_condata(self.__extract_condata_1_input, self.__find_contours_1_input)
            
            # Step Find_Distance1:
            if focalLength != None:
                self.__find_distance_1_input = self.extract_condata_1_output
                if self.__find_distance_1_input != None:
                    (self.find_distance_1_output) = self.__find_distance(self.known_widthcube, focalLength, self.__find_distance_1_input[4])

            


    @staticmethod
    def __resize_image(input, width, height, interpolation):
        """Scales and image to an exact size.
        Args:
            input: A numpy.ndarray.
            Width: The desired width in pixels.
            Height: The desired height in pixels.
            interpolation: Opencv enum for the type fo interpolation.
        Returns:
            A numpy.ndarray of the new size.
        """
        return cv2.resize(input, ((int)(width), (int)(height)), 0, 0, interpolation)

    @staticmethod
    def __hsv_threshold(input, hue, sat, val):
        """Segment an image based on hue, saturation, and value ranges.
        Args:
            input: A BGR numpy.ndarray.
            hue: A list of two numbers the are the min and max hue.
            sat: A list of two numbers the are the min and max saturation.
            lum: A list of two numbers the are the min and max value.
        Returns:
            A black and white numpy.ndarray.
        """
        out = cv2.cvtColor(input, cv2.COLOR_BGR2HSV)
        return cv2.inRange(out, (hue[0], sat[0], val[0]),  (hue[1], sat[1], val[1]))

    @staticmethod
    def __cv_erode(src, kernel, anchor, iterations, border_type, border_value):
        """Expands area of lower value in an image.
        Args:
           src: A numpy.ndarray.
           kernel: The kernel for erosion. A numpy.ndarray.
           iterations: the number of times to erode.
           border_type: Opencv enum that represents a border type.
           border_value: value to be used for a constant border.
        Returns:
            A numpy.ndarray after erosion.
        """
        return cv2.erode(src, kernel, anchor, iterations = (int) (iterations +0.5),
                            borderType = border_type, borderValue = border_value)

    @staticmethod
    def __find_contours(input, external_only):
        """Sets the values of pixels in a binary image to their distance to the nearest black pixel.
        Args:
            input: A numpy.ndarray.
            external_only: A boolean. If true only external contours are found.
        Return:
            A list of numpy.ndarray where each one represents a contour.
        """
        if(external_only):
            mode = cv2.RETR_EXTERNAL
        else:
            mode = cv2.RETR_LIST
        method = cv2.CHAIN_APPROX_SIMPLE
        contours, hierarchy =cv2.findContours(input, mode=mode, method=method)
        
        if contours:

            return contours

    @staticmethod
    def __filter_contours(input_contours, min_area, min_perimeter, min_width, max_width,
                        min_height, max_height, solidity, max_vertex_count, min_vertex_count,
                        min_ratio, max_ratio):
        """Filters out contours that do not meet certain criteria.
        Args:
            input_contours: Contours as a list of numpy.ndarray.
            min_area: The minimum area of a contour that will be kept.
            min_perimeter: The minimum perimeter of a contour that will be kept.
            min_width: Minimum width of a contour.
            max_width: MaxWidth maximum width.
            min_height: Minimum height.
            max_height: Maximimum height.
            solidity: The minimum and maximum solidity of a contour.
            min_vertex_count: Minimum vertex Count of the contours.
            max_vertex_count: Maximum vertex Count.
            min_ratio: Minimum ratio of width to height.
            max_ratio: Maximum ratio of width to height.
        Returns:
            Contours as a list of numpy.ndarray.
        """
        
        output = []
        if input_contours != None:
            for contour in input_contours:
                x,y,w,h = cv2.boundingRect(contour)
                if (w < min_width or w > max_width):
                    continue
                if (h < min_height or h > max_height):
                    continue
                area = cv2.contourArea(contour)
                if (area < min_area):
                    continue
                if (cv2.arcLength(contour, True) < min_perimeter):
                    continue
                hull = cv2.convexHull(contour)
                solid = 100 * area / cv2.contourArea(hull)
                if (solid < solidity[0] or solid > solidity[1]):
                    continue
                if (len(contour) < min_vertex_count or len(contour) > max_vertex_count):
                    continue
                ratio = (float)(w) / h
                if (ratio < min_ratio or ratio > max_ratio):
                    continue
                output.append(contour)
        return output

    @staticmethod
    def __extract_condata(filtered_contours, hsv_cam):
            
        if filtered_contours:
            c = max(filtered_contours, key=cv2.contourArea)

            #creates a bounding rect, finds the center of the rect for rotation
            
            x, y, w, h = cv2.boundingRect(c)
            centerw = x+(w / 2)
            centerh = y+(h / 2)
            area = w*h

            #draws a rectangle onto "input", where (x,y) are one vertice, and (x+width, y+height) is the opposite one.
            cv2.rectangle(hsv_cam,(x,y),(x+w,y+h),(135,50,30),3)
            cv2.circle(hsv_cam, (int(centerw),int(centerh)),5,(135,50,30),-1)

            #return "centerh = " + str(centerh), "centerw = " + str(centerw), "x coord = " + str(x), "y coord = " + str(y), "rect width = " + str(w), "rect height = " + str(h), "rect area = " + str(area)
            return centerh, centerw, x, y, int(w), int(h), int(area)
    
    @staticmethod
    def __find_distance(KNOWN_WIDTH, focalLength, new_width):
        return (KNOWN_WIDTH * focalLength) / new_width