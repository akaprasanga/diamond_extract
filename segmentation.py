import numpy as np
import cv2
import os, os.path
import sys




def process(name):
    folder = name.split('/')
    file_name = folder[len(folder)-1]
    img = cv2.imread(name)

    original_image = img

    ############# EDGEEEEEEEEEE AND DILATION#################

    def edge_dilate(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
        kernel = np.ones((15, 15), np.uint8)

        opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)  # Open (erode, then dilate)
        edges = cv2.Canny(opening, 50, 150, apertureSize=3)  # Canny edge detection

        kernel = np.ones((3,3),np.uint8)
        dilate = cv2.dilate(edges,kernel,iterations=10)
        return dilate

    intermediate1 = edge_dilate(img)
    cv2.imwrite('Edged/Edge_'+file_name,intermediate1)




    def closing(image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        kernel = np.ones((25,25),np.uint8)
        sensitivity = 18
        lower_white = np.array([0,0,255-sensitivity])
        upper_white = np.array([255,sensitivity,255])
        mask = cv2.inRange(hsv, lower_white, upper_white)
        closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return closing


    image = cv2.imread('Edged/Edge_'+file_name)
    intermediate2 = closing(image)
    cv2.imwrite('Closing/Closing_'+file_name,intermediate2)


    def counteranalysis(preprocessed,originallll,name):
        img =preprocessed
        original= originallll
        # imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(img, 40, 255, 0)
        _,contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


        # Find the index of the largest make contour
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt=contours[max_index]


        (x,y),r = cv2.minEnclosingCircle(cnt)
        print(x,y,r)
        x_r,y_r,w,h = cv2.boundingRect(cnt)


        cX = int(x)
        cY = int(y)

        ########### tuning for better circle with the multiplier of W
        r = int(w/2-w*0.005)
        print(cX,cY,r)
        cv2.circle(img, (cX,cY), r, (40, 255, 0), -1)
        # cv2.imshow('green',img)
        # cv2.waitKey(0)

        for i in range(original.shape[0]):
            for j in range(original.shape[1]):
                if img[i,j]==40:
                    continue

                else:
                    original[i,j][0] =0
                    original[i,j][1] =0
                    original[i,j][2] =0
        return(original)

    trying = cv2.imread('Closing/Closing_'+file_name,0)


    final = counteranalysis(trying,original_image,name)
    

    cv2.imwrite('Output/'+file_name,final)

# path = "/media/prasanga/New Volume/raj/images"


first_arg = sys.argv[1]
print(first_arg)
first_arg= first_arg+'/images'
path = first_arg


counter = 0
for root, dirs, files in os.walk(path):  
    for filename in files:
        print("Processing Image =",counter)
        fname=root+'/'+filename
        counter +=1
        process(fname)



