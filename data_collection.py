import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
from cvzone.ClassificationModule import Classifier




# Initialize video capture
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1) # to detect 1 hand at max
offset=20
imgSize=300
folder="Data/facebook"
counter=0


# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video device.")
    exit()

paused = False  # Flag to track the paused state

while True:
    if not paused:
        success, img = cap.read()
        hands,img=detector.findHands(img)# to detect hands from the image
        if hands:
            hand = hands[0] # for 1 hand only
            x,y,w,h= hand['bbox']  # here "bbox" stands for bounding box -->just to crop the hand portion only
            imgWhite=np.ones((imgSize,imgSize,3),np.uint8)*255 #to create matrix of ones

            imgCrop=img[y-offset:y+h+offset,x-offset:x+w+offset]  #y-> starting height   y+h-> ending height  x->starting width  x+w->ending width   to exactly give the bounding box that we required
                                                                   # to give the appropriate cropped image for the classifier
            
            #imgCropShape =imgCrop.shape
            #imgWhite[0:imgCropShape[0],0:imgCropShape[1]]=imgCrop # to overlay the cropped image from imgCrop over imgWhite
            
            aspectRatio =h/w
            if aspectRatio>1:
                k=imgSize/h  
                wCal=math.ceil(k*w)  #this done for compenstaing the image size in width if height is changing variably
                imgResize=cv2.resize(imgCrop,(wCal,imgSize)) # here height is fixed but the width will be calculated
                imgResizeShape =imgResize.shape
                wGap=math.ceil((imgSize-wCal)/2)#this is the gap we need to push forward to centre the image
                
                #imgWhite[0:imgResizeShape[0],0:imgResizeShape[1]]=imgResize
                imgWhite[:,wGap:wCal+wGap]=imgResize
                

            
            else:
                k=imgSize/w  
                hCal=math.ceil(k*h)  
                imgResize=cv2.resize(imgCrop,(imgSize,hCal))# here the width is fixed and the height will be calculated
                imgResizeShape =imgResize.shape
                hGap=math.ceil((imgSize-hCal)/2)
                imgWhite[hGap:hCal+hGap,:]=imgResize




            cv2.imshow("ImageCrop",imgCrop)
            cv2.imshow("ImageWhite",imgWhite)
        if not success:
            print("Error: Failed to capture image.")
            break

        cv2.imshow("Image", img)
        key=cv2.waitKey(1)
        if key==ord("s"):   # button pressed to save the images
            counter +=1
            cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
            print(counter)  # to know how many images we have saved to have some consisitency

    key1 = cv2.waitKey(1) & 0xFF

    # If 'p' key is pressed, toggle the paused state
    if key1 == ord('p'):
        paused = not paused

    # If 'q' key is pressed, exit the loop
    if key1 == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
