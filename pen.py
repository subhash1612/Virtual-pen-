import cv2
import numpy as np

hsv = np.load('C:/Users/User/Desktop/Projects/hsv_value.npy')

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
kernel = np.ones((5,5),np.uint8) #for morphological operations

pad = None # pad for drawing

x1,y1 = 0,0
noiseThresh = 800

while True:
    ret,frame = cap.read()
    frame = np.flip(frame,1)
    if pad is None:
        pad = np.ones_like(frame)
    
    hsvframe = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    lower = hsv[0]
    upper = hsv[1]

    mask = cv2.inRange(hsvframe,lower,upper)
    
    #morphological transformations to get rid of noise
    mask = cv2.erode(mask,kernel,iterations = 1)
    mask = cv2.dilate(mask,kernel,iterations = 2)

    # Find Contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Making sure if contour is present and its size is greater than the noise threshold
    if contours and cv2.contourArea(max(contours, 
                                 key = cv2.contourArea)) > noiseThresh:
                
        c = max(contours, key = cv2.contourArea)    
        x2,y2,w,h = cv2.boundingRect(c)
        
        #If no detected points were found
        #When we write for the first time
        if x1 == 0 and y1 == 0:
            x1,y1= x2,y2
            
        else:
            # Draw the line on the pad
            canvas = cv2.line(pad, (x1,y1),(x2,y2), [255,0,0], 4)
        
        # The new points become the previous points after the line is drawn
        x1,y1= x2,y2
    else:
        # If there were no contours detected then make x1,y1 = 0
        x1,y1 =0,0
    
    # Merge the canvas and the frame.
    frame = cv2.add(frame,pad)
    
    # stack both frames and show it.
    stack = np.hstack((pad,frame))
    cv2.imshow('Virtual Pen',cv2.resize(stack,None,fx=0.6,fy=0.6))
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
        
    # When c is pressed clear the pad
    if k == ord('c'):
        pad = None

cv2.destroyAllWindows()
cap.release()

