import numpy as np
import cv2

cap = cv2.VideoCapture('D:\\HW2\\test1.avi')

while(True):
    ret, frame = cap.read()
    if ret == False:
        break
    cv2.imshow('frame', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):  
        break 
    

cap.release()