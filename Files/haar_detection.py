# This script uses an image stored from the same working directory and extract faces 
# from the image and store it.

import cv2
import numpy as np

#To assign a particular name to person
#id = raw_input('Enter User name:')
sample = 0

# to generate classifier which classifies the frontalFace
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#eyeDetect = cv2.CascadeClassifier('haarcascade_eye.xml')
# TO capture image from webcam we need video capture object
cam = cv2.VideoCapture(0) # Most of the time 0 is working but if it doesnt then use different iDs

# Capture images one by one and detect the faces and show it in window

'''	
		 Capturing the image.. cam read will return one status variable and one captured image 
''' 	
#	ret,img = cam.read();
img = cv2.imread('IMG_7669.JPG');
#and this image is colored and for classifier to work we need grayscaled image
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY);
cv2.imshow('img',gray);	
# Now we have gray scaled image and we can detect face from that
#faces = faceDetect.detectMultiScale(gray,1.4, 4);

# faces have all the images and what we need to do is we need to put a square on each of the face
#for(x,y,w,h) in faces:
#	sample = sample+1
	# here we need to save the faces
#	cv2.imwrite("test_seperate/User."+str(sample)+".jpg", img[y:y+h, x:x+w])
#	cv2.waitKey(100)

#	cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2);
#cv2.imshow('image',img)		
#cv2.waitKey(10000)
	
#This will release the camera
cam.release()
#	THIS CODE FOR WHILE LOOP
cv2.destroyAllWindows()
	
		
	
	

