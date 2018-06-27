import cv2
import math

print(cv2.__version__)
vidcap = cv2.VideoCapture('MVI_7668.MOV')
success,image = vidcap.read()
count = 0
success = True
while success:
	success,image = vidcap.read()
	print 'Read a new frame: ', success
	if (count % 5 == 0):	
		cv2.imwrite("/home/kunal/Desktop/RI/frames/frame%d.jpg" % count, image)     # save frame as JPEG file
	count += 1
#frameRate = cap.get(5) #frame rate
#while(cap.isOpened()):
#    frameId = cap.get(1) #current frame number
#    ret, frame = cap.read()
#    if (ret != True):
#        break
#    if (frameId % math.floor(frameRate) == 0):
#        filename = "/home/maulik/RI/frames/" +  str(int(frameId)) + ".jpg"
#        cv2.imwrite(filename, frame)
