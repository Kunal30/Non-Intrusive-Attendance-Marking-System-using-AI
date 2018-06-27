import numpy as np
import cv2

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades
def frame_detect(name):
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
#eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

#cap = cv2.VideoCapture(0)


	img = cv2.imread(name,cv2.IMREAD_UNCHANGED)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(img, 1.1 , 5)
        count=0
	for (x,y,w,h) in faces:
	    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
	    roi_gray = gray[y:y+h, x:x+w]
	    roi_color = img[y:y+h, x:x+w]
            sub_face = img[y:y+h, x:x+w]
            FaceFileName = "cropped/det_face%d.jpg" %count
            cv2.imwrite(FaceFileName, sub_face)
	    count+=1
 #       eyes = eye_cascade.detectMultiScale(roi_gray)
  #      for (ex,ey,ew,eh) in eyes:
   #         cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
	   # cv2.imwrite('cropped/det_frame%d.jpg' %count,img)	
	    #cv2.imshow('img',img)
	    k = cv2.waitKey(5) & 0xff
	    if k == 27:
	       break

	    #cv2.imwrite('/home/maulik/RI/detected_frames/detected%s.jpg' % name,img)
#cap.release()
	cv2.waitKey(5000)
	cv2.destroyAllWindows()
	return


frame_detect('/home/kunal/Desktop/RI/IMG_7669.JPG')

	
