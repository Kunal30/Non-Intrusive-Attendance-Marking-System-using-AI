import cv2
import os

image_labels=[]
def loadImages(folder):
    images = []

    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
	path=os.path.join(folder,filename).split(".")
	image_labels.append(path[1])
	

        if img is not None:
            images.append(img)
    return images
path="/home/maulik/RI/dataSet"

# your images in an array
imgs = loadImages(path)
data=np.array(imgs)
print(data.shape)

