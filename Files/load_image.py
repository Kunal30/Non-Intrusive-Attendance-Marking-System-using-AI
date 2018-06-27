import cv2
import os
import numpy as np
from skimage import io
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras import backend as K
import os
from matplotlib import pyplot as plt
import operator

from PIL import Image



def set_keras_backend(backend):

    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        reload(K)
        assert K.backend() == backend

set_keras_backend("theano")
imageLabels=[]
size=25,25
def loadImages(folder):
    images = []

    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename),0)
	img=cv2.resize(img, (25, 25)) 
#	cv2.imshow('img',img)
#	cv2.waitKey(1000)
#	cv2.destroyAllWindows()

        if img is not None:
	    
            


            images.append(img)
	    path=os.path.join(folder,filename).split(".")
	    imageLabels.append(path[1])
	    #print(img.shape)
	    
	
    return images
path="/home/maulik/RI/faces"
imageDataFin = loadImages(path)
#print(imageData.shape)
c = np.array(imageDataFin)

X_train, X_test, y_train, y_test = train_test_split(np.array(imageDataFin),np.array(imageLabels), train_size=0.7, random_state = 20)

X_train = np.array(X_train)
X_test = np.array(X_test)

#plt.imshow(X_test[0], interpolation='nearest')
#plt.show()

#img = Image.fromarray(X_test[0], 'RGB')
#	img=cv2.imread(X_test[i])
#cv2.imshow('img',img)
#cv2.waitkey(2000) 
#cv2.destroyAllWindows()
print(X_train.shape)

print(X_test.shape)

nb_classes = 20
y_train = np.array(y_train) 
y_test = np.array(y_test)

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

X_train = X_train.reshape(1255, 25*25)
X_test = X_test.reshape(539, 25*25)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

print("Training matrix shape", X_train.shape)
print("Testing matrix shape", X_test.shape)

model = Sequential()#probability=True)
model.add(Dense(512,input_shape=(X_train.shape[1],)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=20, nb_epoch=10, verbose=1, validation_data=(X_test, Y_test))
#print(model.predict(X_test))
#print(X_test.shape)

predicted_classes = model.predict_classes(X_test)

#for i in range(len(Y_test)):
#	index, value = max(enumerate(Y_test[i]), key=operator.itemgetter(1))
#	print(index,predicted_classes[i])
	#if predicted_classes[i]!=index:
	#	print(index)
#for i in Y_test:
#	print i
#print( model.predict_proba(X_test))
#print(np.nonzero(predicted_classes == y_test)[0])
#print(np.nonzero(predicted_classes != y_test)[0])
#print(predicted_classes)

#print(prediction.shape)
prediction=model.predict(X_test)

#for i in range(len(prediction)):
	
#	print(i)
#	print(max(prediction[i]))
	#print(prediction.index(max(prediction)))

loss, accuracy = model.evaluate(X_test,Y_test, verbose=0)

print(loss)
print(accuracy)







