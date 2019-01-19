'''
Application of Conventional Neural Networks for audio processing 

GHANMI Helmi 
22 December 2018

'''

#----------------------Import the required librairie and packages-----------------#
import numpy as np 
import pandas as pd 
import librosa 
import librosa.display 
import random
import keras 
from keras import optimizers
from keras.models import Sequential
from keras.layers import Activation, Conv2D, Dense, \
						Dense, Flatten, MaxPooling2D, Dropout

import matplotlib.pyplot as plt 

import warnings
warnings.filterwarnings('ignore')

#--------------------Read dataset : urbanSound8k-------------------------#
data = pd.read_csv('datasets/UrbanSound8K/metadata/UrbanSound8K.csv')
print(data.head(10)) # show a 5 sample of thedata file 

#-----------------Test and applyed several command to show the datasets----------#
""" Get the dataset over 3 seconds long  """
duration = [data['end']-data['start']>=3 ] # get the diration heigher than 3 S
valid_data = data[['slice_file_name', 'fold' ,'classID', 'class']][ data['end']-data['start'] >= 3 ]
print(valid_data.shape)

""" Exampele of spectrogram plot (of audio file in class of dog) """
y, sr = librosa.load('datasets/UrbanSound8K/audio/fold2/4201-3-0-0.wav', duration=2.97)
ps=librosa.feature.melspectrogram(y=y, sr=sr)
shape_of_ps=ps.shape
print('Shape of ps is:',shape_of_ps)
#display the spectrogram 
librosa.display.specshow(ps, y_axis='mel',x_axis='time')
plt.title('Example of spectrogram of one auio sample')
plt.show()

#------------------------- evalute the spectrogram using librosa for all audio datasets ---------------#
valid_data['path'] = 'fold' + valid_data['fold'].astype('str') + '/' + valid_data['slice_file_name'].astype('str')

D = [] # Dataset

for row in valid_data.itertuples():
    y, sr = librosa.load('datasets/UrbanSound8K/audio/' + row.path, duration=2.97)  
    ps = librosa.feature.melspectrogram(y=y, sr=sr)
    if ps.shape != (128, 128): continue
    D.append( (ps, row.classID) )

# number of sample 
print('The number of sample is: ', len(D))

print(D(1))

#-------------------Split dataset into train test and validate set ---------------#
#shuffle the dataset 
dataset= D
random.shuffle(dataset)

#split the datset in  train and test 
train = dataset[:7000]
test = dataset[7000:]

X_train, y_train =zip(*train)
X_test, y_test =zip(*test)

# reshape the sample for the CNN (input size is 128*128)
X_train = np.array([x.reshape( (128, 128, 1) ) for x in X_train])
X_test = np.array([x.reshape((128,128,1)) for x in X_test])

# One-Hot encoding for classes: here because we have 10 classes 
y_train = np.array(keras.utils.to_categorical(y_train, 10))
y_test = np.array(keras.utils.to_categorical(y_test, 10))


#----------------------Definethe sequential  CNN mdel---------------#
model = Sequential()
input_shape = (128,128,1)
model.add(Conv2D(24,(4,4),strides=(1,1), input_shape=input_shape))
model.add(MaxPooling2D((4,2), strides=(4,2)))
model.add(Activation('relu'))

model.add(Conv2D(48, (5, 5), padding="valid"))
model.add(MaxPooling2D((4, 2), strides=(4, 2)))
model.add(Activation('relu'))

model.add(Conv2D(48, (5, 5), padding="valid"))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dropout(rate=0.5))

model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(rate=0.5))

model.add(Dense(10))
model.add(Activation('softmax'))

#summary of the model 
model.summary()

#-------------------Compile , fit(train) adn shwo the scores of the model---------------#
#-1- compile 
model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=['accuracy'])

#-2- train 
model.fit(x=X_train,y=y_train,epochs=12, batch_size=128, validation_data=(X_test, y_test))

#-3- evaluate the accuracy 
score =model.evaluate(x=X_test,y=y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# save the model as .h5
model.save('my_trained_model.h5') 

# save the model as json file 
model_json = model.to_json()
with open("output_trained_model.json","w") as json_file:
	json_file.write(model_json)
print("Saved model json to disk")

print('Processing finished !!!')
