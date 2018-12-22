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
from keras.models import Sequential
from keras.layers import Activation, Conv2D, Dense, \
						Dense, Flatten, MaxPooling2D

import matplotlib.pyplot as plt 

import warnings
warnings.filterwarnings('ignore')

#--------------------Read dataset : urbanSound8k-------------------------#
data = pd.read_csv('datasets/UrbanSound8K/metadata/UrbanSound8K.csv')
print(data.head(10)) # show a 5 sample of thedata file 

#-----------------Test and applyed several command to show the datasets----------#
""" Get the datset over 3 seconds long  """
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
plt.show()

#------------------------- evalute the spectrogram using librosa for all audio datasets ---------------#

print('Processing finished !!!')
