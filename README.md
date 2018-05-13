# Freesound

* Step1:
Download data files from following URL and store it under data/audio_train/

https://www.kaggle.com/c/freesound-audio-tagging/data


* Step2:
Set full path of the folder to variable home

home='/Users/knnatarasan/workspace/ds/udacity/freesound/'

* Step3:

This project is tested on Python3 , make sure that follwoing packages are available 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa,librosa.display,pyaudio,wave,time

import os,shutil
from scipy.io import wavfile
import IPython.display as ipd

from keras import losses,models,optimizers
from keras.activations import relu,softmax
from keras.callbacks import(EarlyStopping,LearningRateScheduler,ModelCheckpoint,
                           TensorBoard,ReduceLROnPlateau)
from keras.layers import (Convolution1D,Dense,Dropout,GlobalAveragePooling1D,
                         GlobalMaxPool1D,Input,MaxPool1D,concatenate)
from keras.utils import Sequence,to_categorical
import ml_metrics as metrics