# import sys
import librosa
import numpy as np

# import os, re
from os import listdir
# from os.path import isfile, join
import keras
import tensorflow as tf
from keras.models import load_model
from keras.layers import Input

def load_DLmodel(file_name):
    id = 1  # Song ID
    X_user = np.empty((0, 128, 130))
    path = f'./userfile/{file_name}/3s/' 
    file_data = [f for f in listdir(path)]

    for line in file_data:
        y, sr = librosa.load(path + line, sr=22050, duration=60) # 用 librosa讀取檔案
        S = np.abs(librosa.stft(y)) # 傅立葉轉換取振幅
        melspectrogram = librosa.feature.melspectrogram(S=S, sr=sr)
        melspectrogram = np.expand_dims(melspectrogram, axis = 0)
        X_user = np.append(X_user, melspectrogram, axis=0)
        id = id+1

    X_user = X_user.swapaxes(1,2)
    model = tf.keras.models.load_model('./recognition/crnnmodel.h5')
    n_features = X_user.shape[2]
    input_shape = (None, n_features)
    model_input = Input(input_shape, name='input')
    predict = model.predict(X_user).argmax(axis=1)
    dict_genres = {'blues':0, 'classical':1, 'country':2, 'disco':3, 
                  'hiphop':4,'jazz':5, 'metal' :6, 'pop': 7 ,'reggae': 8 ,'rock':9}
    ans = np.argmax(np.bincount(predict))
    arr_genres = np.array(list(dict_genres.keys()))
    genre = arr_genres[ans]
    
    return genre


# file_name = sys.argv[1]    
# load_DLmodel(file_name)