#!/usr/bin/env python3
import numpy as np
from alive_progress import alive_bar
from sklearn.model_selection import train_test_split
from os import listdir
from os.path import isfile
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPool2D
from keras.models import Model, load_model

def load_data(root: str) -> tuple:
    labels = []
    data = None
    files = listdir(root)
    with alive_bar(len(files), dual_line = True, title = 'Loading Data') as bar:
        for dir in files:
            label = dir.split('.')[0]
            bar.title = '-> Loading Class %s' % label.capitalize()
            temp_data = np.load(root+dir) 
            labels += [label.replace(' ', '_')] * temp_data.shape[0]
            data = temp_data if data is None else np.append(data, temp_data, 0)
            bar()
    # The original data are 28, 28 grayscale bitmaps
    return data.reshape(-1,28,28,1).astype('float') / np.max(data), labels

def get_data() -> Model:
    if isfile('models/autoencoder'):
        return load_model('models/autoencoder')

    train_data, test_data, train_labels, test_labels = train_test_split(*load_data('data/'), test_size=0.2)
    
    input = Input(shape=(28, 28, 1))
    layer = Conv2D(32, (4, 4), activation='relu', strides=(2, 2), padding='same')(input)
    layer = Conv2D(3, (2, 2), activation='sigmoid', strides=(2,2), padding='same')(layer)
    layer = Conv2DTranspose(32, (2, 2), activation='relu', strides=(2,2), padding='same')(layer)
    layer = Conv2DTranspose(1, (4, 4), activation='sigmoid', strides=(2,2), padding='same')(layer)
    model = Model(input, layer)
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy')

    model.fit(train_data, train_data, epochs = 50, batch_size = 128, shuffle = True, validation_data = (test_data, test_data))
    model.save('models/autoencoder')
    return model

if __name__ == '__main__':
    get_data()