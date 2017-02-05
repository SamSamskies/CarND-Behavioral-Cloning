import os
from pathlib import Path
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D
import numpy as np
import pandas as pd
import json
import utils

IMAGE_SHAPE = (100, 320, 3)
DATA_PATH = './data/'


def save_model(model, model_filename, weights_filename):
    if Path(model_filename).is_file():
        os.remove(model_filename)

    with open(model_filename, 'w') as f:
        json.dump(model.to_json(), f)

    if Path(weights_filename).is_file():
        os.remove(weights_filename)

    model.save_weights(weights_filename)


def get_nvidia_model():
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=IMAGE_SHAPE))
    model.add(Convolution2D(24, 5, 5, init='he_normal', subsample=(2, 2), name='conv1_1'))
    model.add(Activation('relu'))
    model.add(Convolution2D(36, 5, 5, init='he_normal', subsample=(2, 2), name='conv2_1'))
    model.add(Activation('relu'))
    model.add(Convolution2D(48, 5, 5, init='he_normal', subsample=(2, 2), name='conv3_1'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, init='he_normal', subsample=(1, 1), name='conv4_1'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, init='he_normal', subsample=(1, 1), name='conv4_2'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(1164, init='he_normal', name="dense_0"))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100, init='he_normal', name="dense_1"))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, init='he_normal', name="dense_2"))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, init='he_normal', name="dense_3"))
    model.add(Activation('relu'))
    model.add(Dense(1, init='he_normal', name="dense_4"))

    model.compile(optimizer='adam', loss='mse')

    return model


def train_model(model, num_images):
    df = utils.augment_dataframe(pd.read_csv(DATA_PATH + 'driving_log.csv'))

    df_sample = df.sample(num_images)
    train_features = []
    train_labels = []

    for _, row in df_sample.iterrows():
        image_path = DATA_PATH + row.image.strip()
        train_features.append(utils.load_image(image_path, row.is_flipped))
        train_labels.append(row.steering)

    history = model.fit(np.array(train_features), np.array(train_labels), batch_size=10, nb_epoch=5, validation_split=0.2)
    save_model(model, 'model.json', 'model.h5')

    return history


model = get_nvidia_model()
num_images = 2000
train_model(model, num_images)
