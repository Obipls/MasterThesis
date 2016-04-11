# !/usr/bin/env python3.5
# coding=utf-8
from keras.models import Graph, Sequential
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
import numpy as np

def LSTMcomp(docTuple):
    data_dim = 16
    timesteps = 8
    nb_classes = 10

    encoder = Sequential()
    encoder.add(LSTM(32, input_shape=(timesteps, data_dim)))

    model = Graph()
    model.add_input(name='input_a', input_shape=(timesteps, data_dim))
    model.add_input(name='input_b', input_shape=(timesteps, data_dim))
    model.add_shared_node(encoder, name='shared_encoder', inputs=['input_a', 'input_b'],
                          merge_mode='concat')
    model.add_node(Dense(64, activation='relu'), name='fc1', input='shared_encoder')
    model.add_node(Dense(3, activation='softmax'), name='output', input='fc1', create_output=True)

    model.compile(optimizer='adam', loss={'output': 'categorical_crossentropy'})

    # generate dummy training data
    x_train_a = np.random.random((1000, timesteps, data_dim))
    x_train_b = np.random.random((1000, timesteps, data_dim))
    y_train = np.random.random((1000, 3))

    # generate dummy validation data
    x_val_a = np.random.random((100, timesteps, data_dim))
    x_val_b = np.random.random((100, timesteps, data_dim))
    y_val = np.random.random((100, 3))

    model.fit({'input_a': x_train_a, 'input_b': x_train_b, 'output': y_train},
              batch_size=64, nb_epoch=5,
              validation_data={'input_a': x_val_a, 'input_b': x_val_b, 'output': y_val})

