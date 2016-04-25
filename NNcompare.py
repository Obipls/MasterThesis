# !/usr/bin/env python3.5
# coding=utf-8
from keras.models import Graph, Sequential, Model
from keras.layers import Input, LSTM, Dense, merge, Reshape
import numpy as np
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer


def LSTMcomp(docTuple, maxTokens):

    seqLength = 5000

    doc_a = Input(shape=(seqLength, maxTokens))
    doc_b = Input(shape=(seqLength, maxTokens))


    shared_lstm = LSTM(64)

    encoded_a = shared_lstm(doc_a)
    encoded_b = shared_lstm(doc_b)


    merged_vector = merge([encoded_a, encoded_b], mode='concat', concat_axis=-1)

    predictions = Dense(1,activation='sigmoid')(merged_vector)

    model = Model(input=[doc_a, doc_b], output=predictions)

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    doc1= Reshape((10,3), input_shape=docTuple[0])
    doc2= Reshape((10,3), input_shape=docTuple[1])

    model.fit([docTuple[0],docTuple[1]],1,nb_epoch=10)
