# !/usr/bin/env python3.5
# coding=utf-8
from keras.models import Graph, Sequential, Model
from keras.layers import Input, LSTM, Dense, merge, Reshape
import numpy as np
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer


def LSTMcomp(docTuple,x):
    seqLength = 3000

    doc_a = Input(shape=(len(docTuple[0]),))
    doc_b = Input(shape=(len(docTuple[1]),))

    shared_layer = Dense(64)

    encoded_a = shared_layer(doc_a)
    encoded_b = shared_layer(doc_b)

    doc1 = Reshape((None,len(docTuple[0])),input_shape=(docTuple[0],))
    doc2 = Reshape((None,len(docTuple[1])),input_shape=(docTuple[1],))



    merged_vector = merge([encoded_a, encoded_b], mode='concat', concat_axis=-1)

    predictions = Dense(1, activation='sigmoid')(merged_vector)

    model = Model(input=[doc_a, doc_b], output=predictions)

    model.compile(optimizer='rmsprop', loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit([doc1,doc2], 2, nb_epoch=10)
