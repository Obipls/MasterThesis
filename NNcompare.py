# !/usr/bin/env python3.5
# coding=utf-8
from keras.models import Graph, Sequential, Model
from keras.layers import Input, LSTM, Dense, merge, Reshape
import numpy as np
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer


def sharedNN(docDict, nnDict):
    Y = list(nnDict.values())
    Y = [np.bool_(y) for y in Y]
    X = []
    encodedList = []
    shared_layer = Dense(64)
    docList = []

    for key in nnDict.keys():
        docTuple = (docDict[key[0]], docDict[key[1]])

        doc1 = docTuple[0].reshape(1, len(docTuple[0]))
        doc2 = docTuple[1].reshape(1, len(docTuple[1]))

        X.append(doc1)
        X.append(doc2)

        doc_a = Input(shape=(len(docTuple[0]),))
        doc_b = Input(shape=(len(docTuple[1]),))
        docList.append(doc_a)
        docList.append(doc_b)

        encoded_a = shared_layer(doc_a)
        encoded_b = shared_layer(doc_b)

        encodedList.append(encoded_a)
        encodedList.append(encoded_b)
    # print(encodedList)
    print(type(Y))
    print(len(X[0][0]))

    merged_vector = merge(encodedList, mode='concat', concat_axis=-1)
    predictions = Dense(len(Y), activation='sigmoid')(merged_vector)
    model = Model(input=docList, output=predictions)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(X, np.r_[Y], nb_epoch=10)


    #model.fit(X, np.r_[Y], nb_epoch=10)
