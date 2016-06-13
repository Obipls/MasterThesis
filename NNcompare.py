# !/usr/bin/env python3.5
# coding=utf-8
from keras.models import Sequential, Model
from keras.layers import Input, LSTM, Dense, merge, Reshape
import numpy as np
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from collections import Counter


def sharedNN(docDict, nnDict):
	Y = [np.bool_(y) for y in list(nnDict.values())]
	X = []

	for key in nnDict.keys():
		docTuple = (docDict[key[0]], docDict[key[1]])

		doc1 = docTuple[0].reshape(1, len(docTuple[0]))
		doc2 = docTuple[1].reshape(1, len(docTuple[1]))

		X.append(np.vstack((doc1, doc2)))

	#import pdb; pdb.set_trace()

	X = np.asarray([i[0] for i in X], dtype=np.float32)
	Y = np.asarray([[0, 1] if i == True else [1, 0] for i in Y], dtype=np.int32)

	print(X.shape)
	print(Y.shape)

	model = Sequential()
	model.add(Dense(64, input_shape=(X.shape[1], ), activation='relu'))
	model.add(Dense(2, activation='softmax'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.fit(X, Y, nb_epoch=20)
	#import pdb; pdb.set_trace()

def embedNN(X,Y):
	print(Counter(Y))
	X = np.asarray([i for i in X], dtype=np.float32)
	Y = np.asarray([i for i in Y], dtype=np.int32)
	print(X.shape)
	print(Y.shape)

	model = Sequential()
	model.add(Dense(64, input_shape=(X.shape[1], ), activation='relu'))
	model.add(Dense(2, activation='softmax'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.fit(X, Y, nb_epoch=20)

		
