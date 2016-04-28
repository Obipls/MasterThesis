# !/usr/bin/env python3.5
# coding=utf-8
from keras.models import Graph, Sequential, Model
from keras.layers import Input, LSTM, Dense, merge, Reshape
import numpy as np
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer


def LSTMcomp(docTuple, maxTokens):

	seqLength = 3000

	doc_a = Input(shape=(seqLength,maxTokens+1))
	doc_b = Input(shape=(seqLength,maxTokens+1))


	shared_lstm = LSTM(64)

	encoded_a = shared_lstm(doc_a)
	encoded_b = shared_lstm(doc_b)

	doc1 = docTuple[0]
	doc2 = docTuple[1]


	merged_vector = merge([encoded_a, encoded_b], mode='concat', concat_axis=-1)

	predictions = Dense(1,activation='sigmoid')(merged_vector)

	model = Model(input=[doc_a, doc_b], output=predictions)

	model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])


	model.fit([doc1,doc2],1,nb_epoch=10)
