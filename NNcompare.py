# !/usr/bin/env python3.5
# coding=utf-8
from keras.models import Graph, Sequential, Model
from keras.layers import Input, LSTM, Dense, merge, Reshape
import numpy as np
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer



def sharedNN(docDict,nnDict):

	Y = list(nnDict.values())
	X = []
	encodedList = []
	shared_layer = Dense(64)
	docList = []
	
	for key in nnDict.keys():
		docTuple = (docDict[key[0]],docDict[key[1]])

		doc1=docTuple[0].reshape(1,len(docTuple[0]))
		doc2=docTuple[1].reshape(1,len(docTuple[1]))

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
		#print(encodedList)
	print(len(Y))
	print(len(X))

	merged_vector = merge(encodedList, mode='concat', concat_axis=-1)
	predictions = Dense(1, activation='sigmoid')(merged_vector)
	model = Model(input=docList, output=predictions)
	model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

<<<<<<< HEAD
    merged_vector = merge([encoded_a, encoded_b], mode='concat', concat_axis=-1)

    predictions = Dense(1, activation='sigmoid')(merged_vector)

    model = Model(input=[doc_a, doc_b], output=predictions)

    model.compile(optimizer='rmsprop', loss='binary_crossentropy',
                  metrics=['accuracy'])
    print(X[:3])


    model.train_on_batch(X[0], np.r_[Y])


    #model.fit([doc1,doc2], np.r_[Y], nb_epoch=10)
=======
	model.fit(X,Y, nb_epoch=10)
>>>>>>> f5f96726d2ae48b91c26c3b05bea863d42774863
