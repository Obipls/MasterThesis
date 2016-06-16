# !/usr/bin/env python3.5
# coding=utf-8
import numpy as np
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth

def KNNclusterer(nclusters,X):
	cls = KMeans(n_clusters=nclusters, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1)
	#cls.fit(X)
	cls.fit_predict(X)
	return cls.labels_

def MSclusterer(X):
	bandwidth = estimate_bandwidth(X, quantile=0.09, n_samples=5000)
	ms = MeanShift()
	ms.fit_predict(X)
	return ms.labels_



