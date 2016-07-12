# !/usr/bin/env python3.5
# coding=utf-8
import numpy as np
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth

def KMclusterer(nclusters,X):
	cls = KMeans(n_clusters=nclusters, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=-1)
	cls.fit_predict(X)
	return cls.labels_

def MSclusterer(X):
	X = X.toarray()
	bandwidth = estimate_bandwidth(X, quantile=0.5, n_samples=5000)
	ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
	ms.fit_predict(X)
	return ms.labels_



