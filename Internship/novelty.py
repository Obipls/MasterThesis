import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn.svm import OneClassSVM
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import make_scorer
import pickle
from math import log10


def main():
	entries = pickle.load( open( "entries.p", "rb" ) )
	X_train = []
	for value in entries.values():
		if value[1] == 1:
			x = [value[2],log10(value[3])]
		elif value[2] == 1:
			x = [value[1],log10(value[3])]
		X_train.append(x)
	X_train = np.r_[X_train] 

	xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
	X_test =np.r_[[[0.8,3.3],[0.99,3],[0.97,2.5],[0.98,3],[0.99,2.3]]]
	X_outliers = np.r_[[[0.5,3.5],[0.3,2.6],[0.6,3.4],[0.1,0.2]]]

	# Parameter borders
	parameterBorders = [{'nu':[0.001,0.01,0.1],'kernel': ['rbf'], 'gamma': [0.1,1,10]}]
	clfDict={}
	for i, setting in enumerate(ParameterGrid(parameterBorders)):
		clf = OneClassSVM(nu=setting['nu'], kernel=setting['kernel'],gamma=setting['gamma'])
		clf.fit(X_train)
		y_pred_train = clf.predict(X_train)
		n_error_train = y_pred_train[y_pred_train == -1].size
		

		clfDict[clf] = float(n_error_train)

	clf = min(clfDict, key=clfDict.get)
	print(clfDict[clf])/len(X_train)
	# Make predictions
	y_pred_train = clf.predict(X_train)
	y_pred_test = clf.predict(X_test)
	y_pred_outliers = clf.predict(X_outliers)
	
	#Testing
	print(clf.predict([[1,3.3]]))
	
	# Count wrong predictions
	n_error_train = y_pred_train[y_pred_train == -1].size
	n_error_test = y_pred_test[y_pred_test == -1].size
	n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size


	# plot the line, the points, and the nearest vectors to the plane
	Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)

	plt.title("Novelty Detection")
	plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.Blues_r)
	a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='red')
	plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='orange')

	b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white')
	b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='green')
	c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red')
	plt.axis('tight')
	plt.xlim((0.5,1.1))
	plt.ylim((3, 3.5))
	plt.legend([a.collections[0], b1, b2, c],
		["learned frontier", "training observations", "new regular observations", "new abnormal observations"], 
		loc="upper left", prop=matplotlib.font_manager.FontProperties(size=11))
	plt.xlabel("Cosinus Phi\n" "error train: %d ; errors novel regular: %d ; " "errors novel abnormal: %d"
			% (n_error_train, n_error_test, n_error_outliers))
	plt.ylabel("Logarithmic Wattage")
	plt.show()

if __name__ == '__main__':
	main()