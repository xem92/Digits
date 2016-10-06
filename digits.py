#Author: Joan Guillem Castell Ros-Zanet
#Enginyeria La Salle - Universitat Ramon Llull
#Data Mining
#Optimització, Preprocés i IBL aplicats amb scikit-learn
#Pràctica 2 (Week 8): Aprendre a categoritzar imatges de dígits per IBL

#!/opt/local/bin/python2.7

#required libraries
from sklearn import preprocessing
import argparse
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.cross_validation
import sklearn.decomposition
import sklearn.grid_search
import sklearn.neighbors
import sklearn.metrics
import matplotlib.pyplot as plt
import sys
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import FactorAnalysis
from sklearn.cross_validation import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import operator

def averages():
	averageX = []
	for i in xrange(X.shape[0]):
		averageX.append(np.average(X[i]))

	averageY = np.average(Y)
	print ("Mitjanes de les Xs:")
	print averageX
	print ("Mitjana de Y:")
	print averageY

def standardDeviation():
	stdY = np.std(Y)
	print stdY

def trainingElemPerClass():
	sampleCount = np.zeros(10)
	for i in xrange(Y.shape[0]):
		sampleCount[Y[i]] += 1
	for i in range(0,10):
		print "%d: %.0f"%(i,sampleCount[i])

def scatterPlot():
	sample = int(raw_input('Mostra a pintar: '))
	#Samples in black and white
	print "Numero que es printa: " + str(Y[sample])
	XtoMatrix = X[sample].reshape(8,8)
	fig, ax = plt.subplots()
	image = XtoMatrix
	ax.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
	ax.set_title("Numero: " + str(Y[sample]))
	# Move left and bottom spines outward by 10 points
	ax.spines['left'].set_position(('outward', 10))
	ax.spines['bottom'].set_position(('outward', 10))
	# Hide the right and top spines
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	# Only show ticks on the left and bottom spines
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')
	plt.show()

def scatterPlotTrainNorm():
	sample = int(raw_input('Mostra a pintar: '))
	#Samples in black and white
	print "Numero que es printa: " + str(Y[sample])
	XtoMatrix = X_normalized[sample].reshape(8,8)
	fig, ax = plt.subplots()
	image = XtoMatrix
	ax.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
	ax.set_title("Numero: " + str(Y[sample]))
	# Move left and bottom spines outward by 10 points
	ax.spines['left'].set_position(('outward', 10))
	ax.spines['bottom'].set_position(('outward', 10))
	# Hide the right and top spines
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	# Only show ticks on the left and bottom spines
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')
	plt.show()

def printPCAScaled():
	#Principal Component Analisys (PCA)
	#X train normalized
	pca = PCA(n_components=2, whiten=False)
	X_PCA_train = pca.fit_transform(X_normalized_train)
	plt.clf()
	plt.scatter(X_PCA_train[:,0], X_PCA_train[:,1], s=100, c=Y_train)
	plt.colorbar()
	plt.xlabel('PC1')
	plt.ylabel('PC2')
	plt.title("Principal Component Analisys (PCA), scaled")
	plt.show()

def printPCA():
	#Principal Component Analisys (PCA)
	#X train 
	pca = PCA(n_components=2, whiten=False)
	X_PCA_train = pca.fit_transform(X_train)
	plt.clf()
	plt.scatter(X_PCA_train[:,0], X_PCA_train[:,1], s=100, c=Y_train)
	plt.colorbar()
	plt.xlabel('PC1')
	plt.ylabel('PC2')
	plt.title("Principal Component Analisys (PCA)")
	plt.show()

def printSVDScaled():
	#Singular Value Decomposition (SVD)
	#X train normalized
	svd = TruncatedSVD(n_components=2)
	X_SVD_train = svd.fit_transform(X_normalized_train)
	plt.clf()
	plt.scatter(X_SVD_train[:,0], X_SVD_train[:,1], s=100, c=Y_train)
	plt.colorbar()
	plt.xlabel('PC1')
	plt.ylabel('PC2')
	plt.title("Singular Value Decomposition (SVD), scaled")
	plt.show()

def printSVD():
	#Singular Value Decomposition (SVD)
	#X train 
	svd = TruncatedSVD(n_components=2)
	X_SVD_train = svd.fit_transform(X_train)
	plt.clf()
	plt.scatter(X_SVD_train[:,0], X_SVD_train[:,1], s=100, c=Y_train)
	plt.colorbar()
	plt.xlabel('PC1')
	plt.ylabel('PC2')
	plt.title("Singular Value Decomposition (SVD)")
	plt.show()

def printFA():
	#Factor Analysis (FA)
	#X train 
	fa = FactorAnalysis(n_components=2)
	X_FA_train = fa.fit_transform(X_train)
	plt.clf()
	plt.scatter(X_FA_train[:,0], X_FA_train[:,1], s=100, c=Y_train)
	plt.colorbar()
	plt.xlabel('PC1')
	plt.ylabel('PC2')
	plt.title("Factor Analysis (FA)")
	plt.show()

def printFAScaled():
	#Factor Analysis (FA)
	#Scaled X train 
	fa = FactorAnalysis(n_components=2)
	X_FA_train = fa.fit_transform(X_normalized_train)
	plt.clf()
	plt.scatter(X_FA_train[:,0], X_FA_train[:,1], s=100, c=Y_train)
	plt.colorbar()
	plt.xlabel('PC1')
	plt.ylabel('PC2')
	plt.title("Factor Analysis (FA), scaled")
	plt.show()

def tests():
	components = int(raw_input('# components: '))
	
	#Compute test
	print "PCA:"
	compute_test_PCA(components)
	print "\nScaled PCA:"
	compute_test_PCA_Scaled(components)
	print "\nSVD:"
	compute_test_SVD(components)
	print "\nScaled SVD:"
	compute_test_SVD_Scaled(components)
	print "\nFA:"
	compute_test_FA(components)
	print "\nScaled FA:"
	compute_test_FA_Scaled(components)
	print "\nScaled(64 components):"
	compute_test_Scaled()
	print "\nOriginal(64 components):"
	compute_test()

	scoreBoard.sort(key=lambda x: x.percentage, reverse=True)
	
def ranking():
	results = int(raw_input('# results: '))
	if results < len(scoreBoard):
		print "\n\nTop%d results: "%(results)
		for i in range(0,results):
			print "Pos" + str(i+1) + ": " + scoreBoard[i].description + "->" + str(scoreBoard[i].percentage)
	else:
		print "To many results"


def main():
	try:
		print ("\n\n0.- Exit\n1.- Mitjanes\n2.- Desviacions tipiques de Y\n3.- Nombre d'elements d'entrenament per cada classe\n4.- Scatter plot\n5.- Mostra PCA de X\n6.- Mostrar PCA de X escalat\n7.- Mostra SVD de X\n8.- Mostra SVD de X escalat\n9.- Tests\n10.- Ranking millors puntuacions\n11.- Mostra Factor Analysis\n12.- Mostra Factor Analysis escalat\n13.- Scatter plot amb X escalat")
		mode = int(raw_input('Opcio: '))
		if mode == 0:
			sys.exit()
		elif mode == 1:
			averages()
		elif mode == 2:
			standardDeviation()
		elif mode == 3:
			trainingElemPerClass()
		elif mode == 4:
			scatterPlot()
		elif mode == 5:
			printPCA()
		elif mode == 6:
			printPCAScaled()
		elif mode == 7:
			printSVD()
		elif mode == 8:
			printSVDScaled()
		elif mode == 9:
			tests()
		elif mode == 10:
			ranking()
		elif mode == 11:
			printFA()
		elif mode == 12:
			printFAScaled()
		elif mode == 13:
			scatterPlotTrainNorm()
		else:
			print "Invalid option"
	except ValueError:
		print "Not a number"

def compute_test_PCA_Scaled(components):
	n = X.shape[0]
	Kfolds = KFold(n, n_folds=10, shuffle=False, random_state=None)
	scores = []
	for i in range(1,11):
		score = 0
		for train_index, test_index in Kfolds:
			X_train, X_test = X[train_index], X[test_index]
			Y_train, Y_test = Y[train_index], Y[test_index]

			#Scale X
			scaler = preprocessing.StandardScaler().fit(X_train)
			X_normalized_train = scaler.transform(X_train)
			X_normalized_test = scaler.transform(X_test)

			#Principal Component Analisys (PCA)
			#X train normalized
			pca = PCA(n_components=components, whiten=False)
			pca.fit(X_normalized_train)
			X_PCA_train = pca.transform(X_normalized_train)
			X_PCA_test = pca.transform(X_normalized_test)

			neigh = KNeighborsClassifier(n_neighbors=i)
			model = neigh.fit(X_PCA_train,Y_train)
			predicted_y = neigh.predict(X_PCA_test)
			score += accuracy_score(Y_test, predicted_y)*100

		print "%d neighbors: %.2f" %(i, score/10)
		scoreClass = Score(score/10, "%d neighbors (%s %d comp)" %(i, "PCA scaled", components))
		scoreBoard.append(scoreClass)
	return scores

def compute_test_PCA(components):
	n = X.shape[0]
	Kfolds = KFold(n, n_folds=10, shuffle=False, random_state=None)

	scores = []
	for i in range(1,11):
		score = 0
		for train_index, test_index in Kfolds:
			X_train, X_test = X[train_index], X[test_index]
			Y_train, Y_test = Y[train_index], Y[test_index]

			#Principal Component Analisys (PCA)
			#X train normalized
			pca = PCA(n_components=components, whiten=False)
			pca.fit(X_train)
			X_PCA_train = pca.transform(X_train)
			X_PCA_test = pca.transform(X_test)

			neigh = KNeighborsClassifier(n_neighbors=i)
			model = neigh.fit(X_PCA_train,Y_train)
			predicted_y = neigh.predict(X_PCA_test)
			score += accuracy_score(Y_test, predicted_y)*100

		print "%d neighbors: %.2f" %(i, score/10)
		scoreClass = Score(score/10, "%d neighbors (%s %d comp)" %(i, "PCA", components))
		scoreBoard.append(scoreClass)
	return scores

def compute_test_Scaled():
	n = X.shape[0]
	Kfolds = KFold(n, n_folds=10, shuffle=False, random_state=None)

	scores = []
	for i in range(1,11):
		score = 0
		for train_index, test_index in Kfolds:
			X_train, X_test = X[train_index], X[test_index]
			Y_train, Y_test = Y[train_index], Y[test_index]

			#Scale X
			scaler = preprocessing.StandardScaler().fit(X_train)
			X_normalized_train = scaler.transform(X_train)
			X_normalized_test = scaler.transform(X_test)

			neigh = KNeighborsClassifier(n_neighbors=i)
			model = neigh.fit(X_normalized_train,Y_train)
			predicted_y = neigh.predict(X_normalized_test)
			score += accuracy_score(Y_test, predicted_y)*100

		print "%d neighbors: %.2f" %(i, score/10)
		scoreClass = Score(score/10, "%d neighbors (%s %d comp)" %(i, "Scaled", 64))
		scoreBoard.append(scoreClass)
	return scores

def compute_test():
	n = X.shape[0]
	Kfolds = KFold(n, n_folds=10, shuffle=False, random_state=None)

	scores = []
	for i in range(1,11):
		score = 0
		for train_index, test_index in Kfolds:
			X_train, X_test = X[train_index], X[test_index]
			Y_train, Y_test = Y[train_index], Y[test_index]

			neigh = KNeighborsClassifier(n_neighbors=i)
			model = neigh.fit(X_train,Y_train)
			predicted_y = neigh.predict(X_test)
			score += accuracy_score(Y_test, predicted_y)*100

		print "%d neighbors: %.2f" %(i, score/10)
		scoreClass = Score(score/10, "%d neighbors (%s %d comp)" %(i, "Original", 64))
		scoreBoard.append(scoreClass)
	return scores

def compute_test_SVD(components):
	n = X.shape[0]
	Kfolds = KFold(n, n_folds=10, shuffle=False, random_state=None)

	scores = []
	for i in range(1,11):
		score = 0
		for train_index, test_index in Kfolds:
			X_train, X_test = X[train_index], X[test_index]
			Y_train, Y_test = Y[train_index], Y[test_index]

			#Singular Value Decomposition (SVD)
			#X train 
			svd = TruncatedSVD(n_components=components)
			svd.fit(X_normalized_train)
			X_SVD_train = svd.transform(X_train)
			X_SVD_test = svd.transform(X_test)

			neigh = KNeighborsClassifier(n_neighbors=i)
			model = neigh.fit(X_SVD_train,Y_train)
			predicted_y = neigh.predict(X_SVD_test)
			score += accuracy_score(Y_test, predicted_y)*100

		print "%d neighbors: %.2f" %(i, score/10)
		scoreClass = Score(score/10, "%d neighbors (%s %d comp)" %(i, "SVD", components))
		scoreBoard.append(scoreClass)
	return scores

def compute_test_SVD_Scaled(components):
	n = X.shape[0]
	Kfolds = KFold(n, n_folds=10, shuffle=False, random_state=None)

	scores = []
	for i in range(1,11):
		score = 0
		for train_index, test_index in Kfolds:
			X_train, X_test = X[train_index], X[test_index]
			Y_train, Y_test = Y[train_index], Y[test_index]

			#Scale X
			scaler = preprocessing.StandardScaler().fit(X_train)
			X_normalized_train = scaler.transform(X_train)
			X_normalized_test = scaler.transform(X_test)

			#Singular Value Decomposition (SVD)
			#X train scaled
			svd = TruncatedSVD(n_components=components)
			svd.fit(X_normalized_train)
			X_SVD_train = svd.transform(X_normalized_train)
			X_SVD_test = svd.transform(X_normalized_test)

			neigh = KNeighborsClassifier(n_neighbors=i)
			model = neigh.fit(X_SVD_train,Y_train)
			predicted_y = neigh.predict(X_SVD_test)
			score += accuracy_score(Y_test, predicted_y)*100

		print "%d neighbors: %.2f" %(i, score/10)
		scoreClass = Score(score/10, "%d neighbors (%s %d comp)" %(i, "SVD Scaled", 64))
		scoreBoard.append(scoreClass)
	return scores

def compute_test_FA_Scaled(components):
	n = X.shape[0]
	Kfolds = KFold(n, n_folds=10, shuffle=False, random_state=None)

	scores = []
	for i in range(1,11):
		score = 0
		for train_index, test_index in Kfolds:
			X_train, X_test = X[train_index], X[test_index]
			Y_train, Y_test = Y[train_index], Y[test_index]

			#Scale X
			scaler = preprocessing.StandardScaler().fit(X_train)
			X_normalized_train = scaler.transform(X_train)
			X_normalized_test = scaler.transform(X_test)

			#Factor Analysis (FA)
			#Scaled X train
			fa = FactorAnalysis(n_components=components)
			fa.fit(X_normalized_train)
			X_FA_train = fa.transform(X_normalized_train)
			X_FA_test = fa.transform(X_normalized_test)

			neigh = KNeighborsClassifier(n_neighbors=i)
			model = neigh.fit(X_FA_train,Y_train)
			predicted_y = neigh.predict(X_FA_test)
			score += accuracy_score(Y_test, predicted_y)*100

		print "%d neighbors: %.2f" %(i, score/10)
		scoreClass = Score(score/10, "%d neighbors (%s %d comp)" %(i, "FA Scaled", 64))
		scoreBoard.append(scoreClass)
	return scores

def compute_test_FA(components):
	n = X.shape[0]
	Kfolds = KFold(n, n_folds=10, shuffle=False, random_state=None)

	scores = []
	for i in range(1,11):
		score = 0
		for train_index, test_index in Kfolds:
			X_train, X_test = X[train_index], X[test_index]
			Y_train, Y_test = Y[train_index], Y[test_index]

			#Factor Analysis (FA)
			#Scaled X train
			fa = FactorAnalysis(n_components=components)
			fa.fit(X_normalized_train)
			X_FA_train = fa.transform(X_train)
			X_FA_test = fa.transform(X_test)

			neigh = KNeighborsClassifier(n_neighbors=i)
			model = neigh.fit(X_FA_train,Y_train)
			predicted_y = neigh.predict(X_FA_test)
			score += accuracy_score(Y_test, predicted_y)*100

		print "%d neighbors: %.2f" %(i, score/10)
		scoreClass = Score(score/10, "%d neighbors (%s %d comp)" %(i, "FA Scaled", 64))
		scoreBoard.append(scoreClass)
	return scores

class Score:
	def __init__(self,per,desc):
		self.percentage = per
		self.description = desc

if __name__ == "__main__":

	#load dataset
	digits = sklearn.datasets.load_digits()
	X = digits.data
	Y = digits.target

	#Split arrays into random train and test subsets
	X_train, X_test, Y_train, Y_test = sklearn.cross_validation.train_test_split(X, Y, test_size=0.30)

	#X scaled
	X_normalized = preprocessing.scale(X, axis=0, with_mean=True, with_std=True, copy=True)

	#Scale X
	scaler = preprocessing.StandardScaler().fit(X)
	X_normalized_train = scaler.transform(X_train)
	X_normalized_test = scaler.transform(X_test)

	scoreBoard = []

	while True:
		main()
