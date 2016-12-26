import numpy as np
import pandas as pd

def init_weight_and_bias(M1, M2):
	W = np.random.randn(M1, M2) / np.sqrt(M1 + M2)
	# random gaussian with standard deviation of (1 / sqrt(M1 + M2))
	b = np.zeros(M2)
	return W.astype(np.float32), b.astype(np.float32)
	# cast to float32, so can be used in Theano and Tensorflow

def init_filter(shape, poolsz):
	# for convolutional neural networks
	pass
	
def relu(x):
	# can be used when using a older version of Theano with no relu built in
	return x * (x > 0)

def sigmoid(A):
	return 1 / (1 + np.exp(-A))

def softmax(A):
	expA = np.exp(A)
	return expA / expA.sum(axis = 1, keepdims = True)

def sigmoid_cost(T, Y):
	# cross entropy from definition for sigmoid cost
	# binary classification
	return -(T * np.log(Y) + (1 - T) * np.log(1 - Y)).sum()

def cost(T, Y):
	# general cross entropy function for softmax 
	# from definition
	return -(T * np.log(Y)).sum()

def cost2(T, Y):
	# same as cost(), just uses the targets to index Y
    # instead of multiplying by a large indicator matrix with mostly 0s
    N = len(T)
    return -log(Y[np.arange(N), T]).mean()


def error_rate(targets, predictions):
	return np.mean(targets != predictions)

def y2indicator(y):
	# one hot encoding for targets of k classes
	N = len(y)
	K = len(set(y))
	ind = np.zeros((N, K))
	for i in xrange(N):
		ind[i, y[i]] = 1
	return ind

def getData(balance_ones = True):
	# the first line is the header
	# first column is the target, second column is the pixels separated by space
	# images are 48 x 48 = 2304 size vectors
	# N = 35887

	X = []
	Y = []
	first = True

	for line in open('fer2013.csv'):
		if first:
			first = False
		else:
			row = line.split(',')
			Y.append(int(row(0)))
			X.append([int(p) for p in row(1).split()])
	X, Y = np.array(X) / 255.0, np.array(Y)
	# normalize X(0-255) to 0-1
	if balance_ones:
		X0, Y0 = X[Y!=1,:], Y[Y!=1]
		X1 = X[Y==1,:]
		X1 = np.repeat(X1, 9, axis=0)
		X = np.vstack([X0, X1])
		Y = np.concatenate((Y0, [1]*len(X1)))

	return X, Y

def getImageData():
	# keeps the original image shape
	# for convolutinal neural networks 
	X, Y = getData()
	N, D = X.shape
	d = int(np.sqrt(D))
	X = X.reshape(N, 1, d, d)
	# 1 for the color B & W
	return X, Y

def getBinaryData():
	X = []
	Y = []
	first = True
	for line in open('fer2013.csv'):
		if first:
			first = False
		else:
			row = line.split(',')
			y = int(row[0])
			if y == 0 or y == 1:
				Y.append(y)
				X.append([int(p) for p in row[1].split()])
	return np.array(X) / 255.0, np.array(Y)

def crossValidation(model, X, Y, K=5):
	X, Y = shuffle(X, Y)
	sz = len(Y) / K
	errors = []
	for k in xrange(K):
		xtr = np.concatenate([X[:k*sz, :], X[(k*sz+sz):,:]])
		ytr = np.concatenate([Y[:k*sz], Y[(k*sz+sz):]])
		xte = X[k*sz:(k*sz+sz), :]
		yte = Y[k*sz:(k*sz+sz)]

		model.fit(xtr, ytr)
		err = model.score(xte, yte)
		errors.append(err)

	print "errors:", errors
	return np.mean(errors)
	



