import numpy as np
import matplotlib.pyplot as plt

from util import getData, softmax, cost2, y2indicator, error_rate, relu
from sklearn.utils import shuffle

class ANN(object):
	def __init__(self, M):
		self.M = M
	def fit(self, X, Y, learning_rate=10e-7, reg=10e-7, epochs=10000, show_fig=False):
		X, Y = shuffle(X, Y)
		Xvalid, Yvalid = X[-1000:], Y[-1000:]
		X, Y = X[:-1000], Y[:-1000]

		N, D = X.shape
		K = len(set(Y))
		T = y2indicator(Y)
		self.W1 = np.random.randn(D, self.M) / np.sqrt(D + self.M)
		self.b1 = np.zeros(self.M)
		self.W2 = np.random.randn(self.M, K) / np.sqrt(self.M + K)
		self.b2 = np.zeros(K)

		costs = []
		best_validation_error = 1
		for i in xrange(epochs):
			pY, Z = self.forward(X)

			pY_T = pY - T
			self.W2 -= learning_rate*(Z.T.dot(pY_T) + reg*self.W2)
			self.b2 -= learning_rate*(pY_T.sum(axis=0) + reg*self.b2)
			dZ = pY_T.dot(self.W2.T) * (1 - Z*Z) # tanh BACKPROPAGATION
			self.W1 -= learning_rate*(X.T.dot(dZ) + reg*self.W1)
			self.b1 -= learning_rate*(dZ.sum(axis=0) + reg*self.b1)

			if i % 10 == 0:
				pYvalid, _ = self.forward(Xvalid)
				c = cost2(Yvalid, pYvalid)
				costs.append(c)
				e = error_rate(Yvalid, np.argmax(pYvalid, axis=1))
				print "i:", i, "cost:",c, "error:",e
				if e < best_validation_error:
					best_validation_error = e
		print "Best validation error:", best_validation_error

		if show_fig:
			plt.plot(costs)
			plt.show()
	def forward(self, X):
		Z = np.tanh(X.dot(self.W1) + self.b1)
		return softmax(Z.dot(self.W2) + self.b2), Z
	def predict(self, X):
		pY, _ = self.forward(X)
		return np.argmax(pY, axis=1)
	def score(self, X, Y):
		prediction = self.predict(X)
		return 1 - error_rate(Y, prediction)


def main():
	X, Y = getData()
	model = ANN(200)
	model.fit(X, Y, reg=0, show_fig=True)
	print model.score(X, Y)

if __name__ == '__main__':
	main()