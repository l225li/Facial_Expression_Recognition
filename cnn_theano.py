import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from theano.tensor.nnet import conv2d
from theano.tensor.signal import downsample

from util import getImageData, error_rate, init_weight_and_bias, init_filter
from ann_theano import HiddenLayer

class ConvPoolLayer(object):
	def __init__(self, mi, mo, fw=5, fh=5, poolsz=(2,2)):
		# mi = input feature map size
		# mo = output feature map size
		sz = (mo, mi, fw, fh)
		W0 = init_filter(sz, poolsz)
		self.W = theano.shared(W0)
		b0 = np.zeros(mo, dtype=np.float32)
		self.b = theano.shared(b0)
		self.poolsz = poolsz
		self.params = [self.W, self.b]

	def forward(self, X):
		conv_out = conv2d(input=X, filters=self.W)
		pooled_out = downsample.max_pool_2d(
			input=conv_out,
			ds=self.poolsz,
			ignore_border=True
		)
		return T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

class CNN(object):
	def __init__(self, convpool_layer_sizes, hidden_layer_sizes):
		self.convpool_layer_sizes = convpool_layer_sizes
		self.hidden_layer_sizes = hidden_layer_sizes

	def fit(self, X, Y, lr=10e-5, mu=0.99, reg=10e-7, decay=0.99999, eps=10e-3, batch_sz=30, epochs=100, show_fig=True):
		lr = np.float32(lr)
		mu = np.float32(mu)
		reg = np.float32(reg)
		decay = np.float32(decay)
		eps = np.float32(eps)

		# make a validation set
		X, Y = shuffle(X, Y)
		