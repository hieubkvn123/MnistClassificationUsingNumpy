import math
import numpy as np

class MSE(object):
	def __init__(self):
		pass

	def __call__(self, y_true, y_pred):
	    N = y_true.shape[0]
	    loss = (1 / (2 * N)) * np.sum(np.square(y_true - y_pred))

	    loss_scalar = np.squeeze(loss) # loss in scalar
	    dy = -1 / N * (y_true - y_pred) # grad w.r.t predicted value

	    return loss_scalar, dy

class CrossEntropy(object):
	def __init__(self):
		pass 

	def __call__(self, y_true, y_pred):
		pass