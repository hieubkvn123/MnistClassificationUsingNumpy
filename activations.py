import math 
import numpy as np

class SigmoidLayer(object):
	def __init__(self):
		self.trainable = False
		self.inputs = None

	def forward(self, inputs):
		self.z = 1 / (1 + np.exp(-inputs))

		return self.z

	def backward(self, prev_grad):
		# gradients with respect to inputs
		self.dA = np.multiply(prev_grad , np.multiply(self.z, (1 - self.z)))