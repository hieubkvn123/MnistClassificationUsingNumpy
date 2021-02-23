import math
import pickle
import numpy as np 

class Model(object):
	def __init__(self, lr=0.00001):
		self.layers = []
		self.lr = lr

	def add(self, layer):
		if(isinstance(layer, list)):
			self.layers = layer
		else:
			self.layers.append(layer)

	def forward(self, inputs):
		assert len(self.layers) >= 1
		
		outputs = None
		for i, layer in enumerate(self.layers):
			if(i == 0):
				outputs = layer.forward(inputs)
			else:
				outputs = layer.forward(outputs)

		return outputs

	def backward(self, loss_grad):
		current_grad = loss_grad
		for layer in self.layers[::-1]:
			layer.backward(current_grad)
			current_grad = layer.dA

	def step(self):
		for layer in self.layers[::-1]:
			if(layer.trainable):
				layer.step(self.lr)

	def save_model(self, model_path):
		pickle.dump(self.layers, open(model_path, 'wb'))

	def load_model(self, model_path):
		self.layers = pickle.load(open(model_path, 'rb'))