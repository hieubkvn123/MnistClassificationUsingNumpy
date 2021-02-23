import math
import numpy as np

class DenseLayer(object):
    def __init__(self, input_shape, n_out, trainable=True):
        self.input_shape = input_shape
        self.n_out = n_out

        ### Xaver/Glorot initialization ###
        std = np.sqrt(2 / (input_shape[1] + n_out))
        self.weights = np.random.normal(size=(input_shape[1], n_out), loc=0, scale=std)
        self.bias = np.random.rand(n_out)
        self.trainable = True

    def forward(self, inputs):
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights)

        return self.z

    def backward(self, prev_grad):
        # gradients with respect to parameters
        self.dW = np.dot(self.inputs.T, prev_grad)

        # gradients with respect to inputs
        self.dA = np.dot(prev_grad, self.weights.T)

    def step(self, lr):
        ### Standard SGD - need expansion on different optim algo ###
        self.weights = self.weights - lr * self.dW
