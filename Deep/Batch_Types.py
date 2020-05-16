import sys
sys.path.insert(1, "..")
import numpy as np
from General_NN_Functions import sigmoid, tanh, relu, derivative
from Regularization_Methods import *

class BatchType:
    """
    Parent class of batch type classes
    """

    def forward_propagate(self):
        pass

    def backward_propagate(self):
        pass


class batch_gradient_descent(BatchType):

    def __init__(self, regularization_method):
        self.regularization_method = regularization_method

    def forward_propagate(self, X, weights, funcs):
        return self.regularization_method.forward_propagate(X, weights, funcs)

    def backward_propagate(self, weights, cache, funcs, X, Y):
        return self.regularization_method.backward_propagate(weights, cache, funcs, X, Y)


class mini_batch_gradient_descent(BatchType):
    """
    Mini Batch Gradient Descent

    Note that for train index to update, backward_propagate must be called.
    """

    def __init__(self, batch_size, regularization_method):
        self.batch_size = batch_size
        self.train_index = 0

        self.regularization_method = regularization_method

    def get_res(self, X):
        return max(0, (self.train_index + self.batch_size) - X.shape[1])

    def update_train_index(self, X):
        res = self.get_res(X)
        self.train_index = res if res != 0 else self.train_index + self.batch_size

    def get_batch(self, A):
        A1 = A[:, self.train_index:self.train_index + self.batch_size]
        res = self.get_res(A)
        if res:
            A2 = A[:, 0:res:]
            return np.c_[A1, A2]
        else:
            return A1

    def forward_propagate(self, X, weights, funcs):
        Xbatch = self.get_batch(X)
        return self.regularization_method.forward_propagate(Xbatch, weights, funcs)

    def backward_propagate(self, weights, cache, funcs, X, Y):
        Xbatch = self.get_batch(X)
        Ybatch = self.get_batch(Y)
        self.update_train_index(X)
        return self.regularization_method.backward_propagate(weights, cache, funcs, Xbatch, Ybatch)


class stochastic_gradient_descent(mini_batch_gradient_descent):

    def __init__(self, regularization_method):
        super().__init__(batch_size=1, regularization_method=regularization_method)
