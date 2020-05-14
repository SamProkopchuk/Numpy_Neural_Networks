import sys
sys.path.insert(1, "..")
import numpy as np
from General_NN_Functions import sigmoid, tanh, relu, derivative


class RegularizationMethod:
    """
    Parent class of regularization method classes
    """

    pass


class l2_regularization(RegularizationMethod):
    """
    l2 regularization
    """

    def __init__(self, lambd):
        self.lambd = lambd

    def forward_propagate(self, X, weights, funcs):
        pass


class dropout(RegularizationMethod):
    """
    Dropout

    __init__ params:
        keep_prob -> probability of weights being kept
        can be set to 1 for no regularization.
    """

    def __init__(self, keep_prob):
        self.keep_prob = keep_prob
        self.drops = {}

    def forward_propagate(self, X, weights, funcs):
        cache = {"A0": X}
        for l in range(1, 1 + len(weights) // 2):
            cache[f"Z{l}"] = np.dot(weights[f"W{l}"], cache[f"A{l-1}"]) + weights[f"b{l}"]
            Al = funcs[f"L{l}_func"](cache[f"Z{l}"])
            Dl = np.random.rand(*Al.shape)
            self.drops[f"D{l}"] = np.where(Dl < self.keep_prob, 1, 0)
            Al = Al * self.drops[f"D{l}"]
            cache[f"A{l}"] = Al / self.keep_prob
        return cache

    def backward_propagate(self, weights, cache, funcs, X, Y):
        m = X.shape[1]
        L = len(weights) // 2
        grads = {}
        # Do first back prob calculation separately since it's different:
        dZl = cache[f"A{L}"] - Y
        grads[f"dW{L}"] = np.dot(dZl, cache[f"A{L-1}"].T) / m
        grads[f"db{L}"] = np.sum(dZl, axis=1, keepdims=True) / m

        for l in range(L - 1, 0, -1):
            # Define lth function for cleaner code:
            funcl = funcs[f"L{l}_func"]

            # Define Al based upon drops dict:
            Al = np.dot(weights[f"W{l+1}"].T, dZl) * self.drops[f"D{l}"] / self.keep_prob

            # perform a backprop step:
            dZl = Al * derivative(
                f=funcl, fx=funcl(cache[f"Z{l}"]))
            grads[f"dW{l}"] = np.dot(dZl, cache[f"A{l-1}"].T) / m
            grads[f"db{l}"] = np.sum(dZl, axis=1, keepdims=True) / m

        return grads


class DescentMethod:
    """
    Parent class of descent method classes
    """

    def forward_propagate(self):
        pass

    def backward_propagate(self):
        pass


class batch_gradient_descent(DescentMethod):

    def __init__(self, regularization_method):
        self.regularization_method = regularization_method

    def forward_propagate(self, X, weights, funcs):
        return self.regularization_method.forward_propagate(X, weights, funcs)

    def backward_propagate(self, weights, cache, funcs, X, Y):
        return self.regularization_method.backward_propagate(weights, cache, funcs, X, Y)


class mini_batch_gradient_descent(DescentMethod):
    """
    Mini Batch Gradient Descent

    Note that for train index to update, backward_propagate must be called.
    """

    def __init__(self, batch_size, regularization_method=None):
        self.batch_size = batch_size
        self.train_index = 0

        self.regularization_method = regularization_method

    def get_res(self, X):
        return max(0, X.shape[1] - (self.train_index + self.batch_size))

    def update_train_index(self, X):
        res = self.get_res(X)
        self.train_index = res if res == 0 else self.train_index + self.batch_size

    def get_batch(self, X):
        X1 = X[:, self.train_index:self.train_index + self.batch_size]
        res = self.get_res(X)
        if res:
            X2 = X[:, 0:res:]
            return np.c_[X1, X2]
        else:
            return X1

    def forward_propagate(self, X, weights, funcs):
        Xbatch = self.get_batch(X)
        return self.regularization_method.forward_propagate(Xbatch, weights, funcs)

    def backward_propagate(self, weights, cache, funcs, X, Y):
        Xbatch = self.get_batch(X)
        Ybatch = self.get_batch(Y)
        return self.regularization_method.backward_propagate(weights, cache, funcs, Xbatch, Ybatch)


class stochastic_gradient_descent(mini_batch_gradient_descent):

    def __init__(self, regularization_method):
        super().__init__(batch_size=1, regularization_method=regularization_method)
