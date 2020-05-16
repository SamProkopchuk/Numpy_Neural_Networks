import numpy as np
import sys
sys.path.insert(1, "..")
from General_NN_Functions import sigmoid, tanh, relu, derivative


class RegularizationMethod:
    """
    Parent class of regularization method classes
    """

    def forward_propagate(self):
        pass

    def forward_propagate(self):
        pass


class Unregularized:
    """
    Unregularized class for simple unregularized
    forward propagation and backpropagation
    """

    def forward_propagate(self, X, weights, funcs):
        cache = {"A0": X}
        for l in range(1, 1 + len(weights) // 2):
            cache[f"Z{l}"] = np.dot(weights[f"W{l}"], cache[f"A{l-1}"]) + weights[f"b{l}"]
            cache[f"A{l}"] = funcs[f"L{l}_func"](cache[f"Z{l}"])
        return cache

    def backward_propagate(self, weights, cache, funcs, X, Y):
        m = X.shape[1]
        L = len(weights) // 2
        grads = {}

        dZl = cache[f"A{L}"] - Y
        grads[f"dW{L}"] = np.dot(dZl, cache[f"A{L-1}"].T) / m
        grads[f"db{L}"] = np.sum(dZl, axis=1, keepdims=True) / m
        for l in range(L - 1, 0, -1):
            funcl = funcs[f"L{l}_func"]
            dZl = np.dot(weights[f"W{l+1}"].T, dZl) * derivative(
                f=funcl, fx=funcl(cache[f"Z{l}"]))
            grads[f"dW{l}"] = np.dot(dZl, cache[f"A{l-1}"].T)
            grads[f"db{l}"] = np.sum(dZl, axis=1, keepdims=True) / m
        return grads


class L2(RegularizationMethod):
    """
    l2 regularization

    __init__ params:
        lambd -> the l2 regularization parameter.
            Multiplies with sum of the squares of each weight.
    """

    def __init__(self, lambd):
        self.lambd = lambd

    def forward_propagate(self, X, weights, funcs):
        cache = {"A0": X}
        for l in range(1, 1 + len(weights) // 2):
            cache[f"Z{l}"] = np.dot(weights[f"W{l}"], cache[f"A{l-1}"]) + weights[f"b{l}"]
            cache[f"A{l}"] = funcs[f"L{l}_func"](cache[f"Z{l}"])
        return cache

    def backward_propagate(self, weights, cache, funcs, X, Y):
        m = X.shape[1]
        L = len(weights) // 2
        grads = {}

        dZl = cache[f"A{L}"] - Y
        grads[f"dW{L}"] = (np.dot(dZl, cache[f"A{L-1}"].T) + self.lambd * weights[f"W{L}"]) / m
        grads[f"db{L}"] = np.sum(dZl, axis=1, keepdims=True) / m
        for l in range(L - 1, 0, -1):
            funcl = funcs[f"L{l}_func"]
            dZl = np.dot(weights[f"W{l+1}"].T, dZl) * derivative(
                f=funcl, fx=funcl(cache[f"Z{l}"]))
            grads[f"dW{l}"] = (np.dot(dZl, cache[f"A{l-1}"].T) + self.lambd * weights[f"W{l}"]) / m
            grads[f"db{l}"] = np.sum(dZl, axis=1, keepdims=True) / m
        return grads


class Dropout(RegularizationMethod):
    """
    Dropout

    __init__ params:
        keep_prob -> probability of weights being kept.
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

        dZl = cache[f"A{L}"] - Y
        grads[f"dW{L}"] = np.dot(dZl, cache[f"A{L-1}"].T) / m
        grads[f"db{L}"] = np.sum(dZl, axis=1, keepdims=True) / m
        for l in range(L - 1, 0, -1):
            funcl = funcs[f"L{l}_func"]
            Al = np.dot(weights[f"W{l+1}"].T, dZl) * self.drops[f"D{l}"] / self.keep_prob
            dZl = Al * derivative(
                f=funcl, fx=funcl(cache[f"Z{l}"]))
            grads[f"dW{l}"] = np.dot(dZl, cache[f"A{l-1}"].T) / m
            grads[f"db{l}"] = np.sum(dZl, axis=1, keepdims=True) / m

        return grads
