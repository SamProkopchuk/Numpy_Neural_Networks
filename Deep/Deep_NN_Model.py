import sys
sys.path.insert(1, "..")
import numpy as np
from General_NN_Functions import sigmoid, tanh, relu, derivative, get_cost
from Batch_Types import *
from Descent_Methods import *

"""
A numpy implementation of all necessary functions for
a neural network with an arbitrary number of arbitrarily sized hidden layers.
With Deep_NN_Optimizers, has the ability of using optimizers such as sgd and dropout on fitting.
"""


class DeepNNModel():

    def __init__(self, layer_sizes, funcs,
                 batch_type,         # Must be an instance
                 random_seed=None):
        if random_seed is not None:
            np.random.seed(random_seed)
        self.layer_sizes = layer_sizes
        self.funcs = funcs

        self.weights = self.initialized_weights()

        self.batch_type = batch_type

        self._batch_gradient_descent = batch_gradient_descent(
            dropout(keep_prob=1))

    def initialized_weights(self, multiplier=0.01):
        weights = {}

        for l in range(1, len(self.layer_sizes)):
            weights[f"W{l}"] = np.random.randn(self.layer_sizes[l], self.layer_sizes[l - 1]) * multiplier
            weights[f"b{l}"] = np.zeros((self.layer_sizes[l], 1))

        return weights

    def update_weights(self, grads, learning_rate):
        L = len(self.weights) // 2
        for l in range(L, 0, -1):
            self.weights[f"W{l}"] -= learning_rate * grads[f"dW{l}"]
            self.weights[f"b{l}"] -= learning_rate * grads[f"db{l}"]

    def fit(self, X, Y, learning_rate, num_iterations, cost_calc_interval):
        train_costs = []

        L = len(self.weights) // 2

        for i in range(num_iterations):
            cache = self.batch_type.forward_propagate(
                X, self.weights, self.funcs)
            grads = self.batch_type.backward_propagate(
                self.weights, cache, self.funcs, X, Y)
            self.update_weights(grads, learning_rate)

            if i % cost_calc_interval == 0:
                tempcache = self._batch_gradient_descent.forward_propagate(
                    X, self.weights, self.funcs)
                train_costs.append(get_cost(tempcache[f"A{L}"], Y))
                print(f"{round(100*i/num_iterations)}% Complete", flush=True, end="\r")
        print("100% Complete")

        return train_costs

    def predict(self, X):
        L = len(self.weights) // 2
        cache = self._batch_gradient_descent.forward_propagate(
            X, self.weights, self.funcs)

        predictions = np.amax(cache[f"A{L}"], axis=0, keepdims=True)
        predictions = np.where(cache[f"A{L}"] == predictions, 1, 0)

        return predictions
