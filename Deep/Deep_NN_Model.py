import sys
sys.path.insert(1, "..")
import numpy as np
from General_NN_Functions import sigmoid, tanh, relu, derivative, get_cost
from Batch_Methods import *
from Regularization_Methods import *
from Descent_Methods import *

"""
A numpy implementation of all necessary functions for
a neural network with an arbitrary number of arbitrarily sized hidden layers.
With Deep_NN_Optimizers, has the ability of using optimizers such as sgd and dropout on fitting.
"""


class DeepNNModel():

    def __init__(self, layer_sizes, funcs,
                 batch_method: BatchMethod,
                 regularization_method: RegularizationMethod,
                 descent_method: DescentMethod = None,
                 random_seed=None):
        if random_seed is not None:
            np.random.seed(random_seed)
        self.layer_sizes = layer_sizes
        self.funcs = funcs

        self.weights = self.initialized_weights()

        self.batch_method = batch_method
        self.regularization_method = regularization_method
        self.descent_method = descent_method

        self._unregularized_prop = l2(lambd=0)

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

        for i in range(1, num_iterations+1):
            Xbatch, Ybatch = self.batch_method.get_batch(X, Y)
            cache = self.regularization_method.forward_propagate(
                Xbatch, self.weights, self.funcs)
            grads = self.regularization_method.backward_propagate(
                self.weights, cache, self.funcs, Xbatch, Ybatch)

            if self.descent_method is not None:
                grads = self.descent_method.optimized_grads(grads, i)
            self.update_weights(grads, learning_rate)

            if i % cost_calc_interval == 0:
                tempcache = self._unregularized_prop.forward_propagate(
                    X, self.weights, self.funcs)
                train_costs.append(get_cost(tempcache[f"A{L}"], Y))
                print(f"{round(100*i/num_iterations)}% Complete", flush=True, end="\r")
        print("100% Complete")

        return train_costs

    def predict(self, X):
        L = len(self.weights) // 2
        cache = self._unregularized_prop.forward_propagate(
            X, self.weights, self.funcs)

        predictions = np.amax(cache[f"A{L}"], axis=0, keepdims=True)
        predictions = np.where(cache[f"A{L}"] == predictions, 1, 0)

        return predictions
