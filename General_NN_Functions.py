import numpy as np
import matplotlib.pyplot as plt


"""
Functions that presumably any neural network implementation will use.
"""


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def tanh(z):
    return np.tanh(z)


def relu(z):
    return np.maximum(0, z)


def derivative(f, fx):
    """
    Returns derivative of f at x given
    a known funciton f, and f(x).
    """
    if f is sigmoid:
        return fx * (1 - fx)
    elif f is tanh:
        return 1 - np.square(fx)
    elif f is relu:
        return np.where(fx == 0, 0, 1)
    else:
        raise Exception(f"Unknown function: {f}")


def get_cost(AL, Y):
    m = Y.shape[1]
    losses = np.multiply(np.log(AL), Y) + np.multiply(np.log(1 - AL), 1 - Y)
    cost = -np.sum(losses) / m
    cost = np.squeeze(cost)
    return cost


def graph_costs(costs, x_label=None, y_label=None):
    plt.plot(range(len(costs)), costs)
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)
    plt.show()


def evaluate_model(predictions, Y):
    """
    Assume predictions has only one 1 per column
    Count when both predictions and Y are 1
    divided by the number of data vectors in Y.
    """
    # Ensure every column has exactly one 1:
    assert(np.count_nonzero(np.where(np.count_nonzero(
        predictions, axis=0) == 1, 1, 0)) == Y.shape[1])
    return np.count_nonzero(np.logical_and(predictions, Y)) / Y.shape[1]
