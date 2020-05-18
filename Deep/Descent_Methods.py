import numpy as np


class DescentMethod:
    """
    Parent class of descent method classes
    """
    EPSILON = 1e-8

    def optimized_grads(self):
        pass


class Momentum(DescentMethod):
    """
    Momentum
    """

    def __init__(self, beta=0.9):
        self.beta = beta
        self.velocities = {}

        self._velocities_initialized = False

    def initialize_velocities(self, grads, L):
        L = len(grads) // 2
        for l in range(1, L + 1):
            self.velocities[f"dW{l}"] = 0.
            self.velocities[f"db{l}"] = 0.

    def optimized_grads(self, grads):
        if not self._velocities_initialized:
            self.initialize_velocities(grads)
            self._velocities_initialized = True

        L = len(grads) // 2
        for l in range(1, L + 1):
            self.velocities[f"dW{l}"] = \
                self.beta * self.velocities[f"dW{l}"] + \
                (1 - self.beta) * grads[f"dW{l}"]
            self.velocities[f"db{l}"] = \
                self.beta * self.velocities[f"db{l}"] + \
                (1 - self.beta) * grads[f"db{l}"]
        return self.velocities


class RMSProp(DescentMethod):
    """
    Root Mean Square Propagation
    """

    def __init__(self, beta=0.9):
        self.beta = beta
        self.mean_sqr_grads = {}

        self._velocities_initialized = False

    def initialize_mean_sqr_grads(self, grads):
        L = len(grads) // 2
        for l in range(1, L + 1):
            self.mean_sqr_grads[f"dW{l}"] = 0.
            self.mean_sqr_grads[f"db{l}"] = 0.

    def optimized_grads(self, grads):
        if not self._velocities_initialized:
            self.initialize_mean_sqr_grads(grads)
            self._velocities_initialized = True

        L = len(grads) // 2
        for l in range(1, L + 1):
            self.mean_sqr_grads[f"dW{l}"] = \
                self.beta * self.mean_sqr_grads[f"dW{l}"] + \
                (1 - self.beta) * np.square(grads[f"dW{l}"])
            self.mean_sqr_grads[f"db{l}"] = \
                self.beta * self.mean_sqr_grads[f"db{l}"] + \
                (1 - self.beta) * np.square(grads[f"db{l}"])

            grads[f"dW{l}"] /= \
                np.sqrt(self.mean_sqr_grads[f"dW{l}"]) + DescentMethod.EPSILON
            grads[f"db{l}"] /= \
                np.sqrt(self.mean_sqr_grads[f"db{l}"]) + DescentMethod.EPSILON

        return grads


class Adam(DescentMethod):
    """
    Adaptive Moment Estimation
    """

    def __init__(self, beta1=0.9, beta2=0.999):
        self.iteration = 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.velocities = {}
        self.mean_sqr_grads = {}

        self._mean_terms_initialized = False

    def initialize_mean_terms(self, grads):
        L = len(grads) // 2

        for l in range(1, L + 1):
            self.velocities[f"dW{l}"] = 0
            self.velocities[f"db{l}"] = 0
            self.mean_sqr_grads[f"dW{l}"] = 0
            self.mean_sqr_grads[f"db{l}"] = 0

    def set_velocities(self, grads):
        L = len(grads) // 2
        for l in range(1, L + 1):
            self.velocities[f"dW{l}"] = \
                self.beta1 * self.velocities[f"dW{l}"] + \
                (1 - self.beta1) * grads[f"dW{l}"]
            self.velocities[f"db{l}"] = \
                self.beta1 * self.velocities[f"db{l}"] + \
                (1 - self.beta1) * grads[f"db{l}"]

    def set_mean_sqr_grads(self, grads):
        L = len(grads) // 2
        for l in range(1, L + 1):
            self.mean_sqr_grads[f"dW{l}"] = \
                self.beta2 * self.mean_sqr_grads[f"dW{l}"] + \
                (1 - self.beta2) * np.square(grads[f"dW{l}"])
            self.mean_sqr_grads[f"db{l}"] = \
                self.beta2 * self.mean_sqr_grads[f"db{l}"] + \
                (1 - self.beta2) * np.square(grads[f"db{l}"])

    def bias_correct(self, grads):
        L = len(grads) // 2
        for l in range(1, L+1):
            self.velocities[f"dW{l}"] /= 1 - self.beta1 ** self.iteration
            self.velocities[f"db{l}"] /= 1 - self.beta1 ** self.iteration
            self.mean_sqr_grads[f"dW{l}"] /= 1 - self.beta2 ** self.iteration
            self.mean_sqr_grads[f"db{l}"] /= 1 - self.beta2 ** self.iteration

    def optimized_grads(self, grads):
        if not self._mean_terms_initialized:
            self.initialize_mean_terms(grads)

        self.iteration += 1

        self.set_velocities(grads)
        self.set_mean_sqr_grads(grads)
        self.bias_correct(grads)

        L = len(grads) // 2
        for l in range(1, L+1):
            self.velocities[f"dW{l}"] /= \
                np.sqrt(self.mean_sqr_grads[f"dW{l}"]) + DescentMethod.EPSILON
            self.velocities[f"db{l}"] /= \
                np.sqrt(self.mean_sqr_grads[f"db{l}"]) + DescentMethod.EPSILON

        return self.velocities
