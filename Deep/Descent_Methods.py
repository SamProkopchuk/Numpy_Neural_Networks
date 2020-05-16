import sys
sys.path.insert(1, "..")
import numpy as np
from General_NN_Functions import sigmoid, tanh, relu, derivative


class DescentMethod:
    """
    Parent class of descent method classes
    """
    EPSILON = 1e-8

    def optimized_grads(self):
        pass


class momentum(DescentMethod):
    """
    Momentum
    """

    def __init__(self, beta=0.9):
        self.beta = beta
        self.velocities = {}

        self._velocities_initialized = False

    def initialize_velocities(self, grads):
        for grad in grads:
            self.velocities[grad] = 0

    def optimized_grads(self, grads, iteration):
        if not self._velocities_initialized:
            self.initialize_velocities(grads)
            self._velocities_initialized = True

        for grad in grads:
            self.velocities[grad] = \
                self.beta * self.velocities[grad] + \
                (1 - self.beta) * grads[grad]
        return self.velocities


class RMSprop(DescentMethod):
    """
    Root Mean Square Propagation
    """

    def __init__(self, beta=0.9):
        self.beta = beta
        self.mean_sqr_grads = {}

        self._velocities_initialized = False

    def initialize_mean_sqr_grads(self, grads):
        for grad in grads:
            self.mean_sqr_grads[grad] = 0

    def optimized_grads(self, grads, iteration):
        if not self._velocities_initialized:
            self.initialize_mean_sqr_grads(grads)
            self._velocities_initialized = True

        for grad in grads:
            self.mean_sqr_grads[grad] = \
                self.beta * self.mean_sqr_grads[grad] + \
                (1 - self.beta) * np.square(grads[grad])
            grads[grad] /= \
                np.sqrt(self.mean_sqr_grads[grad] + DescentMethod.EPSILON)

        return grads


class adam(DescentMethod):
    """
    Adaptive Moment Estimation
    """

    def __init__(self, momentum_beta=0.9, rms_beta=0.9):
        self._momentum = momentum(momentum_beta)
        self._RMSprop = RMSprop(rms_beta)

    def bias_correct(self, velocities, mean_sqr_grads, iteration):
        for grad in velocities:
            velocities[grad] /= (1 - self._momentum.beta ** iteration)
            mean_sqr_grads[grad] /= (1 - self._RMSprop.beta ** iteration)

    def optimized_grads(self, grads, iteration):
        self._momentum.optimized_grads(grads, iteration)
        self._RMSprop.optimized_grads(grads, iteration)

        velocities = self._momentum.velocities
        mean_sqr_grads = self._RMSprop.mean_sqr_grads
        self.bias_correct(velocities, mean_sqr_grads, iteration)

        for grad in grads:
            grads[grad] = velocities[grad] / \
                np.sqrt(mean_sqr_grads[grad] + DescentMethod.EPSILON)

        return grads
