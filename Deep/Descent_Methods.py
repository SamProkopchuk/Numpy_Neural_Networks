import sys
sys.path.insert(1, "..")
import numpy as np
from General_NN_Functions import sigmoid, tanh, relu, derivative

class DescentMethod:
    """
    Parent class of descent method classes
    """
    pass


class RMSprop(DescentMethod):

    def __init__(self, beta=0.9):
        self.beta = beta
        self.prev_grads = 0

    def update_grads(self):
        pass


class Adam(DescentMethod):
    def __init__(self):
        pass