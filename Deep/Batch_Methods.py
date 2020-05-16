import numpy as np


class BatchMethod:
    """
    Parent class of batch type classes
    """

    def get_batch(self):
        pass


class Batch(BatchMethod):

    def __init__(self):
        pass

    def get_batch(self, X, Y):
        return X, Y


class MiniBatch(BatchMethod):
    """
    Mini Batch Gradient Descent

    Note that for train index to update, backward_propagate must be called.
    """

    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.train_index = 0

    def get_res(self, train_cases):
        return max(0, (self.train_index + self.batch_size) - train_cases)

    def update_train_index(self, train_cases):
        res = self.get_res(train_cases)
        self.train_index = res if res != 0 else self.train_index + self.batch_size

    def get_mini_batch(self, A):
        A1 = A[:, self.train_index:self.train_index + self.batch_size]
        res = self.get_res(A.shape[1])
        if res:
            A2 = A[:, 0:res:]
            return np.c_[A1, A2]
        else:
            return A1

    def get_batch(self, X, Y):
        self.update_train_index(X.shape[1])
        return self.get_mini_batch(X), self.get_mini_batch(Y)


class SGD(MiniBatch):

    def __init__(self):
        super().__init__(batch_size=1)
