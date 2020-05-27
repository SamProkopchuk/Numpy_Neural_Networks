from sys import path
path.insert(1, '..')
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits

from General_NN_Functions import relu, tanh, sigmoid, evaluate_model, graph_costs
from Batch_Methods import *
from Regularization_Methods import *
from Descent_Methods import *
from Deep_NN_Model import DeepNNModel

"""
Run an example of a deep neural network, tuned to one's liking.
Optmizers can be used and all micro weights can be manually changed.
"""

# START MICRO-parameters:
# The basic parameters and micro-parameters for a deep NN:
LEARNING_RATE = 0.001
NUM_ITERATIONS = 1000
LAYER_SIZES = (64, 37, 37, 10)
# (Input layer and last layer must match X, Y dimentions)

# Num of iterations between which cost is calculated,
# (Also the print % Complete interval):
COST_CALC_INTERVAL = NUM_ITERATIONS // 100

# The functions for every layer:
FUNCS = {f"L{i}_func": relu for i in range(1, len(LAYER_SIZES) - 1)}
FUNCS[f"L{len(LAYER_SIZES)-1}_func"] = sigmoid
# END MICRO-parameters

# Optimizer functions:
BATCH_METHOD = MiniBatch(batch_size=128)
REGULARIZATION_METHOD = Dropout(keep_prob=0.8)
DESCENT_METHOD = Adam()

RANDOM_SEED = 10


def main():
    """
    8x8 MNIST is kind of simple for a deep NN, maybe try a different dataset...
    """
    data = load_digits()
    X_orig = data.data
    Y_orig = data.target.reshape(-1, 1)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_orig, Y_orig, test_size=0.2, random_state=12)

    # Scale X:
    X_train = X_train / 16 - 0.5
    X_test = X_test / 16 - 0.5

    # Make Y Binarized:
    lb = LabelBinarizer()
    Y_train = lb.fit_transform(Y_train)
    Y_test = lb.transform(Y_test)

    # Transpose X & Y as required by the model
    X_train, X_test = X_train.T, X_test.T
    Y_train, Y_test = Y_train.T, Y_test.T

    deepNNModel = DeepNNModel(
        layer_sizes=LAYER_SIZES,
        initialization="he",
        funcs=FUNCS,
        batch_method=BATCH_METHOD,
        regularization_method=REGULARIZATION_METHOD,
        descent_method=DESCENT_METHOD,
        random_seed=RANDOM_SEED)

    train_costs = deepNNModel.fit(
        X_train, Y_train,
        learning_rate=LEARNING_RATE,
        num_iterations=NUM_ITERATIONS,
        cost_calc_interval=COST_CALC_INTERVAL)

    graph_costs(
        train_costs,
        x_label=f"{COST_CALC_INTERVAL}s of iterations",
        y_label="Mean loss (Cost)")

    train_predictions = deepNNModel.predict(X_train)
    test_predictions = deepNNModel.predict(X_test)

    train_accuracy = evaluate_model(train_predictions, Y_train)
    test_accuracy = evaluate_model(test_predictions, Y_test)

    print(f"The model acheived {train_accuracy*100:.2f}% accuracy on the train set.")
    print(f"The model acheived {test_accuracy*100:.2f}% accuracy on the test set.")


if __name__ == "__main__":
    main()
