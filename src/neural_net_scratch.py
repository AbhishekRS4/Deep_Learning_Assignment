import argparse
import numpy as np
from tqdm import tqdm
from dataset import load_gq_data, split_data
from visualization_utils import save_plot_losses
from metrics import compute_test_metrics, compute_accuracy


class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        # TODO: return output
        pass

    def backward(self, output_gradient, learning_rate):
        # TODO: update parameters and return input gradient
        pass


class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient


class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))


class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            return sigmoid(x) * (1. - sigmoid(x))

        super().__init__(sigmoid, sigmoid_prime)

class ReLu(Activation):
    def __init__(self):
        def relu(x):
            self.inputs = x
            return np.maximum(0, x)

        def relu_prime(x):
            return (x > 0) * 1

        super().__init__(relu, relu_prime)

"""
class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def tanh_prime(x):
            return 1 - np.tanh(x) ** 2

        super().__init__(tanh, tanh_prime)
"""


class BinaryCrossEntropy:
    def forward(self, y_true, y_pred, smooth=1e-6):
        loss = - ((y_true * np.log2(y_pred + smooth)) + ((1. - y_true) * np.log2(1. - y_pred + smooth)))
        loss = np.mean(loss)
        return loss

    def backward(self, y_true, y_pred, smooth=1e-6):
        return ((y_pred - y_true) / (y_pred * (1 - y_pred) + smooth)) / np.size(y_true)

"""
class MSE:
    def forward(self, y_true, y_pred):
        return np.mean(np.power(y_true - y_pred, 2))

    def backward(self, y_true, y_pred):
        return -2 * (y_true-y_pred) / np.size(y_true)
"""

class Network:
    def __init__(self):
        self.layers = []

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # set loss to use
    def use(self):
        self.loss = BinaryCrossEntropy()

    # predict output for given input
    def predict(self, input_data):
        # sample dimension first
        num_train_samples = len(input_data)

        # run network over all num_train_samples
        for i in range(num_train_samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward(output)
        return output

    # train the network
    def fit(self, X_train, Y_train, epochs, learning_rate, X_valid=None, Y_valid=None):
        # sample dimension first
        num_train_samples = len(X_train)
        losses = {"train":None, "valid":None}
        train_losses = []
        valid_losses = []

        # training loop
        for i in range(epochs):
            #-------#
            # Train #
            #-------#
            train_loss = 0
            for j in tqdm(range(num_train_samples)):
                # forward propagation
                output = X_train[j]
                for layer in self.layers:
                    output = layer.forward(output)
                # compute loss (for display purpose only)
                #print(X_train, output)
                train_loss += self.loss.forward(Y_train[j], output)

                # backward propagation
                train_loss_backward = self.loss.backward(Y_train[j], output)
                for layer in reversed(self.layers):
                    #print(error)
                    train_loss_backward = layer.backward(train_loss_backward, learning_rate)
                    #print(error.shape)
            # calculate average error on all num_train_samples
            train_loss /= num_train_samples
            train_losses.append(train_loss)

            #------------#
            # Validation #
            #------------#
            if X_valid is not None:
                valid_loss = 0
                num_valid_samples = len(X_valid)
                for k in tqdm(range(num_valid_samples)):
                    output = X_valid[k]
                    for layer in self.layers:
                        output = layer.forward(output)
                    valid_loss += self.loss.forward(Y_valid[k], output)

                valid_loss /= num_valid_samples
                valid_losses.append(valid_loss)

                print(f'epoch: {i+1} / {epochs} train loss: {train_loss:.5f}, valid loss: {valid_loss:.5f}')
            else:
                print(f'epoch: {i+1} / {epochs} train loss: {train_loss:.5f}')
        losses["train"] = train_losses
        losses["valid"] = valid_losses
        return self.layers, losses


def preprocess_data(x, y):
    # reshape and normalize input data
    x = x.reshape(x.shape[0], 2, 1)
    y = y.reshape(y.shape[0], 1, 1)
    return x, y

def build_network(num_neurons_input=2, num_hidden_layers=2, num_neurons_hidden=10, num_neurons_output=1):
    # network
    net = Network()

    # input + 1 hidden layer
    net.add(Dense(num_neurons_input, num_neurons_hidden))
    net.add(ReLu())

    # hidden layers
    for i in range(num_hidden_layers-1):
        net.add(Dense(num_neurons_hidden, num_neurons_hidden))
        net.add(ReLu())

    # output layer
    net.add(Dense(num_neurons_hidden, num_neurons_output))
    net.add(Sigmoid())

    # add loss for training
    net.use()
    return net

# train
def train(network, X_train, Y_train, X_valid, Y_valid, epochs, lr_rate):
    network, losses = network.fit(X_train, Y_train, epochs=epochs, learning_rate=lr_rate, X_valid=X_valid, Y_valid=Y_valid)
    return network, losses

# predict
def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output

# test
def test(network, X_test, y_test):
    # test
    acc = 0
    num_samples = len(X_test)
    pred_test_labels = []

    for x, y in zip(X_test, y_test):
        output = 1 if predict(network, x) > 0.5 else 0
        pred_test_labels.append(output)

    return np.array(pred_test_labels)

# start training the model
def start_model_training(FLAGS):
    # load Gaussian quantile dataset
    gq_data = load_gq_data()
    X = gq_data[0]
    Y = gq_data[1]

    # split the dataset into train (80%) and test (20%)
    X_train, X_test, Y_train, Y_test = split_data(X, Y, test_size=0.2)

    # further split test dataset into test (72%) and validation (8%)
    X_train, X_valid, Y_train, Y_valid = split_data(X_train, Y_train, test_size=0.1)
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_valid, Y_valid = preprocess_data(X_valid, Y_valid)
    X_test, Y_test = preprocess_data(X_test, Y_test)

    # build the neural network model
    network = build_network(num_hidden_layers=FLAGS.num_hidden_layers)
    network, losses = train(network, X_train, Y_train, X_valid, Y_valid, epochs=FLAGS.num_epochs, lr_rate=FLAGS.learning_rate)

    # if needed save the plot of train and validation losses
    if FLAGS.save_plot:
        save_plot_losses(losses)

    # test and compute metrics for the test set
    pred_valid_labels = test(network, X_valid, Y_valid)
    valid_acc = compute_accuracy(np.squeeze(Y_valid), pred_valid_labels)
    print(f"Validation accuracy : {valid_acc:.4f}")

    pred_test_labels = test(network, X_test, Y_test)
    test_acc, test_cm, test_f1 = compute_test_metrics(np.squeeze(Y_test), pred_test_labels)
    print("\n---------------")
    print("Test metrics")
    print("---------------")
    print(f"test_accuracy_score: {test_acc:.4f}, test_f1_score: {test_f1:.4f}")
    print(f"test confusion matrix")
    print(test_cm)

    return

def main():
    learning_rate = 3e-3
    num_epochs = 100
    num_hidden_layers = 3
    save_plot = 0

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--learning_rate", default=learning_rate,
        type=float, help="learning rate")
    parser.add_argument("--num_epochs", default=num_epochs,
        type=int, help="number of epochs to train")
    parser.add_argument("--num_hidden_layers", default=num_hidden_layers,
        type=int, help="number of hidden layers in the model")
    parser.add_argument("--save_plot", default=save_plot,
        type=int, choices=[0, 1], help="flag to save plot of losses")

    FLAGS, unparsed = parser.parse_known_args()
    start_model_training(FLAGS)

if __name__ == '__main__':
    main()
