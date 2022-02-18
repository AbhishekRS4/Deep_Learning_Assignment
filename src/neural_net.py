import numpy as np
from dataset import load_gq_data, split_data

def one_hot(Y):
    one_hot_Y= np.zeros((len(Y), max(Y) + 1))
    one_hot_Y[np.arange(len(Y)), Y] = 1
    return one_hot_Y

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
            self.inputs = x
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


class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def tanh_prime(x):
            return 1 - np.tanh(x) ** 2

        super().__init__(tanh, tanh_prime)

class Softmax(Layer):
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs))
        self.output = exp_values / np.sum(exp_values)
        #print(self.output)
        return self.output

    def backward(self, output_gradient, learning_rate):
        # This version is faster than the one presented in the video
        n = np.size(self.output)
        return np.dot((np.identity(n) - self.output.T) * self.output, output_gradient)


class BinaryCrossEntropy:
    def forward(self, y_true, y_pred):
        loss = - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        loss = np.mean(loss)
        return loss

    def backward(self, y_true, y_pred, prev_output):
        dl_dw = (y_pred - y_true) * prev_output
        return dl_dw

class CategoricalCrossEntropy:
    def forward(self, y_pred, y_true):
        #samples = len(y_pred)
        #print(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-3, 1-1e-3)

        neg_log_likelihood = -np.log(np.sum(y_pred_clipped*y_true))

        return np.mean(neg_log_likelihood)

    def backward(self, y_pred, y_true):
        #print(y_pred, y_true)
        return y_true - y_pred

class MSE:
    def forward(self, y_true, y_pred):
        return np.mean(np.power(y_true - y_pred, 2))

    def backward(self, y_true, y_pred):
        return -2 * (y_true-y_pred) / np.size(y_true)

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
        samples = len(input_data)

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward(output)
        return output

    # train the network
    def fit(self, x_train, y_train, epochs, learning_rate):
        # sample dimension first
        samples = len(x_train)

        # training loop
        for i in range(epochs):
            err = 0
            for j in range(samples):
                # forward propagation
                output = x_train[j]
                prev_output = output
                for layer in self.layers:
                    prev_output = output
                    output = layer.forward(output)
                # compute loss (for display purpose only)
                #print(x_train, output)
                err += self.loss.forward(y_train[j], output)

                # backward propagation
                error = self.loss.backward(y_train[j], output, prev_output)
                for layer in reversed(self.layers):
                    #print(error)
                    error = layer.backward(error, learning_rate)
                    #print(error.shape)
            # calculate average error on all samples
            err /= samples
            print('epoch %d/%d   error=%f' % (i+1, epochs, err))
        return self.layers

def preprocess_data(x, y):
    # reshape and normalize input data
    x = x.reshape(x.shape[0], 240, 1)
    # encode output which is a number in range [0,9] into a vector of size 10
    # e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    y = one_hot(y)
    y = y.reshape(y.shape[0], 10, 1)
    return x, y

# training set
def train(X_train, Y_train):
    # network
    net = Network()
    net.add(Dense(2, 10))
    net.add(ReLu())
    net.add(Dense(10, 1))
    net.add(Sigmoid())

    # train
    net.use()
    network = net.fit(X_train, Y_train, epochs=100, learning_rate=0.001)
    return network

def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output

def test():
    # test
    err = 0
    samples = len(X_test)
    for x, y in zip(X_test, y_test):
        output = predict(network, x)
        if np.argmax(output) == np.argmax(y):
            err += 1
    err/=samples
    print('Test Error:', err)

def main():
    gq_data = load_gq_data()
    X = gq_data[0]
    Y = gq_data[1]

    X_train, X_test, Y_train, Y_test = split_data(X, Y, test_size=0.2)
    #X_train, y_train = preprocess_data(X_train, y_train)
    #X_test, y_test = preprocess_data(X_test, y_test)

    network = train(X_train, Y_train)

    predict(network, X_test)

main()
