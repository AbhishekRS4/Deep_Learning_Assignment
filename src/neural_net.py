import numpy as np
from tqdm import tqdm
from dataset import load_gq_data, split_data

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


class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def tanh_prime(x):
            return 1 - np.tanh(x) ** 2

        super().__init__(tanh, tanh_prime)



class BinaryCrossEntropy:
    def forward(self, y_true, y_pred):
        loss = - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        loss = np.mean(loss)
        return loss

    def backward(self, y_true, y_pred):
        return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)


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
            for j in tqdm(range(samples)):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward(output)
                # compute loss (for display purpose only)
                #print(x_train, output)
                err += self.loss.forward(y_train[j], output)

                # backward propagation
                error = self.loss.backward(y_train[j], output)
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
    x = x.reshape(x.shape[0], 2, 1)
    y = y.reshape(y.shape[0], 1, 1)
    return x, y


# training set
def train(X_train, Y_train, epochs, lr_rate):
    # network
    net = Network()
    net.add(Dense(2, 10))
    net.add(ReLu())
    net.add(Dense(10, 1))
    net.add(Sigmoid())

    # train
    net.use()
    network = net.fit(X_train, Y_train, epochs=epochs, learning_rate=lr_rate)
    return network

def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output

def test(network, X_test, y_test):
    # test
    err = 0
    samples = len(X_test)
    for x, y in zip(X_test, y_test):
        output = 1 if predict(network, x) > 0.5 else 0
        if output != y:
            err +=1
    return err / samples

def main():
    gq_data = load_gq_data()
    X = gq_data[0]
    Y = gq_data[1]

    X_train, X_test, Y_train, Y_test = split_data(X, Y, test_size=0.2)
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_test, Y_test = preprocess_data(X_test, Y_test)

    network = train(X_train, Y_train, epochs=100, lr_rate=0.001)

    error = test(network, X_test, Y_test)

    print('Test Error:', error)

if __name__ == '__main__':
    main()
