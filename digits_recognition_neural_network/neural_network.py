import numpy as np
import random


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def dif_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


class Layer:

    def __init__(self, num_of_inputs, num_of_neurons):
        self.weights = np.random.randn(num_of_neurons, num_of_inputs)
        self.biases = np.random.randn(num_of_neurons, 1)

    def prop(self, inputs):
        """
        :param inputs: num_of_inputs * 1 array
        :return:
        """
        return sigmoid(np.dot(self.weights, inputs) + self.biases)


class NeuralNetwork:

    def __init__(self, size):
        """
        initiate a NN with random weights and biases
        :param size: len(size) : number of layers include the input layer and the output layer,
                     size[0]: number of inputs
                     size[i]: number of neurons in the ith layer (excluding inputs)
        """
        self.layers = []
        for i in range(1, len(size)):
            self.layers.append(Layer(size[i - 1], size[i]))

    def prop(self, inputs):
        for layer in self.layers:
            inputs = layer.prop(inputs)
        return inputs

    def train(self, dataset, num_of_epochs, mini_batch_size, learning_rate, validation_data=None):
        """
        train the network with a given dataset.
        :param validation_data: used to test accuracy after each epoch
        :param dataset: training data
        :param num_of_epochs: times to run through the whole training set
        :param mini_batch_size: batch size in a single run of SGD
        :param learning_rate: step length when taking stochastic gradient descent
        :return:
        """
        if validation_data:
            validation_data = list(validation_data)
        dataset = list(dataset)
        data_size = len(dataset)
        mini_batch_size = min(data_size, mini_batch_size)
        for i in range(num_of_epochs):
            random.shuffle(dataset)
            # divide mini batches for each epoch
            mini_batches = [dataset[i:i + mini_batch_size] for i in
                            range(0, data_size, mini_batch_size)]
            for mini_batch in mini_batches:
                self.process_mini_batch(mini_batch, learning_rate)
            if validation_data:
                print("completed epoch %d, accuracy %f" % ((i + 1), self.get_accuracy(validation_data)))
            else:
                print("completed epoch %d" % (i + 1))

    def process_mini_batch(self, mini_batch, learning_rate):
        """
        implement a back-propagation and gradient descent to process a mini batch
        """
        batch_size = len(mini_batch)
        num_of_layers = len(self.layers)
        for x, y in mini_batch:
            # feed forward, take a record of a's and z's for later use in back prop
            activations = [x]
            zs = []
            for layer in self.layers:
                zs.append(np.dot(layer.weights, activations[-1]) + layer.biases)
                activations.append(sigmoid(zs[-1]))
            # back propagation of errors
            deltas = [(activations[-1] - y) * dif_sigmoid(zs[-1])]
            for i in range(num_of_layers - 1):
                layer_index = -1 - i
                delta = np.dot(self.layers[layer_index].weights.T, deltas[layer_index]) * dif_sigmoid(
                    zs[layer_index - 1])
                deltas.append(delta)
            deltas = deltas[::-1]
            # calculate the partial derivatives using the deltas for each layer
            partial_ws = [np.dot(deltas[i], activations[i].T) for i in range(num_of_layers)]
            partial_bs = deltas
            # do the gradient descent, update weights and biases for each layer
            for i in range(num_of_layers):
                this_layer = self.layers[i]
                this_layer.weights = this_layer.weights - learning_rate * partial_ws[i] / batch_size
                this_layer.biases = this_layer.biases - learning_rate * partial_bs[i] / batch_size

    def get_accuracy(self, dataset):
        dataset = list(dataset)
        size = len(dataset)
        test_results = [(np.argmax(self.prop(x)), y) for (x, y) in dataset]
        accuracy = sum(int(x == y) for (x, y) in test_results) / float(size)
        return accuracy


if __name__ == "__main__":
    import digits_recognition_neural_network.mnist_loader as mnist_loader

    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = NeuralNetwork([784, 30, 10])
    net.train(training_data, 30, 10, 3.0, validation_data)
    print("posterior accuracy on test data %f" % net.get_accuracy(test_data))
