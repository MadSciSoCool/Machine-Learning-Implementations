import numpy as np
import random


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def dif_sigmoid(x):
    return sigmoid(x) * (1. - sigmoid(x))


class Layer:
    def __init__(self, last_layer=None):
        self.last_layer = last_layer
        self._activations = None

    def get_activations(self):
        return self._activations


class InputLayer(Layer):
    def __init__(self, output_size):
        super().__init__(last_layer=None)
        self.output_size = output_size
        self._activations = np.zeros(output_size)

    def feed_forward(self, inputs):
        self._activations = np.reshape(inputs, self.output_size)
        return self._activations


class ConvolutionalLayer(Layer):

    def __init__(self, im_size, kernel_size, last_layer):
        """
        :param im_size: (C, H, W): Channels(RGB = 3, Gray Scale=1), Height, Width
        :param kernel_size: (O, C ,kh, kw): Outputs, Channels(Same as image), kernel_height, kernel width
        """
        super().__init__(last_layer)
        self.im_size = im_size
        self.kernel_size = kernel_size
        C, H, W = self.im_size
        O, C, kh, kw = self.kernel_size
        output_size = (O, H - kh + 1, W - kw + 1)
        self.kernels = np.random.randn(*kernel_size)
        self.biases = np.random.randn(*output_size)
        # only used in training
        self._col = np.zeros((C, H - kh + 1, W - kw + 1, kh, kw))
        self._z = np.zeros(output_size)
        self._activations = np.zeros(output_size)

    def feed_forward(self, inputs):
        # col shape (C, H - kh + 1, W - kw + 1, kh, kw)
        # kernel shape (O, C ,kh, kw)
        # output shape (O, H - kh + 1, W - kw + 1)
        axes = [(1, 2, 3), (0, 3, 4)]
        self._z = np.tensordot(self.kernels, self._ff_im2col(inputs), axes=axes) + self.biases
        self._activations = sigmoid(self._z)
        return self._activations

    def _ff_im2col(self, im):
        im_strides = im.strides
        strides = (*im_strides, *im_strides[-2:])
        shape = self._col.shape
        self._col = np.lib.stride_tricks.as_strided(im, shape=shape, strides=strides)
        return self._col

    def back_prop(self, error):
        """
        back-prop the error from convolution result to image (no activation function involved)
        :param error: error should be the same shape as outputs, i.e. (O, H - kh + 1, W - kw + 1)
        :return:
        """
        # pad the error matrix with zeros: (O, H - kh + 1, W - kw + 1) -> (O, H + kh - 1, W + kw - 1)
        O, C, kh, kw = self.kernel_size
        padded = np.array([np.pad(m, (2 * (kh + 1), 2 * (kw + 1))) for m in error])
        # filp the kernel
        flipped_kernels = self.kernels[:, :, ::-1, ::-1]
        # do tensordot(flipped_kernels, im2col(padded)), the shapes are
        # (O, C, kh, kw) . (O, H, W, kh, kw) -> (C, H, W)
        return np.tensordot(flipped_kernels, self._bp_im2col(padded), axes=[(0, 2, 3), (0, 3, 4)])

    def _bp_im2col(self, im):
        # for the convolution in back-propagation, convert
        # (O, H + kh - 1, W + kw - 1) -> (O, H, W, kh, kw)
        C, H, W = self.im_size
        O, C, kh, kw = self.kernel_size
        im_strides = im.strides
        strides = (*im_strides, *im_strides[-2:])
        shape = (O, H, W, kh, kw)
        return np.lib.stride_tricks.as_strided(im, shape=shape, strides=strides)

    def get_kernels_error(self, error):
        """
        get the error (gradient) for kernels from the error back-propagated from the next layer
        """
        # error: (O, H - kh + 1, W - kw + 1) self._col: (C, H - kh + 1, W - kw + 1, kh, kw)
        return np.tensordot(error, self._col, axes=[(1, 2), (1, 2)])

    def get_biases_error(self, error):
        return error

    def update(self, error, learning_rate):
        self.kernels = self.kernels - learning_rate * self.get_kernels_error(error)
        self.biases = self.biases - learning_rate * self.get_biases_error(error)

    def back_prop_activation(self, error):
        # this function back-propagates the error across the activation function
        return error * dif_sigmoid(self._z)


class MaxPoolingLayer(Layer):

    def __init__(self, input_size, pool_size, last_layer):
        """
        :param input_size: the input is from convolutional layer, so input size is
                           (O, H, W) = (O, H' - kh + 1, W' - kw + 1)
        :param pool_size: (ph, pw), should be divisible by (H - kh + 1, W - kw + 1)
        """
        super().__init__(last_layer)
        self.input_size = input_size
        self.pool_size = pool_size
        O, H, W = self.input_size
        ph, pw = self.pool_size
        # only used in training
        self._max_index = np.zeros(O * (H // ph) * (W // pw))
        self._activations = np.zeros((O, H // ph, W // pw))

    def feed_forward(self, inputs):
        col = self._im2col(inputs)
        ph, pw = self.pool_size
        self._max_index = [np.argmax(block) for block in np.reshape(col, (len(self._max_index), ph, pw))]
        self._activations = np.max(col, axis=(3, 4))
        return self._activations

    def back_prop(self, error):
        """
        :param error: size (O, H // ph, W // pw)
        :return: size (O, H, W)
        """
        flattened = np.array([self._pad(element, index) for element, index in zip(np.nditer(error), self._max_index)])
        return np.reshape(flattened, self.input_size)

    def _pad(self, element, index):
        padded = np.zeros(self.pool_size)
        padded[np.unravel_index(index, self.pool_size)] = element
        return padded

    def _im2col(self, im):
        """
        here we do the same trick as in the convolutional layer, but take each stride as large as a pool height or width
        :param im: size (O, H, W) from convolutional layer so  H, W = H' - kh + 1, W' - kw + 1
        :return: size (O, H // ph, W // pw, ph, pw)
        """
        O, H, W = im.shape
        s0, s1, s2 = im.strides
        ph, pw = self.pool_size
        strides = (s0, s1 * ph, s2 * pw, s1, s2)
        shape = (O, H // ph, W // pw, ph, pw)
        return np.lib.stride_tricks.as_strided(im, shape=shape, strides=strides)


class FullyConnectedLayer(Layer):
    def __init__(self, input_size, neuron_size, last_layer):
        super().__init__(last_layer)
        self.input_size = input_size
        self.neuron_size = neuron_size
        input_size = np.prod(input_size)
        self.weights = np.random.randn(neuron_size, input_size)
        self.biases = np.random.randn(neuron_size, 1)
        # only used in training
        self._z = np.zeros((neuron_size, 1))
        self._activations = np.zeros((neuron_size, 1))

    def feed_forward(self, inputs):
        inputs = inputs.flatten()
        self._z = np.reshape(np.dot(self.weights, inputs), (self.neuron_size, 1)) + self.biases
        self._activations = sigmoid(self._z)
        return self._activations

    def back_prop(self, z):
        # note that this function only back-propagates the error from z(l + 1) to a(l)
        return np.reshape(np.dot(self.weights.T, z), self.input_size)

    def back_prop_activation(self, error):
        # this function back-propagates the error across the activation function
        return np.reshape(error, (self.neuron_size, 1)) * dif_sigmoid(self._z)

    def get_weights_error(self, error):
        input_size = np.prod(self.input_size)
        last_activations = np.reshape(self.last_layer.get_activations(), (input_size, 1))
        return np.dot(error, last_activations.T)

    def get_biases_error(self, error):
        return error

    def update(self, error, learning_rate):
        self.weights = self.weights - learning_rate * self.get_weights_error(error)
        self.biases = self.biases - learning_rate * self.get_biases_error(error)


class Layers:
    def __init__(self):
        self._layers = []

    def get_last_layer(self):
        if len(self._layers):
            return self._layers[-1]

    def append_layer(self, layer):
        self._layers.append(layer)

    def feed_forward(self, inputs):
        for layer in self._layers:
            inputs = layer.feed_forward(inputs)
        return inputs

    def back_prop(self, error, learning_rate):
        for layer in self._layers[::-1]:
            if layer.last_layer:
                if hasattr(layer, "back_prop_activation"):
                    error = layer.back_prop_activation(error)
                if hasattr(layer, "update"):
                    layer.update(error, learning_rate)
                error = layer.back_prop(error)


class ConvolutionalNeuralNetwork:
    def __init__(self):
        # here we define the structure of our neural network
        self.layers = Layers()
        self.layers.append_layer(InputLayer(output_size=(1, 28, 28)))
        self.layers.append_layer(ConvolutionalLayer(im_size=(1, 28, 28),
                                                    kernel_size=(3, 1, 5, 5),
                                                    last_layer=self.layers.get_last_layer()))
        self.layers.append_layer(MaxPoolingLayer(input_size=(3, 24, 24),
                                                 pool_size=(2, 2),
                                                 last_layer=self.layers.get_last_layer()))
        self.layers.append_layer(ConvolutionalLayer(im_size=(3, 12, 12),
                                                    kernel_size=(3, 3, 5, 5),
                                                    last_layer=self.layers.get_last_layer()))
        self.layers.append_layer(MaxPoolingLayer(input_size=(3, 8, 8),
                                                 pool_size=(2, 2),
                                                 last_layer=self.layers.get_last_layer()))
        self.layers.append_layer(FullyConnectedLayer(input_size=(3, 4, 4),
                                                     neuron_size=30,
                                                     last_layer=self.layers.get_last_layer()))
        self.layers.append_layer(FullyConnectedLayer(input_size=30,
                                                     neuron_size=10,
                                                     last_layer=self.layers.get_last_layer()))

    def feed_forward(self, inputs):
        return self.layers.feed_forward(inputs)

    def train(self, training_data, num_of_epochs, mini_batch_size, learning_rate, validation_data=None):
        if validation_data:
            validation_data = list(validation_data)
        training_data = list(training_data)
        size = len(training_data)
        mini_batch_size = min(size, mini_batch_size)
        for i in range(num_of_epochs):
            random.shuffle(training_data)
            # divide mini batches for each epoch
            mini_batches = [training_data[i:i + mini_batch_size] for i in
                            range(0, size, mini_batch_size)]
            for mini_batch in mini_batches:
                self.process_mini_batch(mini_batch, learning_rate)
            if validation_data:
                print("completed epoch %d, accuracy %f" % ((i + 1), self.get_accuracy(validation_data)))
            else:
                print("completed epoch %d" % (i + 1))

    def process_mini_batch(self, mini_batch, learning_rate):
        batch_size = len(mini_batch)
        for x, y in mini_batch:
            # feed forward
            pred_y = self.feed_forward(x)
            # back propagation of errors
            error = pred_y - y
            self.layers.back_prop(error, learning_rate / batch_size)

    def get_accuracy(self, dataset):
        dataset = list(dataset)
        size = len(dataset)
        test_results = [(np.argmax(self.feed_forward(x)), y) for (x, y) in dataset]
        accuracy = sum(int(x == y) for (x, y) in test_results) / float(size)
        return accuracy


if __name__ == "__main__":
    import digits_recognition_neural_network.mnist_loader as mnist_loader

    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = ConvolutionalNeuralNetwork()
    net.train(training_data, 30, 10, 2., validation_data)
    print("posterior accuracy on test data %f" % net.get_accuracy(test_data))
