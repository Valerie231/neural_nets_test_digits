"""
Test of neural nets written exlusively on numpy
"""

import logging
import random
import numpy as np


def preprocessing(path):
    """
    The function returns preprocessed dataset,
    taking path to dataset as input
    """
    dataset = []
    logging.info("Begin preprocessing")
    with open(path, "r") as f:
        dataset_raw = f.readlines()
        logging.info(f"Opened file {path}")
    post_file = f"postprocess_{path}"
    with open(post_file, "w") as f:
        for instance_raw in dataset_raw:
            instance_int = list(map(int, instance_raw.split(",")))
            instance = [el/16 for el in instance_int[:-1]]
            expected_output = [0]*10
            expected_output[instance_int[-1]] = 1
            instance.extend(expected_output)
            dataset.append(instance)
            f.write(str(instance))
            f.write("\n")
    np_dataset = np.array(dataset)
    logging.info("End preprocessing")
    logging.info(f"Dataset size is {np_dataset.shape}")
    return np_dataset


def sigmoid(z):
    """
    Activation function
    """
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    """
    Derivative of sigmoid function
    """
    return sigmoid(z)*(1-sigmoid(z))


class Network:
    def __init__(self, sizes):
        """
        sizes is a list containing the length of each layer
        """
        logging.info("Creating neural network")
        self.nr_layers = len(sizes)
        logging.info(f"Number of layers - {self.nr_layers}")
        self.sizes = sizes
        self.layers = np.array([np.zeros((i, 1)) for i in sizes], dtype=object)
        logging.info(f"Number of neurons in each layer - {self.sizes}")
        weight_sizes = list(zip(self.sizes[:-1], self.sizes[1:]))
        logging.info(f"Weight Sizes - {weight_sizes}")
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.bias = [np.random.rand(el, 1) for el in self.sizes[1:]]
        self.z = [np.zeros(i) for i in self.sizes]
        logging.info(f"Created neural network with these layers {sizes}")

    def feedforward(self, input):
        """
        neural network applying for our input data
        """
        self.layers[0] = input[:, np.newaxis]
        for y in range(1, len(self.sizes)):
            # logging.info(f'Working on feedforward for {y+1} layer')
            # logging.info(f'Shape of weights - {self.weights[y-1].shape}')
            # logging.info(f'Shape of layer - {self.layers[y-1].shape}')
            # logging.info(f'Shape of bias - {self.bias[y-1].shape}')
            self.z[y] = np.dot(self.weights[y-1], self.layers[y-1]) + self.bias[y-1]
            # logging.info(f'Shape of z after - {self.z[y].shape}')
            self.layers[y] = sigmoid(self.z[y])

    def error(self, expected_output):
        """
        error function,
        returns error of network
        """
        output = self.layers[-1]
        # logging.info(f'Output of net is {output.T}')
        # logging.info(f'Expected output of net is {expected_output.T}')
        error = np.sum((output.T - expected_output)**2/self.sizes[-1])
        # logging.info(f'Error is {error}')
        return error

    def backprop(self, network_input, expected_output):
        """
        backpropagate algorithm function,
        returns deltas for bias and weights
        """
        self.feedforward(network_input)
        expected_output = expected_output[:, np.newaxis]
        error_0 = self.error(expected_output)
        delta_b = [np.zeros(bias.shape) for bias in self.bias]
        delta_w = [np.zeros(weight.shape) for weight in self.weights]
        delta = (self.layers[-1]-expected_output)/self.sizes[-1]*sigmoid_prime(self.z[-1])
        # logging.info(f'Layer -1 shape is {self.layers[-1].shape}')
        # logging.info(f'Layer -2 shape is {self.layers[-2].shape}')
        # logging.info(f'Expected_output shape is {expected_output.shape}')
        delta_b[-1] = delta
        delta_w[-1] = np.dot(delta, self.layers[-2].T)
        for i in range(2,self.nr_layers):
            delta = np.dot(self.weights[-i+1].T, delta) * sigmoid_prime(self.z[-i])
            new_delta_w = np.dot(delta, self.layers[-i-1].T)
            new_delta_b = delta
            delta_b[-i] = new_delta_b
            delta_w[-i] = new_delta_w
            # logging.info(f'Back propagation step {i}')
            # logging.info(f'New delta_b shape {new_delta_w.shape}')
            # logging.info(f'New delta_b shape {new_delta_b.shape}')
            # logging.info(f'Delta shape {[el.shape for el in delta_b]}')
        return delta_w, delta_b

    def train(self, dataset, learn_rate, epochs):
        """
        Train method for neural net
        """
        random.shuffle(dataset)
        for epoch in range(epochs):
            for instance in dataset:
                # instance = random.choice(dataset)
                network_input = instance[:-10]
                expected_output = instance[-10:]
                delta_w, delta_b = self.backprop(network_input, expected_output)
                error = self.error(expected_output)
                self.weights = [weight-learn_rate*dw for weight, dw in zip(self.weights, delta_w)]
                self.bias = [bias-learn_rate*db for bias, db in zip(self.bias, delta_b)]




if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    training_data = preprocessing("optdigits.tra")
    testing_data = preprocessing("optdigits.tes")

    net = Network([64, 13, 9, 10])
    net.train(training_data, 5, 50)

    i = 0

    for instance in testing_data:
        network_input = instance[:-10]
        expected_output = instance[-10:]
        net.feedforward(network_input)
        if np.argmax(net.layers[-1])==np.argmax(expected_output):
            i=i+1
            logging.info(f"Correct prediction: {i}/{1797}, number {np.argmax(expected_output)}")
            logging.info(f"Expected output\n{expected_output}")
            logging.info(f"Net output\n{net.layers[-1]}")

    logging.info(f"Prediction rate: {100 * i/1797} %")
