"""
Test of neural nets written exlusively on numpy
"""
# pylint: disable=C0103, C0301

import random
import numpy as np


def preprocessing(path):
    """
    The function returns preprocessed dataset,
    taking path to dataset as input
    """
    dataset = []
    with open(path, "r") as f:
        dataset_raw = f.readlines()
    for instance in dataset_raw:
        instance = list(map(int, instance[:-1].split(",")))
        instance[:-1] = [el/16 for el in instance[:-1]]
        expected_output = np.array([[0]*10])
        expected_output[0][instance[-1]] = 1
        instance[-1] = expected_output
        dataset.append(instance)
    return dataset


def sigmoid(z):
    """
    Activation function
    """
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    """
    Derivative of sigmoid function
    """
    return (-np.exp(-z)*(sigmoid(z)**2))


class Network:
    def __init__(self, sizes):
        """
        sizes is a list containing the length of each layer
        """
        self.nr_layers = len(sizes)
        self.sizes = sizes
        self.layers = np.array([np.zeros((i, 1)) for i in sizes], dtype=object)
        weight_sizes = list(zip(self.sizes[1:], self.sizes))
        self.weights = np.array([np.random.rand(el[0], el[1]) for el in weight_sizes], dtype=object)
        self.bias = np.array([np.random.rand(el, 1) for el in self.sizes[1:]], dtype=object)
        self.z = np.array([np.zeros(i) for i in self.sizes], dtype=object)

    def feedforward(self, input):
        """
        neural network applying for our input data
        """
        self.layers[0] = np.array([[i] for i in input])
        for y in range(1, len(self.sizes)):
            self.z[y] = np.dot(self.weights[y-1], self.layers[y-1]) + self.bias[y-1]
            self.layers[y] = sigmoid(self.z[y])
        return self.layers[-1]

    def cost(self, expected_output):
        """
        cost function
        """
        output = self.layers[-1]
        cost = 0.5*(output - expected_output.T)**2/self.sizes[-1]
        return np.sum(cost)

    def backprop(self, input, expected_output, learn_rate):
        """
        train function
        """
        delta_b = []
        delta_w = []
        self.feedforward(input)
        cost_0 = self.cost(expected_output)
        delta = (self.layers[-1]-expected_output.T)/self.sizes[-1]*sigmoid_prime(self.z[-1])
        for i in reversed(range(1, self.nr_layers)):
            delta_w.append(np.dot(delta, self.layers[i-1].T))
            delta_b.append(delta)
            delta = np.dot(np.transpose(self.weights[i-1]), delta) * sigmoid_prime(self.z[i-1])
        delta_b_list = list(reversed(delta_b))
        delta_w_list = list(reversed(delta_w))
        delta_b = np.array([el for el in delta_b_list], dtype=object)
        delta_w = np.array([el for el in delta_w_list], dtype=object)
        with open("delta_b", "w") as f:
            f.write(str(delta_b))
        self.bias = self.bias+learn_rate*delta_b
        self.weights = self.weights+learn_rate*delta_w
        self.feedforward(input)
        cost_1 = self.cost(expected_output)
        if cost_1 > cost_0:
            self.bias = self.bias - 0.5*learn_rate*delta_b
            self.weights = self.weights - 0.5*learn_rate*delta_w
        return cost_0

    def train(self, dataset, learn_rate):
        """
        Train method for  neural net
        """
        instance = random.choice(dataset)
        cost = self.backprop(instance[:-1], instance[-1], learn_rate)
        while cost > 0.025:
            instance = random.choice(dataset)
            cost = self.backprop(instance[:-1], instance[-1], learn_rate)
            print(cost)


if __name__ == "__main__":
    data = preprocessing("optdigits.tra")
    net = Network([64, 20, 10])
    net.train(data, 1.75)
    test_data = preprocessing("optdigits.tes")
    print("------------------------------")
    print("Network results")
    print(net.feedforward(data[0][:-1]).T)
    print("Expected output")
    print(data[0][-1])
    print("Cost function")
    print(net.cost(data[0][-1]))
    print("Error by neurons")
    a = 0.5*(net.layers[-1]-data[0][-1].T)**2
    print(a.T)
    print("------------------------------")
