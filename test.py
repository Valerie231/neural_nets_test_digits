"""
Test of neural nets written exlusively on numpy
"""

import numpy as np
import random 


def preprocessing(path):
    """
    The function returns preprocessed dataset, 
    taking path to dataset as input
    """
    dataset=[]
    with open(path, "r") as f:
        dataset_raw = f.readlines()
    for instance in dataset_raw: 
        instance=list(map(int, instance[:-1].split(",")))
        instance[:-1]=[el/16 for el in instance[:-1]]
        expected_output = [0]*10
        expected_output[instance[-1]]=1
        instance[-1]=expected_output
        dataset.append(instance)
    return dataset


def sigmoid(z):
    """
    Activation function
    """
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return (np.exp(-z)*(sigmoid(z)**2))


class Network(object):
    def __init__(self, sizes):
        """
        sizes is a list containing the length of each layer
        """
        self.nr_layers = len(sizes)
        self.sizes = sizes
        self.layers = np.array([np.zeros(i) for i in self.sizes], dtype=object)
        weight_sizes=list(zip(self.sizes[1:], self.sizes))
        self.weights = [np.random.rand(el[0], el[1]) for el in weight_sizes]
        self.bias = [np.random.rand(el) for el in self.sizes[1:]]
        self.z = np.array([np.zeros(i) for i in self.sizes], dtype=object)
    
    def feedforward(self, input):
        """
        neural network applying for our input data
        """
        self.layers[0]=input
        for y in range(1,len(self.sizes)):
            self.z[y]=np.dot(self.weights[y-1], self.layers[y-1]) + self.bias[y-1]
            self.layers[y] = sigmoid(self.z[y]) 

    def cost(self, expected_output):
        """
        cost function
        """
        output=self.layers[-1]
        cost = 0.5*(output - expected_output)**2/self.sizes[-1]
        return cost
    

    def backprop(self, input, expected_output, learn_rate):
        """
        train function
        """
        delta_b=[]
        delta_w=[]
        self.feedforward(input)
        cost = self.cost(expected_output)
        delta = (self.layers[-1]-expected_output)/self.sizes[-1]*sigmoid_prime(self.z[-1])
        delta_b.append(delta)
        for i in reversed(range(1, self.nr_layers-1)):
            print(i)
            delta_w.append(np.dot(self.layers[i-1], delta))
            delta = np.dot(np.transpose(self.weights[i]), delta) * sigmoid_prime(self.z[i-1])
            delta_b.append(delta)
        delta_b=list(reversed(delta_b))
        delta_w=list(reversed(delta_w))
        self.bias=self.bias+learn_rate*delta_b
        self.weights=self.weights+learn_rate*delta_w
        return cost

    def train(self, dataset, learn_rate):
        random.shuffle(dataset)
        for instance in dataset:
            cost = self.backprop(instance[:-1], instance[-1], learn_rate)
            print(cost)
                
                
if __name__=="__main__":
    data=preprocessing("optdigits.tra")
    net = Network([64, 20, 10])
    net.train(data, 0.1)






