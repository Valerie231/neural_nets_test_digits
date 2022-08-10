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
        dataset.append(instance)
    return dataset


def sigmoid(z):
    """
    Activation function
    """
    return 1.0/(1.0+np.exp(-z))


class Network(object):
    def __init__(self, sizes):
        """
        sizes is a list containing the length of each layer
        """
        self.sizes = sizes
        self.layers = np.array([np.zeros(i) for i in self.sizes], dtype=object)
        weight_sizes=list(zip(self.sizes[1:], self.sizes))
        self.weights = [np.random.rand(el[0], el[1]) for el in weight_sizes]
        self.bias = [np.random.rand(el) for el in self.sizes[1:]]
        

if __name__=="__main__":
    a=preprocessing("optdigits.tra")
    print(a[0])


