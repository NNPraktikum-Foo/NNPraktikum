import time

import numpy as np

from util.activation_functions import Activation


class LogisticLayer():
    """
    A layer of perceptrons acting as the output layer

    Parameters
    ----------
    nIn: int: number of units from the previous layer (or input data)
    nOut: int: number of units of the current layer (or output)
    activation: string: activation function of every units in the layer
    isClassifierLayer: bool:  to do classification or regression

    Attributes
    ----------
    nIn : positive int:
        number of units from the previous layer
    nOut : positive int:
        number of units of the current layer
    weights : ndarray
        weight matrix
    activation : functional
        activation function
    activationString : string
        the name of the activation function
    isClassifierLayer: bool
        to do classification or regression
    delta : ndarray
        partial derivatives
    size : positive int
        number of units in the current layer
    shape : tuple
        shape of the layer, is also shape of the weight matrix
    """

    def __init__(self, nIn, nOut, weights=None, learningRate=0.01,
                 activation='softmax', isClassifierLayer=True):

        # Get activation function from string
        # Notice the functional programming paradigms of Python + Numpy
        self.activationString = activation
        self.activation = Activation.getActivation(self.activationString)

        self.nIn = nIn
        self.nOut = nOut
        self.learningRate = learningRate

        # Adding bias
        self.output = np.ndarray(nOut)
        self.delta = np.zeros((self.nOut, self.nIn + 1))

        # You can have better initialization here
        if weights is None:
            rns = np.random.RandomState(int(time.time()))
            self.weights = rns.uniform(size=(nOut, nIn + 1))-0.5
        else:
            self.weights = weights

        self.isClassifierLayer = isClassifierLayer

        # Some handy properties of the layers
        self.size = self.nOut
        self.shape = self.weights.shape

    def forward(self, input):
        """
        Compute forward step over the input using its weights

        Parameters
        ----------
        input : ndarray
            a numpy array (1,nIn + 1) containing the input of the layer

        Returns
        -------
        ndarray :
            a numpy array (1,nOut) containing the output of the layer
        """
        # Forward pass: Extend the input vector to support bias and multiply it by
        # our weights. 
        self.input = input
        input = self.addBias(input)
        return self.activation(np.matmul(self.weights, input))

    def computeDerivative(self, grad, input, output):
        # TODO: Not sure why nextWeights is unused here. 
        """
        Compute the derivatives (back)

        Parameters
        ----------
        nextDerivatives: ndarray
            a numpy array containing the derivatives from next layer
        nextWeights : ndarray
            a numpy array containing the weights from next layer

        Returns
        -------
        ndarray :
            a numpy array containing the partial derivatives on this layer
        """

        activationDerivative = Activation.getDerivative(self.activationString)

        input = self.addBias(input)
        
        # Outer derivative, multiplied by derivative of activation function, multiplied by input
        delta = np.outer(np.matmul(grad, activationDerivative(output)), input)
        
        # Update delta
        self.delta += delta
        return delta

    def addBias(self, _d):
        # Extend a given vector with a one at the end (neede for bias)
        d = np.zeros(len(_d) + 1)
        d[0] = 1
        d[1:len(d) + 1] = _d
        return d

    def updateWeights(self, batchSize):
        """
        Update the weights of the layer
        """
        # Apply delta, normed by batchSize, then reset deltas
        self.weights = self.weights - self.learningRate * (self.delta / batchSize)
        self.delta = np.zeros((self.nOut, self.nIn + 1))
