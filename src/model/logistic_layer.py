
import time

import numpy as np

from util.activation_functions import Activation
from numpy import dot, matmul
#from model.layer import Layer


class LogisticLayer():#Layer):
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
        input = self.addBias(input)
        self.output = matmul(self.weights, input)
        return Activation.softmax(self.activation(self.output))

    def computeDerivative(self, nextDerivatives, nextWeights):
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
        nextDerivatives = self.addBias(nextDerivatives)
        # TODO this is most likely wrong
        delta = matmul(matmul(self.weights, nextDerivatives), activationDerivative(self.output))
        self.delta = self.delta + delta
        return delta

    def addBias(self, _d):
        d = np.zeros(len(_d) + 1)
        d[0] = 1
        d[1:len(d) + 1] = _d
        return d

    def updateWeights(self):
        """
        Update the weights of the layer
        """
        self.weights = self.weights + self.learningRate * self.delta
        self.delta = np.zeros((self.nOut, self.nIn + 1))
