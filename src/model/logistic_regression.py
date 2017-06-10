# -*- coding: utf-8 -*-

import sys
import logging

import numpy as np

from util.activation_functions import Activation
from model.classifier import Classifier
from  util.loss_functions import DifferentError

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


class LogisticRegression(Classifier):
    """
    A digit-7 recognizer based on logistic regression algorithm

    Parameters
    ----------
    train : list
    valid : list
    test : list
    learningRate : float
    epochs : positive int

    Attributes
    ----------
    trainingSet : list
    validationSet : list
    testSet : list
    weight : list
    learningRate : float
    epochs : positive int
    """

    def __init__(self, train, valid, test, learningRate=0.01, epochs=50):

        self.learningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

        # Initialize the weight vector with small values
        self.weight = 0.01*np.random.randn(self.trainingSet.input.shape[1])

    def train(self, verbose=True):
        """Train the Logistic Regression.

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """
        MyLossFunction = DifferentError()
        grad = 0
        x = self.trainingSet.input
        t = np.array(self.trainingSet.label)
        #t[t == 0] = -1

        dWeightsPrevious = 0

        for i in range(self.epochs):
            o_x = self.fire(x)
            error = MyLossFunction.calculateError(t, o_x)

            #update leaning rate
            dWeightsCurrent = self.learningRate*grad
            cosPhi = np.sum(np.dot(dWeightsCurrent,dWeightsPrevious))
            normalization = np.sqrt(np.dot(dWeightsPrevious, dWeightsPrevious)*np.dot(dWeightsCurrent, dWeightsCurrent)) + 10e-8
            cosPhi = cosPhi/normalization
            const = 1
            self.learningRate = self.learningRate*const*(cosPhi+1+1e-6)/2
            dWeightsPrevious = self.learningRate*grad
            grad = grad + np.dot(error, x)
            self.updateWeights(grad)
            #print('round', i+1, 'error', np.sum(error), 'weight', np.sum(self.weight))

        
    def classify(self, testInstance):
        """Classify a single instance.

        Parameters
        ----------
        testInstance : list of floats

        Returns
        -------
        bool :
            True if the testInstance is recognized as a 7, False otherwise.
        """
        return Activation.sign(self.fire(testInstance), 0.5)

    def evaluate(self, test=None):
        """Evaluate a whole dataset.

        Parameters
        ----------
        test : the dataset to be classified
        if no test data, the test set associated to the classifier will be used

        Returns
        -------
        List:
            List of classified decisions for the dataset's entries.
        """
        if test is None:
            test = self.testSet.input
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))

    def updateWeights(self, grad):
        self.weight += self.learningRate * grad


    def fire(self, input):
        # Look at how we change the activation function here!!!!
        # Not Activation.sign as in the perceptron, but sigmoid
        return Activation.sigmoid(np.dot(np.array(input), self.weight))
