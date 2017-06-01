# -*- coding: utf-8 -*-

import sys
import logging

import numpy as np

from util.activation_functions import Activation
from model.classifier import Classifier

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

    def __init__(self, train, valid, test, learningRate=0.01, epochs=50, simpleLearningRateDecay=False):

        self.learningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test
        self.simpleLearningRateDecay = simpleLearningRateDecay

        # Initialize the weight vector with small values
        self.weight = 0.01*np.random.randn(self.trainingSet.input.shape[1])

    def train(self, verbose=True):
        """Train the Logistic Regression.

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """
        for i in range(self.epochs):
            errors = self.trainBatch(i)

            if verbose:
                logging.info("Epoch: %i; Error: %i", i, errors)
        pass

    def trainBatch(self, epoch, verbose=True):
        grad = np.empty(self.trainingSet.input.shape[1])
        errors = 0
        for i in range(self.trainingSet.input.shape[0]):
            x = self.trainingSet.input[i,]
            o = self.fire(x)
            er = self.trainingSet.label[i] - o
            errors = errors + np.abs(self.trainingSet.label[i] - Activation.sign(o, 0.5))
            grad = np.add(grad, np.multiply(er, x))

        actualLearningRate = self.learningRate
        if (self.simpleLearningRateDecay):
            #a super simple linear learning rate decay: learn*(1-(epoch/#epochs))
            actualLearningRate = np.multiply(self.learningRate,1.0 - np.divide(epoch, self.epochs))
        self.weight = np.add(self.weight, np.multiply(actualLearningRate, grad))
        return errors
        
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
        pass

    def fire(self, input):
        # Look at how we change the activation function here!!!!
        # Not Activation.sign as in the perceptron, but sigmoid
        return Activation.sigmoid(np.dot(np.array(input), self.weight))
