# -*- coding: utf-8 -*-

import sys
import logging

import numpy as np
import time

from util.activation_functions import Activation
from model.classifier import Classifier
import matplotlib.pyplot as plt
import random

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
        x = self.trainingSet.input

        t = np.array(self.trainingSet.label)

        plt.ion()

        current_ms = lambda: int(round(time.time()*1000))

        for i in range(self.epochs):
            time_start = current_ms()

            o_x = self.fire(x)
            error = t - o_x
            grad = np.dot(error, x)
            self.updateWeights(grad)

            time_end = current_ms()

            predict = [pre > 0.5 for pre in o_x]
            true = [truth == 1 for truth in t]
            right = 0.0

            for j in range(len(true)):
                if predict[j] is true[j]:
                    right += 1

            accuracy = right/len(true)


            if i % 7 == 0:
                plt.imshow(self.weight.reshape((28, 28)))
                plt.draw()
                plt.pause(0.05)

            time_ges = current_ms()

            print("epoche: {} \t accuracy: {:3.2f}%  \t eTime: {}ms \t eTime total: {}ms".format(i, accuracy*100, time_end-time_start, time_ges-time_start))

        plt.pause(1)
        
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
