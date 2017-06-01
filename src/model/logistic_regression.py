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

        learned = False
        iteration = 0
        grad = 0

        # Train for some epochs if the error is not 0
        while not learned:
            totalError = 0
            for input, label in zip(self.trainingSet.input, self.trainingSet.label):
                # Predict output of input
                output = self.fire(input)
                error = label - output

                # Increment total error by 1 if error is bigger than 1
                if error >= 0.5:
                    totalError += 1

                # Calculate new gradient
                grad = np.add(grad, np.multiply(error, input))

            # Update weights with new gradient
            self.weight = np.add(self.weight, np.multiply(self.decayLearningRate(iteration), grad))
            iteration += 1

            if verbose:
                logging.info("Epoch: %i; Falsely classified: %i", iteration, totalError)

            if iteration >= self.epochs:
                # Stop criteria is reached
                learned = True
        
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

    def decayLearningRate(self, epoch):
        """ Implement a decaying learning rate. Perform decay at every epoch.

        :param epoch: The current epoch or iteration of the training
        :return: The current learning rate
        """
        return self.learningRate * np.exp(-1.0 * np.multiply(0.001, epoch))

    def updateWeights(self, grad):
        pass

    def fire(self, input):
        # Look at how we change the activation function here!!!!
        # Not Activation.sign as in the perceptron, but sigmoid
        return Activation.sigmoid(np.dot(np.array(input), self.weight))
