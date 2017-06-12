# -*- coding: utf-8 -*-

import sys
import logging

import numpy as np
import matplotlib.pyplot as plt

from util.activation_functions import Activation
from util.loss_functions import DifferentError
from util.loss_functions import RootMeanSquaredError
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

    def __init__(self, train, valid, test, learningRate=0.01, epochs=50,
    	minRmse=0.0, base=1, decayInterval=float("Inf")):

        self.learningRate = learningRate
        self.epochs = epochs
        self.loggingInterval = epochs // 100

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test
        self.minRmse = minRmse
        self.base = base
        self.decayInterval = decayInterval

        # Initialize the weight vector with small values
        self.weight = 0.01*np.random.randn(self.trainingSet.input.shape[1]+1)

    def train(self, verbose=True):
        """Train the Logistic Regression.

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """
        x = self.trainingSet.input
        x = np.insert(x, 0, 1, axis=1)
        size = x.shape[0]

        t = np.array(self.trainingSet.label)

        rmse_list = list()
        acc_list = list()

        for i in range(self.epochs):
            o_x = self.fire(x)
            error = DifferentError.calculateError(t, o_x)

            rmse = RootMeanSquaredError.calculateError(t, o_x)
            rmse_list.append(rmse)
            misses = np.sum(abs(error) >= 0.5)
            trainAccuracy = (size - misses)*100.0/size
            acc_list.append(trainAccuracy)
            if rmse < self.minRmse:
                break

            grad = np.dot(error, x)
            self.updateWeights(grad)
            self.decayLearningRate(i+1) #i+1 because we don't want to decay after the first step already

            if(verbose and ((i+1)%self.loggingInterval == 0)):
                logging.info("Epoch: %i; RMSE: %f; TrainAccuracy: %f; LearningRate: %f",
                             i+1, rmse, trainAccuracy, self.learningRate)

        plt.subplot(1,2,1)
        plt.plot(rmse_list, label='RMSE')
        plt.ylabel('RMSE')
        plt.xlabel('Epoch')
        plt.ylim(0.0, 0.2)
        plt.subplot(1,2,2)
        plt.plot(acc_list, label='Test Accuracy')
        plt.ylabel('Test Accuracy [%]')
        plt.xlabel('Epoch')
        plt.ylim(95, 100)
        plt.tight_layout()
        plt.show()

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
        testInstance = np.insert(testInstance, 0, 1)
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

    def decayLearningRate(self, epoch):
        """Change the learning rate.

        Parameters
        ----------
        epoch : current epoch of the training
        """
        if(epoch % self.decayInterval == 0):
            self.learningRate *= self.base
