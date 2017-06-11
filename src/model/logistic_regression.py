# -*- coding: utf-8 -*-

import sys
import logging

import numpy as np
import matplotlib
matplotlib.use('Qt4Agg')
from matplotlib import pyplot as plt

from util.activation_functions import Activation
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

    def __init__(self, train, valid, test, learningRate=0.01, epochs=50, min_rmse=0.0, max_accuracy=100):

        self.learningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

        # Initialize the weight vector with small values
        self.weight = 0.01*np.random.randn(self.trainingSet.input.shape[1])
        self.min_rmse = min_rmse
        self.max_accuracy = max_accuracy

    def train(self, verbose=True):
        """Train the Logistic Regression.

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """
        #Initializations
        stopping = False
        e = 0
        #Decay constants
        base = 0.96
        decay_interval = 100
        plot_interval = 100

        #Input preprocessing
        input = self.trainingSet.input
        label = np.array(self.trainingSet.label)
        size = input.shape[0]
        positives = np.count_nonzero(label)
        #Plotting
        rmse = list()
        acu = list()

        #Training
        while not stopping:
            misses = 0
            fired = self.fire(input)
            signed_error = label - fired
            
            grad = np.dot(signed_error, input)
            self.updateWeights(grad)
            e += 1

            #Decay learning rate
            if (e%decay_interval == 0):
                self.decayLearningRate(base)

            #Calculate ratings
            misses = np.sum(abs(signed_error) >= 0.5)
            curr_rmse = RootMeanSquaredError.calculateError(label, fired)
            rmse.append(curr_rmse)
            accuracy = (size - misses)*100.0/size
            acu.append(accuracy)

            #Logging
            if verbose and (e%plot_interval == 0):
                logging.info("Epoch: %i; Error: %f; Misses: %i; Acc: %f; LR: %f", e, curr_rmse, misses, accuracy, self.learningRate)
            
            #Is any stopping criterion fullfilled?
            if self.stopping(e, rmse):
                stopping = True
                logging.info("Final learningRate: %f", self.learningRate)

        self.plot(acu, rmse)
        
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

    def decayLearningRate(self, base):
    		self.learningRate *= base

    def plot(self, accuracy, rmse):
        f, (ax1, ax2) = plt.subplots(1, 2)
        ax1.plot(accuracy)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(97.5, 100)
        ax2.plot(rmse)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('RMSE')
        ax2.set_ylim(0.0, 0.2)
        plt.show()

    def stopping(self, epoch, rmse=float("inf")):
        return (epoch >= self.epochs) or (rmse < self.min_rmse)

