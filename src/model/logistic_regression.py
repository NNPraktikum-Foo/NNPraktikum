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
        #Initializations
        stopping = False
        e = 0
        #Decay constants
        base = 0.96
        decay_interval = 100

        #Input preprocessing
        input = self.trainingSet.input
        label = np.array(self.trainingSet.label)
        size = input.shape[0]
        positives = np.count_nonzero(label)
      	#Average number of negatives for each positive
        inbalance = float(size)/positives - 1

        #Training
        while not stopping:
        		misses = 0
        		fired = self.fire(input)
        		signed_error = label - fired
        		for i in range(size):
        				if signed_error[i] >= 0.5:
        						misses += 1
        		
        		grad = np.dot(signed_error, input)
        		self.updateWeights(grad)
        		e += 1
        		if (e%decay_interval == 0):
        				self.decayLearningRate(base)
        		if verbose and (e%100 == 0):
        				quad_error = np.average(np.power(signed_error, 2))
        				accuracy = (size - inbalance*misses)*100.0/size		
        				logging.info("Epoch: %i; Error: %f; Misses: %i, Acc: %f", e, quad_error, misses, accuracy)
        		if e >= self.epochs:
        				stopping = True
        				logging.info("Final learningRate: %f", self.learningRate)
        
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
