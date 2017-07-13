# -*- coding: utf-8 -*-

import sys
import logging

import numpy as np

from util.activation_functions import Activation
from model.classifier import Classifier
from model.logistic_layer import LogisticLayer
from util.loss_functions import BinaryCrossEntropyError

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

    def __init__(self, train, valid, test, batch_size = 64, learningRate=0.01, epochs=50):

        self.learningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

        self.model = LogisticLayer(len(self.trainingSet.input[0]), len(self.trainingSet.label[0]), isClassifierLayer = True)
        # ctor for LogisticLayer
        # def __init__(self, nIn, nOut, weights=None, learningRate=0.01,
        #          activation='softmax', isClassifierLayer=True):


    def train(self, verbose=True):
        """Train the Logistic Regression.

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """

        loss = BinaryCrossEntropyError()

        learned = False
        iteration = 0
        n = len(self.trainingSet)

        while not learned:
            totalError = 0
            for start in xrange(0, n ,batchSize):
                end = min(start + batchSize, n)
                grad = 0
                for input, label in zip(self.trainingSet.input[start:end],
                                        self.trainingSet.label[start:end]):
                    output = self.forward(input)
                    # compute derived error
                    dError = -(label - output)

                    # sum up gradient
                    grad += self.backward(error) 

                    # compute recognizing error
                    predictedLabel = self.classifyFromOutput(output) # same as in classify
                    error = loss.calculateError(label, predictedLabel)
                    totalError += error

                self.model.updateWeights()
            totalError = abs(totalError)
            
            iteration += 1

            if verbose:
                logging.info("Epoch: %i; Error: %i", iteration, totalError)

            if totalError == 0 or iteration >= self.epochs:
                # stop criteria is reached
                learned = True

    def forward(self, input):
        return self.model.forward(input)

    def backward(self, error):
        return self.model.computeDerivative(error, None)

    def classifyFromOutput(self, output):
        return output > 0.5

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
        return self.classifyFromOutput(self.forward(testInstance))

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


        return list(map(self.classify, test))
