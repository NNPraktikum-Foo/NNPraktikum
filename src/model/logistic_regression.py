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

    def __init__(self, train, valid, test, batchSize = 64, learningRate=0.01, epochs=50):

        self.learningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test
        self.batchSize = batchSize

        self.model = LogisticLayer(784, 2, learningRate=self.learningRate, isClassifierLayer = True, activation="sigmoid")
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
        n = len(self.trainingSet.label)

        while not learned:
            totalError = 0
            totalMSEError = 0;
            for start in xrange(0, n, self.batchSize):
                end = min(start + self.batchSize, n)
                for input, label in np.random.permutation(zip(self.trainingSet.input[start:end], self.trainingSet.label[start:end])):
                    output = self.forward(input)

                    target = np.zeros(2)
                    target[label] = 1

                    # Derivate of the logisitc regression
                    dError = np.matmul((output - target), np.linalg.pinv(np.outer(output, (np.ones(output.shape) - output))))

                    # MSE Error, just for debugging
                    totalMSEError += np.sum(abs(output - target));

                    # sum up gradient
                    self.backward(dError, input) 

                    # compute recognizing error
                    error = loss.calculateError(target, output)
                    totalError += abs(error)
                self.model.updateWeights()

            totalError = totalError / n
            totalMSEError = totalMSEError / n;
            
            iteration += 1

            if verbose:
                logging.info("Epoch: %i; BCEError: %f, MSEError: %f", iteration, np.asscalar(totalError), np.asscalar(totalMSEError))
            if totalError == 0 or iteration >= self.epochs:
                # stop criteria is reached
                learned = True

    def forward(self, input):
        return self.model.forward(input)

    def backward(self, error, input):
        return self.model.computeDerivative(error, input)

    def classifyFromOutput(self, output):
        return np.argmax(output) 

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
        #return self.forward(testInstance)

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
