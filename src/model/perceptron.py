# -*- coding: utf-8 -*-

import sys
import logging

import numpy as np

import matplotlib.pyplot as plt

from util.activation_functions import Activation
from model.classifier import Classifier

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


class Perceptron(Classifier):
    """
    A digit-7 recognizer based on perceptron algorithm

    Parameters
    ----------
    train : list
    valid : list
    test : list
    learningRate : float
    epochs : positive int

    Attributes
    ----------
    learningRate : float
    epochs : int
    trainingSet : list
    validationSet : list
    testSet : list
    weight : list
    """
    def __init__(self, train, valid, test,
                 learningRate=0.01, epochs=50):

        self.learningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

        # Initialize the weight vector with small random values
        # around 0 and0.1
        # 784 weights = 28^2
        #self.weight = np.random.rand(self.trainingSet.input.shape[1])/100
        self.weight = np.zeros(self.trainingSet.input.shape[1])
        #print("TrainingSer input shape " + str(self.trainingSet.input.shape[1]))

    def train(self, verbose=True):
        """Train the perceptron with the perceptron learning algorithm.

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """

        # Write your code to train the perceptron here

        #self.tryOne()

        h1, = plt.plot(np.zeros(3000))

        for i in range(self.epochs):
            print("Epoche " + str(i))
            x = self.trainingSet.input
            gx = self.fire(x)
            found = gx > 0
            ground_truth = np.array(self.trainingSet.label, dtype=bool)

            x[np.invert(ground_truth)] *= -1

            is_error = found != ground_truth

            #for e in range(len(is_error)):
            #    if is_error[e]:
            #        self.weight += self.learningRate * x[e]


            J = np.sum(x[is_error], axis=0)

            self.weight += J * self.learningRate / (len(is_error) +1)

            h1.set_ydata(gx)
            plt.draw()
            plt.pause(0.001)






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
        # Write your code to do the classification on an input image
        return self.fire(testInstance) > 0

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

    def updateWeights(self, input, error):
        # Write your code to update the weights of the perceptron here
        pass

    def fire(self, input):
        """Fire the output of the perceptron corresponding to the input """
        return Activation.sign(np.dot(np.array(input), self.weight))


    def tryOne(self):

        x = self.trainingSet.input

        accuracy = []

        for i in range(self.epochs):
            print("Epoche " + str(i))

            # Classification
            gx = self.fire(x)
            found = gx > 0

            groundt = np.array(self.trainingSet.label, dtype=bool)
            ngroundt = np.invert(groundt)

            # check result against label
            errors = found != groundt

            # change sign for second decision class
            x[ngroundt] *= -1

            # sum errors
            # J = np.sum(x[errors], axis=0)
            J = np.zeros(self.weight.shape)
            for e in x[errors]:
                J += e

            accuracy.append(np.sum(J) / (1 + len(x[errors])) * self.learningRate)

            if len(x[errors]):
                self.weight += self.learningRate * J / len(x[errors])

        print(accuracy)
        plt.plot(accuracy)
        plt.ylabel("Error")
        plt.show()