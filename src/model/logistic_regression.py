# -*- coding: utf-8 -*-

import sys
import logging

import numpy as np
import matplotlib.pyplot as plt

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

        # Create logistic layer, we use two output neurons (onehot)
        self.model = LogisticLayer(784, 2, learningRate=self.learningRate, isClassifierLayer = True, activation="sigmoid")


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

        acc_list = list()

        while not learned:
            totalError = 0
            totalMSEError = 0
            
            # Shuffle data for each iteration.
            data = np.random.permutation(list(zip(self.trainingSet.input, self.trainingSet.label)))

            errorCount = 0
            for start in range(0, n, self.batchSize): # Use batchSize als step size for this loop
                end = min(start + self.batchSize, n)
                # We create our batch by slicing the input data appropiately. 

                for input, label in data[start:end]:

                    # Forward pass. 
                    output = self.forward(input)

                    # Create onehot target from label 
                    target = np.zeros(2)
                    target[label] = 1

                    if self.classifyFromOutput(output) == self.classifyFromOutput(target):
                        errorCount+=1

                    # Derivate of the logisitc regression
                    grad = np.matmul((output - target), np.linalg.pinv(np.outer(output, (np.ones(output.shape) - output))))

                    # Backward pass for inner layers
                    self.backward(grad, input, output) 

                    # MSE Error, just for debugging
                    totalMSEError += np.sum(abs(output - target))

                    # compute BCE recognizing error
                    totalError += abs(loss.calculateError(target, output))



                # After a batch iteration, call update weights
                self.model.updateWeights(self.batchSize)

            # Divide errors by item count, so we can read it more easily. 
            totalError = totalError / n
            totalMSEError = totalMSEError / n
            
            iteration += 1

            trainAccuracy = errorCount * 100.0 / len(data)
            acc_list.append(trainAccuracy)
            if verbose:
                logging.info("Epoch: %i; BCEError: %f, MSEError: %f", iteration, np.asscalar(totalError), np.asscalar(totalMSEError))
            if totalError == 0 or iteration >= self.epochs:
                # stop criteria is reached
                learned = True

        plt.plot(acc_list, label='Test Accuracy')
        plt.ylabel('Test Accuracy / %')
        plt.xlabel('Epoch')
        plt.ylim(0, 100)
        plt.tight_layout()
        plt.show()

    def forward(self, input):
        # Invokes a forward pass 
        return self.model.forward(input)

    def backward(self, grad, input, output):
        # Computes the derivatie for our layer. 
        return self.model.computeDerivative(grad, input, output)

    def classifyFromOutput(self, output):
        # Converts onehot to true/false (1 or 0)
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
