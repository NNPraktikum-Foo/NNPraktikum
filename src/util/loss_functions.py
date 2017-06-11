# -*- coding: utf-8 -*-


"""
Loss functions.
"""

import numpy as np

from abc import ABCMeta, abstractmethod, abstractproperty


class Error:
    """
    Abstract class of an Error
    """
    __metaclass__ = ABCMeta

    @abstractproperty
    def errorString(self):
        pass

    @abstractmethod
    def calculateError(self, target, output):
        # calculate the error between target and output
        pass


class AbsoluteError(Error):
    """
    The Loss calculated by the number of differences between target and output
    """
    def errorString(self):
        self.errorString = 'absolute'

    @staticmethod
    def calculateError(self, target, output):
        # It is the numbers of differences between target and output
        return abs(DifferentError.calculateError(target, output))


class DifferentError(Error):
    """
    The Loss calculated by the number of differences between target and output
    """
    def errorString(self):
        self.errorString = 'different'

    @staticmethod
    def calculateError(target, output):
        # It is the numbers of differences between target and output
        return target - output


class MeanSquaredError(Error):
    """
    The Loss calculated by the mean of the total squares of differences between
    target and output.
    """
    def errorString(self):
        self.errorString = 'mse'

    @staticmethod
    def calculateError(target, output):
        # MSE = 1/n*sum (i=1 to n) of (target_i - output_i)^2)
        error = DifferentError.calculateError(target, output)
        return np.mean(np.power(error, 2))


class RootMeanSquaredError(Error):
    """
    The Loss calculated by the root of the mean of the total squares of differences between
    target and output.
    """

    def errorString(self):
        self.errorString = 'rmse'

    @staticmethod
    def calculateError(target, output):
        # RMSE = sqrt(1/n*sum (i=1 to n) of (target_i - output_i)^2))
        return np.sqrt(MeanSquaredError.calculateError(target, output))


class SumSquaredError(Error):
    """
    The Loss calculated by the sum of the total squares of differences between
    target and output.
    """
    def errorString(self):
        self.errorString = 'sse'

    @staticmethod
    def calculateError( target, output):
        # SSE = 1/2*sum (i=1 to n) of (target_i - output_i)^2)
        error = DifferentError.calculateError(target, output)
        return .5*np.sum(np.power(error, 2))


class BinaryCrossEntropyError(Error):
    """
    The Loss calculated by the Cross Entropy between binary target and
    probabilistic output (BCE)
    """
    def errorString(self):
        self.errorString = 'bce'

    def calculateError(self, target, output):
        pass


class CrossEntropyError(Error):
    """
    The Loss calculated by the more general Cross Entropy between two
    probabilistic distributions.
    """
    def errorString(self):
        self.errorString = 'crossentropy'

    def calculateError(self, target, output):
        pass
