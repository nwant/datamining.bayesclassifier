#!/usr/bin/env python3
#
# Nathaniel Want (nwqk6)
# CS5342-G01
# Bayesian Classifier
# March 1, 2018
#
# Overview:
# The following program and library creates a basic Bayesian Classifier based on the requirements of the first
# homework project for CS5342 (Spring 2018).
#
# Training Data:
# The sample training data used was taken from Figure 5.9 in the book, per the instructions for this project. However,
# the program will allow users to insert their own training data, so long as the structure of the data follows that
# of the sample data.
#
# Smoothing:
# By default, smoothing is applied to binary and categorical attributes whenever the class conditional probability for
# the attribute equals 0. By default, the user-specified parameter used for smoothing is equal to 1, meaning a laplace
# smoothing is used. This value was used given the smaller size of the sample data set. However, the interfaces for
# the program and library functions allows a user to define this parameter his or herself.
#
# Testing:
# Tests were performed using the inputs given
# Special Implementation Notes:
#
import math
import statistics


def sample_training_data():
    """
    get all the sample training data set. Taken from Figure 5.9 from the book.

    :return: a list of record data. Each element in the list contains 2 values. The first value is a 3 value tuple,
    which contains the attributes for that particular record. The second value is the class (Default Borrower).
    The attributes for each records are Home Owner (T/F), Martial Status (S, M, or D. For single, married, divorced),
    Annual income (continuous value in thousands). The class is a binary value, which determines if the record
    corresponds to a Defaulting Borrower (T if defaulting, else F).
    """
    return [
        ((True, 'S', 125), False),
        ((False, 'M', 100), False),
        ((False, 'S', 70), False),
        ((True, 'M', 120), False),
        ((False, 'D', 95), True),
        ((False, 'M', 60), False),
        ((True, 'D', 220), False),
        ((False, 'S', 85), True),
        ((False, 'M', 75), False),
        ((False, 'S', 90), True)
    ]


def conditional_probability(x, y, i, t, s=False, p=1):
    """
    get the conditional probability P(x|y) for a particular attribute.

    :param x: (list) the test attribute vector
    :param y: (bool) the class value
    :param i: (int) the (1-based) attribute number (i.e. the corresponding column number in the training data set)
    :param t: (list) the training set
    :param s: (bool) add smoothing?
    :param p: (int) smoothing parameter. Defaults to 1, meaning smoothing will be a laplace smoothing
    :return: (float) the conditional probability P(x|y)
    """
    m = p if s else 0

    def home_owner(a):
        return a[0]

    def marital_status(a):
        return a[1]

    def annual_income(a):
        return a[2]

    if i == 1:  # Home Owner (binary)
        return (len([1 for X, Y in t if home_owner(X) == home_owner(x) and Y is y]) + m) \
            / (len([1 for X, Y in t if Y is y]) + 2*m)
        # do we need to apply smoothing?
    elif i == 2:  # Marital Status (categorical)
        return (len([1 for X, Y in t if marital_status(X) == marital_status(x) and Y is y]) + m) \
               / (len([1 for X, Y in t if Y is y]) + 3*m)
    elif i == 3:  # Annual Income (continuous)
        # use gaussian (normal) distribution
        mu = statistics.mean([annual_income(X) for X, Y in t if Y is y])      # mean
        s2 = statistics.variance([annual_income(X) for X, Y in t if Y is y])  # sample variance
        s = math.sqrt(s2)  # standard distribution
        return 1 / (math.sqrt(2 * math.pi) * s) * math.exp(-(annual_income(x) - mu)**2 / (s2**2))
    else:
        raise IndexError('Training data has 3 attributes. i can only be supplied with 1, 2, or 3')


def class_conditional_probability(x, y, t, p=1):
    """
    Determine the probability P(x|y) for an test attribute vector
    :param x: (tuple) the attribute vector for the test data
    :param y: (bool) the class we are using as the prior probability
    :param t: (list) the training set
    :param p: (int) smoothing parameter. Defaults to 1, meaning smoothing will be a laplace smoothing
    :return: (float) P(x|y)
    """
    p = 1
    for i in range(1, 4):
        # determine the conditional probability for this attribute given class y. Apply smoothing if necessary.
        p *= conditional_probability(x, y, i, t, p=p) if conditional_probability(x, y, i, t, p=p) > 0 \
            else conditional_probability(x, y, i, t, True, p)

    return p * prior_probability(y, t)


def prior_probability(y, t):
    """
    P(y) Determine the probability of a given class value.

    :param y: (bool) the class value
    :param t: (list) the training set
    :return: (float) the likelihood that any particular set of attributes will correspond to the provided class."""
    try:
        return len([1 for X, Y in t if Y is y]) / len(t)
    except ZeroDivisionError:
        return 0


def predict_class(x, t=sample_training_data(), p=1, exact_matching=False):
    """
    predict a class given a certain set of attributes

    :param x: (tuple) a 3 valued tuple that contains the following attributes in the following order:
    Home Owner (T/F), Martial Status (S, M, or D. For single, married, divorced), Annual Income (continuous value
    that represent thousands of dollars).
    :param t: (list) the training data.
    :param p: (int) smoothing parameter. Defaults to 1, meaning smoothing will be a laplace smoothing
    :param exact_matching: (bool) When set to true, if all of the test attributes exactly match that of one or more
        records in the training set AND all of the matching records have the same classification, said classification
        will be returned immediately rather than calculating the probability. If this is set to false, all test records
        will be calculated, regardless if the same records already exist in the training set.
    :return: (bool) the class prediction (True if predicted to be a Defaulting Borrower, else false."""
    # sanity checking
    if not isinstance(x, tuple) or len(x) != 3:
        raise TypeError('X must be a 3 valued tuple')
    elif not isinstance(x[0], bool):
        raise TypeError('The home owner attribute must be a boolean')
    elif x[1] not in {'S', 'M', 'D'}:
        raise TypeError('the marital status attribute may only be "S", "M" or "D" (for single, married, or divorced')
    elif not isinstance(x[2], (int, float)):
        raise TypeError('the annual income attribute must be a continuous numerical value')

    # if exact matching is enabled see if the attribute matches the one of the records in the training set and if
    # there are no conflicting class assignments for said matching records. If so, simply return the class for one
    # of these records
    if exact_matching and x in [X for X, _ in t] and len(set([Y for X, Y in t if x == X])) == 1:
        return [Y for X, Y in t if x == X][0]
    else:  # matching record not found or matching records have conflicting classes => perform prediction
        return class_conditional_probability(x, True, t, p) >= class_conditional_probability(x, False, t, p)


if __name__ == '__main__':
    print('Bayesian Classifier Tests\n\n')

    print('test a): X=(Home Owner=Yes, Martial Status=M, Annual Income=50.7k)')
    X = (True, 'M', 50.7)
    print('==> P(X|Yes)={0}; P(X|No)={1}'.format(class_conditional_probability(X, True, sample_training_data()),
                                                 class_conditional_probability(X, False, sample_training_data())))
    print('=====>Calculated Match Prediction: Defaulted Borrower={0}'
          .format('Yes' if predict_class(X, exact_matching=False) else 'No'))
    print('=====>Exact Match Prediction: Defaulted Borrower={0}'
          .format('Yes' if predict_class(X) else 'No', exact_matching=True))

    print('\n\ntest b): Tests using table in Figure 5.9 (Training Set):\n')
    for i, t in enumerate(sample_training_data()):
        x, y = t
        print('\nTid {0}: X=(Home Owner={1}, Marital Status={2}, Annual Income={3}k)'.format(i+1, x[0], x[1], x[2]))
        print('=====> Calculated Match Prediction: Defaulted Borrower={0} ; Actual: Defaulted Borrower={1}'
              .format('Yes' if predict_class(x, exact_matching=False) else 'No', 'Yes' if y else 'No'))
        print('=====> Exact Match Prediction: Defaulted Borrower={0} ; Actual: Defaulted Borrower={1}'
              .format('Yes' if predict_class(x, exact_matching=True) else 'No', 'Yes' if y else 'No'))

