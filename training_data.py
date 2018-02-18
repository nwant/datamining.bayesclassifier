def get_training_data():
    """get all of the available set data

    Returns: a list of record data. Each element in the list contains 2 values. The first value is a 3 value tuple,
    which contains the attributes for that particular record. The second value is the class (Default Borrower).

    The attributes for each records are Home Owner (T/F), Martial Status (S, M, or D. For single, married, divorced),
    Annual income (continuous value in thousands).

    The class is a binary value, which determines if the record corresponds to a Defaulting Borrower (T if defaulting,
    else F."""
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


def P(x, y):
    """the conditional probability """
    return 5


def P(y):
    """Determine the probability of a given class value.

    Input: y...the class value (boolean value)

    Returns: (float) the likelihood that any particular set of attributes will correspond to the provided class."""
    return len([x[0] for x in get_training_data() if x[1] is y]) / len(get_training_data())


def predict_class(x):
    """predict a class given a certain set of attributes

    Inputs: X...a 3 valued tuple that contains the following attributes in the following order:
    Home Owner (T/F), Martial Status (S, M, or D. For single, married, divorced), Annual Income (continuous value
    that represent thousands of dollars).

    Returns: a boolean value that represents the class prediction (True if predicted to be a Defaulting Borrower,
    else false."""

    # sanity checking
    if not isinstance(x, tuple) or len(x) != 3:
        print('X must be a 3 valued tuple, representing the following attributes in the following order: Home Owner' +
              '(T/F), Martial Status (S, M, or D), and Annual Income (continuous value for thousands of dollars')
        return
    elif not isinstance(x[0], bool):
        print('the home owner attribute must be a boolean')
        return
    elif x[1] not in {'S', 'M', 'D'}:
        print('the marital status attribute may only be "S", "M" or "D" (for single, married, or divorced')
        return
    elif not isinstance(x[2], (int, float)):
        print('the annual income attribute must be a continuous numerical value')
        return
    else:
        # perform prediction
        return True if P(x, True) > P(x, False) else False
