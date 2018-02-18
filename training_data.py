def get_training_data():
    """get all of the available set data

    Returns: a list of record data. Each element in the list contains 2 values. The first value is a 4 value tuple,
    which contains the attributes for that particular record. The second value is the class (Default Borrower).

    The attributes for each records are TID (the record identifier), Home Owner (T/F),
    Martial Status (S, M, or D. For single, married, divorced), Annual income (continuous value in thousands).

    The class is a binary value, which determines if the record corresponds to a Defaulting Borrower (T if defaulting,
    else F."""
    return [
        ((1, True, 'S', 125), False),
        ((2, False, 'M', 100), False),
        ((3, False, 'S', 70), False),
        ((4, True, 'M', 120), False),
        ((5, False, 'D', 95), True),
        ((6, False, 'M', 60), False),
        ((7, True, 'D', 220), False),
        ((8, False, 'S', 85), True),
        ((9, False, 'M', 75), False),
        ((10, False, 'S', 90), True)
    ]


