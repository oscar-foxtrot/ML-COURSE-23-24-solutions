import numpy as np


def evaluate_measures(sample):
    """Calculate measure of split quality (each node separately).

    Please use natural logarithm (e.g. np.log) to evaluate value of entropy measure.

    Parameters
    ----------
    sample : a list of integers.
    The size of the sample equals to the number of objects in the current node.
    The integer values are equal to the class labels of the objects in the node.

    Returns
    -------
    measures - a dictionary which contains three values of the split quality.
    Example of output:

    {
        'gini': 0.1,
        'entropy': 1.0,
        'error': 0.6
    }

    """
    uniques = np.unique(sample, return_counts=True)
    probs = np.unique(sample, return_counts=True)[1] / len(sample)

    print(np.unique(sample, return_counts=True)[1].argmax() / len(sample))
    measures = {
        'gini': float(1 - np.sum((np.unique(sample, return_counts=True)[1] / len(sample))**2)),
        'entropy': float(-np.sum(probs * np.log(probs))),
        'error': float(1 - uniques[1][uniques[1].argmax()] / len(sample))}
    return measures
