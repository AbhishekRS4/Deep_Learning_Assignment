import numpy as np
import sklearn.datasets
from sklearn.model_selection import train_test_split

def load_gq_data(random_state=None):
    """
    ---------
    Arguments
    ---------
    random_state : int (default=None)
        an int for random state to be used to generate the Gaussian quantile dataset

    -------
    Returns
    -------
    gaussian_quantiles : ndarray
        an array containing features and labels of the dataset
    """
    N = 500
    gaussian_quantiles = sklearn.datasets.make_gaussian_quantiles(
        mean=None, cov=0.7, n_samples=N, n_features=2, n_classes=2, shuffle=True, random_state=random_state)

    return gaussian_quantiles

def split_data(x, y, test_size=0.2, random_state=4):
    """
    ---------
    Arguments
    ---------
    x : ndarray
        a numpy array of features
    y : ndarray
        a numpy array of labels
    test_size : float
        fraction of the dataset which should be the size of test set from the resulting split
    random_state : int (default=4)
        an int for random state to be used to generate the Gaussian quantile dataset

    -------
    Returns
    -------
    train_x : ndarray
        an array containing train set features
    train_y : ndarray
        an array containing train set labels
    test_x : ndarray
        an array containing test set features
    test_y : ndarray
        an array containing test set labels
    """
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=test_size, random_state=random_state)
    return train_x, test_x, train_y, test_y
