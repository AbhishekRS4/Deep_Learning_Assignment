import numpy as np
import sklearn.datasets
from sklearn.model_selection import train_test_split

def load_gq_data(random_state=None):
    N = 500
    gaussian_quantiles = sklearn.datasets.make_gaussian_quantiles(
        mean=None, cov=0.7, n_samples=N, n_features=2, n_classes=2, shuffle=True, random_state=random_state)

    return gaussian_quantiles

def split_data(x, y, test_size=0.2, random_state=4):
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=test_size, random_state=random_state)
    return train_x, test_x, train_y, test_y
