import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def compute_accuracy(true_labels, pred_labels):
    """
    ---------
    Arguments
    ---------
    true_labels : ndarray
        a numpy array of true labels
    pred_labels : ndarray
        a numpy array of predicted labels

    -------
    Returns
    -------
    acc : float
        accuracy score
    """
    pred_labels = pred_labels.astype(np.int32)
    true_labels = true_labels.astype(np.int32)

    acc = accuracy_score(true_labels, pred_labels)
    return acc

def compute_test_metrics(pred_labels, true_labels):
    """
    ---------
    Arguments
    ---------
    true_labels : ndarray
        a numpy array of true labels
    pred_labels : ndarray
        a numpy array of predicted labels

    -------
    Returns
    -------
    acc : float
        accuracy score
    cm : ndarray
        confusion matrix
    f1 : float
        f1 score
    """
    true_labels = true_labels.astype(np.int32)
    pred_labels = pred_labels.astype(np.int32)

    acc = accuracy_score(true_labels, pred_labels)
    cm = confusion_matrix(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)

    return acc, cm, f1
