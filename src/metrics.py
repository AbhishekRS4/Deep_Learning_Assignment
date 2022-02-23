import numpy as np

def compute_accuracy(pred_label, true_label):
    pred_label = pred_label.astype(np.int32)
    true_label = true_label.astype(np.int32)

    same = (pred_label == true_label).sum()
    acc = same / len(pred_label)
    return acc
