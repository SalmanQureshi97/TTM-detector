from __future__ import annotations

from sklearn.metrics import confusion_matrix


def compute_confusion(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)
