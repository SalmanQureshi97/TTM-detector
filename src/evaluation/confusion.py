from __future__ import annotations

from sklearn.metrics import confusion_matrix


def compute_confusion(y_true, y_pred, labels=None):
    """Confusion matrix. Pass ``labels`` (e.g. [0,1,2,3]) to guarantee a
    fixed-size matrix even when some class is absent from the predictions."""
    return confusion_matrix(y_true, y_pred, labels=labels)
