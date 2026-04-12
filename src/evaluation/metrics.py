from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, roc_curve


CLASS4_TO_AUTH = np.array([0, 0, 1, 1])
CLASS4_TO_ENC = np.array([0, 1, 0, 1])


def equal_error_rate(y_true, y_prob):
    """Compute the Equal Error Rate (EER) where FPR == FNR."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    if len(np.unique(y_true)) < 2:
        return np.nan
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fnr = 1 - tpr
    eer_func = interp1d(fpr, fnr)
    eer = brentq(lambda x: eer_func(x) - x, 0.0, 1.0)
    return float(eer)


def binary_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    y_true = np.asarray(y_true).astype(int)
    out = {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "eer": equal_error_rate(y_true, y_prob),
    }
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = roc_auc_score(y_true, y_prob)
    else:
        out["roc_auc"] = np.nan
    return out


def multitask_metrics(auth_true, auth_prob, enc_true, enc_prob, threshold=0.5):
    auth_true = np.asarray(auth_true).astype(int)
    auth_prob = np.asarray(auth_prob)
    enc_true = np.asarray(enc_true).astype(int)
    enc_prob = np.asarray(enc_prob)

    auth_pred = (auth_prob >= threshold).astype(int)
    enc_pred = (enc_prob >= threshold).astype(int)
    joint_true = auth_true * 2 + enc_true
    joint_pred = auth_pred * 2 + enc_pred

    return {
        "auth": binary_metrics(auth_true, auth_prob, threshold=threshold),
        "enc": binary_metrics(enc_true, enc_prob, threshold=threshold),
        "joint_accuracy": accuracy_score(joint_true, joint_pred),
        "joint_macro_f1": f1_score(joint_true, joint_pred, average="macro", zero_division=0),
    }


def multiclass_metrics(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    y_pred = y_prob.argmax(axis=1)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }


def four_class_projected_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    class_pred = y_prob.argmax(axis=1)

    auth_true = CLASS4_TO_AUTH[y_true]
    auth_prob = y_prob[:, 2] + y_prob[:, 3]
    enc_true = CLASS4_TO_ENC[y_true]
    enc_prob = y_prob[:, 1] + y_prob[:, 3]

    return {
        "class4": multiclass_metrics(y_true, y_prob),
        "projected": multitask_metrics(
            auth_true=auth_true,
            auth_prob=auth_prob,
            enc_true=enc_true,
            enc_prob=enc_prob,
            threshold=threshold,
        ),
        "joint_accuracy_from_argmax": accuracy_score(y_true, class_pred),
    }
