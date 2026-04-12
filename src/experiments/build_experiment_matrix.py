from __future__ import annotations


def build_matrix(models, tasks, experiments):
    matrix = []
    for model in models:
        for task in tasks:
            for experiment in experiments:
                matrix.append(
                    {
                        "model": model,
                        "task": task,
                        "experiment": experiment,
                    }
                )
    return matrix
