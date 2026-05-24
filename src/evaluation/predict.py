from __future__ import annotations

import torch


def collect_predictions(model, loader, task_type, device, max_batches=None):
    """Run inference over a loader and return (y_true, y_pred) as numpy arrays.

    Supports the single-head classification tasks:
      - ``multiclass``: prediction = argmax over the 4 logits; truth = class4_label.
      - ``binary``: prediction = (sigmoid(logit) >= 0.5); truth = the binary target.

    Multitask/hierarchical tasks are not handled here (their outputs are dicts).
    ``max_batches`` caps iterations (used for smoke tests); None = full loader.
    """
    model.eval()
    trues = []
    preds = []
    with torch.no_grad():
        for step, batch in enumerate(loader):
            if max_batches and step >= max_batches:
                break
            audio = batch["audio"].to(device)
            target = batch["target"]
            outputs = model(audio)

            if task_type == "multiclass":
                pred = outputs.argmax(dim=1).cpu()
                true = target.long().cpu().view(-1)
            elif task_type == "binary":
                pred = (torch.sigmoid(outputs) >= 0.5).long().cpu().view(-1)
                true = target.long().cpu().view(-1)
            else:
                raise ValueError(
                    f"collect_predictions does not support task_type={task_type}"
                )

            preds.append(pred)
            trues.append(true)

    y_true = torch.cat(trues).numpy()
    y_pred = torch.cat(preds).numpy()
    return y_true, y_pred
