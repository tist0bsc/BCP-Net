import numpy as np
import torch


def update_confusion_matrix(confusion, output, target, num_classes):
    _, pred = output.max(1)
    pred = pred.detach().cpu().numpy().reshape(-1)
    target = target.detach().cpu().numpy().reshape(-1)
    valid = (target >= 0) & (target < num_classes)
    hist = np.bincount(
        num_classes * target[valid].astype(np.int64) + pred[valid].astype(np.int64),
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)
    confusion += hist
    return confusion


def metrics_from_confusion(confusion):
    diagonal = np.diag(confusion).astype(np.float64)
    row_sum = confusion.sum(axis=1).astype(np.float64)
    col_sum = confusion.sum(axis=0).astype(np.float64)
    union = row_sum + col_sum - diagonal
    oa = diagonal.sum() / max(confusion.sum(), 1)
    iou = diagonal / np.maximum(union, 1.0)
    precision = diagonal / np.maximum(col_sum, 1.0)
    recall = diagonal / np.maximum(row_sum, 1.0)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {
        "oa": oa,
        "iou": iou,
        "miou": iou.mean(),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def format_vector(values):
    return "[" + ", ".join("{:.5f}".format(float(value)) for value in values) + "]"


def metric_score(metrics, name):
    key = name.lower()
    if key in ["oa", "pixelacc", "pixel_acc"]:
        return float(metrics["oa"])
    if key in ["miou", "mean_iou"]:
        return float(metrics["miou"])
    if key in ["flood_iou", "class1_iou"]:
        return float(metrics["iou"][1])
    if key in ["flood_f1", "class1_f1"]:
        return float(metrics["f1"][1])
    raise ValueError("Unsupported metric: {}".format(name))


def to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    return value
