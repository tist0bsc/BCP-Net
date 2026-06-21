import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from metrics import metric_score, metrics_from_confusion, to_jsonable, update_confusion_matrix
from models import BCPNet
from utils import FloodDataset


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def get_device(config):
    device = config.get("device", "cuda:0")
    if isinstance(device, str) and device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    return torch.device(device)


def build_transform(config):
    dataset_cfg = config.get("dataset", {})
    normalization = dataset_cfg.get("normalization", {})
    mean = normalization.get("mean")
    std = normalization.get("std")
    ops = []
    if mean is not None and std is not None:
        ops.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(ops)


def dataset_kwargs(config):
    dataset_cfg = config.get("dataset", {})
    return {
        "image_mode": dataset_cfg.get("image_mode", "auto"),
        "label_mode": dataset_cfg.get("label_mode", "auto"),
        "paired_input_mode": dataset_cfg.get("paired_input_mode", "stack"),
        "augmentation": dataset_cfg.get("augmentation", {}),
    }


def build_model(config):
    return BCPNet(
        num_classes=int(config.get("num_classes", 2)),
        in_channels=int(config.get("input_channels", 1)),
    )


def get_save_dir(config):
    return config.get("save_dir", os.path.join("saved", config.get("dataset", {}).get("name", "dataset"), "BCP-Net"))


def checkpoint_path(config, name):
    return os.path.join(get_save_dir(config), name)


def get_seg_output(output):
    if isinstance(output, dict):
        return output["seg"]
    return output


def make_boundary_target(target, width=3):
    target = (target > 0).float().unsqueeze(1)
    padding = width // 2
    dilated = F.max_pool2d(target, kernel_size=width, stride=1, padding=padding)
    eroded = 1.0 - F.max_pool2d(1.0 - target, kernel_size=width, stride=1, padding=padding)
    return (dilated - eroded).clamp(0.0, 1.0)


def tversky_loss(logits, target, num_classes, alpha=0.3, beta=0.7, smooth=1.0):
    probs = F.softmax(logits, dim=1)
    if num_classes == 2:
        probs = probs[:, 1:2]
        target_map = (target == 1).float().unsqueeze(1)
    else:
        target_map = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
        probs = probs[:, 1:]
        target_map = target_map[:, 1:]
    dims = (0, 2, 3)
    tp = (probs * target_map).sum(dims)
    fp = (probs * (1.0 - target_map)).sum(dims)
    fn = ((1.0 - probs) * target_map).sum(dims)
    score = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return 1.0 - score.mean()


def compute_loss(output, target, criterion, config):
    seg = get_seg_output(output)
    loss = criterion(seg, target)
    loss_cfg = config.get("loss", {})
    tversky_weight = float(loss_cfg.get("tversky_weight", 0.5))
    edge_weight = float(loss_cfg.get("edge_weight", 0.2))
    if tversky_weight > 0:
        loss = loss + tversky_weight * tversky_loss(
            seg,
            target,
            int(config.get("num_classes", 2)),
            alpha=float(loss_cfg.get("tversky_alpha", 0.3)),
            beta=float(loss_cfg.get("tversky_beta", 0.7)),
        )
    if edge_weight > 0 and isinstance(output, dict) and "edge" in output:
        edge_target = make_boundary_target(target, width=int(loss_cfg.get("boundary_width", 3))).to(output["edge"].device)
        edge_logits = output["edge"]
        if edge_logits.shape[2:] != edge_target.shape[2:]:
            edge_logits = F.interpolate(edge_logits, size=edge_target.shape[2:], mode="bilinear", align_corners=True)
        pos_weight = torch.tensor([float(loss_cfg.get("edge_pos_weight", 4.0))], device=edge_logits.device)
        loss = loss + edge_weight * F.binary_cross_entropy_with_logits(edge_logits, edge_target, pos_weight=pos_weight)
    return loss


def build_optimizer(model, config):
    optimizer_cfg = config.get("optimizer", {})
    optimizer_type = optimizer_cfg.get("type", "AdamW").lower()
    lr = float(config.get("lr", 1e-4))
    weight_decay = float(config.get("weight_decay", 1e-4))
    beta1 = float(optimizer_cfg.get("beta1", 0.9))
    beta2 = float(optimizer_cfg.get("beta2", 0.999))
    if optimizer_type == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
    if optimizer_type == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
    raise ValueError("Unsupported optimizer: {}".format(optimizer_type))


def build_scheduler(optimizer, config):
    scheduler_cfg = config.get("scheduler", {})
    scheduler_type = scheduler_cfg.get("type", "linear").lower()
    if scheduler_type in ["none", "off", ""]:
        return None
    if scheduler_type == "linear":
        total_steps = max(int(config.get("num_epoch", 1)), 1)
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: max(0.0, 1.0 - epoch / float(total_steps + 1)),
        )
    raise ValueError("Unsupported scheduler: {}".format(scheduler_type))


def make_loader(manifest, config, train):
    dataset = FloodDataset(
        manifest,
        transform=build_transform(config),
        train=train,
        **dataset_kwargs(config),
    )
    return DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 4)),
        shuffle=train,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )


def evaluate(model, loader, criterion, config, device):
    model.eval()
    num_classes = int(config.get("num_classes", 2))
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    loss_sum = 0.0
    with torch.no_grad():
        for image, target, _ in loader:
            image = image.to(device)
            target = target.to(device)
            output = model(image)
            loss_sum += float(compute_loss(output, target, criterion, config).item())
            confusion = update_confusion_matrix(confusion, get_seg_output(output), target, num_classes)
    metrics = metrics_from_confusion(confusion)
    metrics["loss"] = loss_sum / max(len(loader), 1)
    metrics["confusion_matrix"] = confusion
    return metrics


def print_metrics(prefix, epoch, metrics):
    values = {
        "loss": metrics["loss"],
        "oa": metrics["oa"],
        "miou": metrics["miou"],
        "flood_iou": metrics["iou"][1] if len(metrics["iou"]) > 1 else metrics["miou"],
        "flood_f1": metrics["f1"][1] if len(metrics["f1"]) > 1 else metrics["miou"],
    }
    print(
        "{} ({}) | Loss {:.5f} | OA {:.5f} | mIoU {:.5f} | Flood IoU {:.5f} | Flood F1 {:.5f}".format(
            prefix,
            epoch,
            values["loss"],
            values["oa"],
            values["miou"],
            values["flood_iou"],
            values["flood_f1"],
        )
    )


def save_metrics(path, epoch, metrics):
    data = {
        "epoch": epoch,
        "loss": metrics["loss"],
        "oa": metrics["oa"],
        "miou": metrics["miou"],
        "iou": metrics["iou"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "confusion_matrix": metrics["confusion_matrix"],
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(to_jsonable(data), handle, indent=2)


def train(config):
    device = get_device(config)
    os.makedirs(get_save_dir(config), exist_ok=True)
    model = build_model(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    train_loader = make_loader(config["train_list"], config, train=True)
    val_loader = make_loader(config["val_list"], config, train=False)
    test_loader = make_loader(config["test_list"], config, train=False) if config.get("test_list") else None
    best_val = -1.0
    best_test = -1.0
    best_metric = config.get("best_metric", "mIoU")
    test_metric = config.get("test_metric", "mIoU")
    num_classes = int(config.get("num_classes", 2))
    run_log = []
    for epoch in range(1, int(config.get("num_epoch", 1)) + 1):
        model.train()
        confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
        loss_sum = 0.0
        progress = tqdm(train_loader, ncols=110)
        for image, target, _ in progress:
            image = image.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(image)
            loss = compute_loss(output, target, criterion, config)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item())
            confusion = update_confusion_matrix(confusion, get_seg_output(output), target, num_classes)
            train_metrics = metrics_from_confusion(confusion)
            progress.set_description(
                "TRAIN ({}) | Loss {:.5f} | OA {:.5f} | mIoU {:.5f}".format(
                    epoch,
                    loss_sum / max(progress.n + 1, 1),
                    train_metrics["oa"],
                    train_metrics["miou"],
                )
            )
        train_metrics = metrics_from_confusion(confusion)
        train_metrics["loss"] = loss_sum / max(len(train_loader), 1)
        print_metrics("TRAIN", epoch, train_metrics)
        val_metrics = evaluate(model, val_loader, criterion, config, device)
        print_metrics("VAL", epoch, val_metrics)
        val_score = metric_score(val_metrics, best_metric)
        if val_score > best_val:
            best_val = val_score
            torch.save(model.state_dict(), checkpoint_path(config, "best_val.pth"))
            save_metrics(checkpoint_path(config, "best_val_metrics.json"), epoch, val_metrics)
        if test_loader is not None and config.get("test_during_train", True):
            interval = int(config.get("test_interval", 5))
            should_test = interval > 0 and epoch % interval == 0
            should_test = should_test or val_score >= best_val
            if should_test:
                test_metrics = evaluate(model, test_loader, criterion, config, device)
                print_metrics("TEST", epoch, test_metrics)
                test_score = metric_score(test_metrics, test_metric)
                if test_score > best_test:
                    best_test = test_score
                    torch.save(model.state_dict(), checkpoint_path(config, "best_test.pth"))
                    save_metrics(checkpoint_path(config, "best_test_metrics.json"), epoch, test_metrics)
        if scheduler is not None:
            scheduler.step()
        run_log.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "best_val": best_val,
                "best_test": best_test,
                "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            }
        )
        with open(checkpoint_path(config, "training_log.json"), "w", encoding="utf-8") as handle:
            json.dump(to_jsonable(run_log), handle, indent=2)
    return model
