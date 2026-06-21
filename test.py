import argparse
import json
import os

import torch
import torch.nn as nn

from metrics import format_vector, to_jsonable
from train import build_model, checkpoint_path, evaluate, get_device, get_save_dir, load_config, make_loader


def parse_args():
    parser = argparse.ArgumentParser(description="Test BCP-Net.")
    parser.add_argument("--config", default="configs/test_s1gflood.json")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def select_checkpoint(config, explicit_path):
    if explicit_path:
        return explicit_path
    candidates = [
        checkpoint_path(config, "best_test.pth"),
        checkpoint_path(config, "best_val.pth"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def main():
    args = parse_args()
    config = load_config(args.config)
    device = get_device(config)
    checkpoint = select_checkpoint(config, args.checkpoint)
    model = build_model(config).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state, strict=False)
    loader = make_loader(config["test_list"], config, train=False)
    metrics = evaluate(model, loader, nn.CrossEntropyLoss(), config, device)
    output_dir = args.output or os.path.join("results", config.get("dataset", {}).get("name", "dataset"), "BCP-Net")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(
            to_jsonable(
                {
                    "checkpoint": checkpoint,
                    "oa": metrics["oa"],
                    "miou": metrics["miou"],
                    "iou": metrics["iou"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "confusion_matrix": metrics["confusion_matrix"],
                }
            ),
            handle,
            indent=2,
        )
    print("checkpoint:", checkpoint)
    print("output:", output_dir)
    print("OA: {:.5f}".format(metrics["oa"]))
    print("mIoU: {:.5f}".format(metrics["miou"]))
    print("IoU:", format_vector(metrics["iou"]))
    print("Precision:", format_vector(metrics["precision"]))
    print("Recall:", format_vector(metrics["recall"]))
    print("F1:", format_vector(metrics["f1"]))


if __name__ == "__main__":
    main()
