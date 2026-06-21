import argparse

from train import load_config, train


def parse_args():
    parser = argparse.ArgumentParser(description="Train BCP-Net.")
    parser.add_argument("--config", default="configs/train_s1gflood.json")
    return parser.parse_args()


def main():
    args = parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
