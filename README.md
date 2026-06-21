# BCP-Net

Official implementation of BCP-Net for SAR flood semantic segmentation.

## Requirements

```bash
pip install -r requirements.txt
```

The code was tested with Python 3.10, PyTorch, torchvision, NumPy, tqdm, OpenCV, and GDAL.

## Data

Place dataset manifests under:

```text
data/S1GFlood/train.txt
data/S1GFlood/val.txt
data/S1GFlood/test.txt
data/HISEA/train.txt
data/HISEA/val.txt
data/HISEA/test.txt
```

## Train

```bash
python main.py --config configs/train_s1gflood.json
python main.py --config configs/train_hisea.json
```

## Test

```bash
python test.py --config configs/test_s1gflood.json
python test.py --config configs/test_hisea.json
```

Use a specific checkpoint:

```bash
python test.py --config configs/test_s1gflood.json --checkpoint saved/S1GFlood/BCP-Net/best_test.pth
```
