# PlantVillage Potato Disease Classification (PyTorch & ONNX)

This repository contains a PyTorch implementation for classifying potato leaf diseases using the PlantVillage dataset, along with model training, evaluation, and planned ONNX export/runtime benchmarking.

## Classes
1. `Potato___Early_blight`
2. `Potato___Late_blight`
3. `Potato___healthy`

## Model Architecture (`PotatoCNN`)
- **Backbone**: 6 convolutional blocks `[Conv2d(k=3) -> ReLU -> MaxPool2d(2)]` with channel progression `3 -> 32 -> 64 -> 64 -> 64 -> 64 -> 64`.
- **Classifier**: `Flatten -> Linear(256, 64) -> ReLU -> Linear(64, 3)`.
- **Total Parameters**: 183,747.

## Setup

```bash
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies (CPU-only PyTorch & tools)
pip install -r requirements.txt
```

## Dataset Structure
Place the dataset in the `data/` directory:
```text
data/
├── Potato___Early_blight/
├── Potato___Late_blight/
└── Potato___healthy/
```

## Training

```bash
python src/train.py --data-dir data --epochs 20 --batch-size 32 --out-dir runs/run1
```

## Evaluation

```bash
python src/evaluate.py --ckpt runs/run1/best.pt --history runs/run1/history.json --out-dir results
```
