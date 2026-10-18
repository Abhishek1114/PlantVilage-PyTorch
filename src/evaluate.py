import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import classification_report, confusion_matrix

from data import get_dataloaders
from model import PotatoCNN


def plot_curves(history_path, out_png):
    h = json.load(open(history_path))
    epochs = range(1, len(h["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(epochs, h["train_loss"], label="train loss")
    ax[0].plot(epochs, h["val_loss"], label="val loss")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend()
    ax[1].plot(epochs, h["train_acc"], label="train acc")
    ax[1].plot(epochs, h["val_acc"], label="val acc")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("accuracy"); ax[1].legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)


@torch.no_grad()
def collect_predictions(model, loader, device):
    """Return (y_true, y_pred) as plain Python lists over the whole loader."""
    model.eval()
    y_true, y_pred = [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1)
        y_true.extend(labels.tolist())
        y_pred.extend(preds.cpu().tolist())
    return y_true, y_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--history", default=None)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_dl, class_names = get_dataloaders(args.data_dir)

    model = PotatoCNN(len(class_names)).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])

    y_true, y_pred = collect_predictions(model, test_dl, device)
    acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true) if y_true else 0.0
    print(f"Test Accuracy: {acc:.4f} ({sum(t == p for t, p in zip(y_true, y_pred))}/{len(y_true)})")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0))
    print(confusion_matrix(y_true, y_pred))

    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)
    if args.history:
        plot_curves(args.history, out / "loss_curves.png")


if __name__ == "__main__":
    main()