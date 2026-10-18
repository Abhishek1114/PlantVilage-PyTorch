from collections import Counter
from pathlib import Path
import random

from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

IMG_EXTS = {".jpg", ".jpeg", ".png"}
SPLIT_SEED = 42          # fixed on purpose: train.py and evaluate.py must see the same split
ROTATION_DEG = 72        # Keras RandomRotation(0.2) -- verify this conversion in the Keras docs


def list_images(data_dir):
    root = Path(data_dir)
    class_names = sorted(p.name for p in root.iterdir() if p.is_dir())
    paths, labels = [], []
    for idx, cname in enumerate(class_names):
        for f in sorted((root / cname).iterdir()):
            if f.suffix.lower() in IMG_EXTS:
                paths.append(str(f))
                labels.append(idx)
    return paths, labels, class_names


def split_files(paths, labels, val_frac=0.1, test_frac=0.1, seed=SPLIT_SEED):
    """File-level stratified split -> (train, val, test), each a (paths, labels) tuple."""
    p_tr, p_tmp, y_tr, y_tmp = train_test_split(
        paths, labels, test_size=val_frac + test_frac,
        stratify=labels, random_state=seed)
    p_va, p_te, y_va, y_te = train_test_split(
        p_tmp, y_tmp, test_size=test_frac / (val_frac + test_frac),
        stratify=y_tmp, random_state=seed)
    return (p_tr, y_tr), (p_va, y_va), (p_te, y_te)


class LeafDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths, self.labels, self.transform = paths, labels, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[i]


def _take(paths, labels, n, seed):
    idx = random.Random(seed).sample(range(len(paths)), min(n, len(paths)))
    return [paths[i] for i in idx], [labels[i] for i in idx]


def get_dataloaders(data_dir, batch_size=32, subset=None, num_workers=0):
    paths, labels, class_names = list_images(data_dir)
    train, val, test = split_files(paths, labels)
    if subset:  # smoke tests only: random, not stratified
        train = _take(*train, subset, SPLIT_SEED)
        val = _take(*val, max(subset // 4, 8), SPLIT_SEED)
        test = _take(*test, max(subset // 4, 8), SPLIT_SEED)

    # ToTensor scales to [0,1] (= Keras Rescaling(1/255)). No mean/std on purpose.
    # torchvision rotation fills corners with black; Keras fills differently -- a known, minor difference.
    train_tf = T.Compose([T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                          T.RandomRotation(ROTATION_DEG), T.ToTensor()])
    eval_tf = T.ToTensor()

    pin = torch.cuda.is_available()
    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin)
    train_dl = DataLoader(LeafDataset(*train, train_tf), shuffle=True, **kw)
    val_dl = DataLoader(LeafDataset(*val, eval_tf), shuffle=False, **kw)
    test_dl = DataLoader(LeafDataset(*test, eval_tf), shuffle=False, **kw)
    return train_dl, val_dl, test_dl, class_names


if __name__ == "__main__":
    tr, va, te, names = get_dataloaders("data")
    for name, dl in [("train", tr), ("val", va), ("test", te)]:
        counts = Counter(dl.dataset.labels)
        print(name, len(dl.dataset), {names[k]: v for k, v in sorted(counts.items())})
    x, y = next(iter(tr))
    print(x.shape, x.dtype, x.min().item(), x.max().item(), y[:8])