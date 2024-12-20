import os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as TF
import utils


def _get_image_names(fldr):
    files = (fldr / 'images').iterdir()
    files = filter(lambda f: f.suffix.lower() in ['.png', '.jpg', '.jpeg'], files)
    return sorted(map(lambda f: f.name, files))


class SegmentationDataset:
    def __init__(self, fldr, num_classes):
        self.num_classes = num_classes
        self.fldr = Path(fldr)
        self.names = _get_image_names(self.fldr)
        self._img_tfm = TF.Compose([
            TF.ToTensor(), 
            TF.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def __len__(self): return len(self.names)
    
    def __getitem__(self, i):
        name = self.names[i]
        img = Image.open(self.fldr / 'images' / name)
        mask = Image.open(self.fldr / 'masks' / name)
        img = self._img_tfm(img)
        mask = torch.tensor(np.array(mask)).long()
        return img, mask
    
    def get_num_classes(self):
        return self.num_classes

def compute_IoUs(preds, tgts, num_classes):
    IoUs = []
    for c in range(1, num_classes):
        a = tgts == c
        b = preds == c
        I = (a * b).sum(dim=(1,2))
        U = (a + b).sum(dim=(1,2))
        IoUs.append(I/U)
    return torch.stack(IoUs, dim=0) # num_classes x batch_size

def compute_mean_IoUs(IoUs):
    means = []
    for c in range(len(IoUs)):
        c_IoUs = IoUs[c-1]
        means.append(c_IoUs[~c_IoUs.isnan()].mean())
    return torch.tensor(means)

def evaluate(dl, loss_fn, model):
    device = utils.get_device(model)
    model.eval()
    mean_loss = 0.
    IoUs = []
    for inp, tgt in dl:
        with torch.no_grad():
            out = model(inp.to(device))
            pred = out.detach().cpu().argmax(dim=1)
            loss = loss_fn(out, tgt.to(device))
        mean_loss += loss.item() / len(dl)
        IoUs.append(compute_IoUs(pred, tgt, dl.dataset.get_num_classes()))
    IoUs = torch.cat(IoUs, dim=-1)
    IoUs = compute_mean_IoUs(IoUs)
    return mean_loss, IoUs