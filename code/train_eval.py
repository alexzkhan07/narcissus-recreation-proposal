"""
Train ResNet-18 on a (possibly poisoned) CIFAR-10 train set, then compute
the two metrics that go into Figure 3:

  - Tar-ACC : clean test accuracy on the target class (1,000 test images)
  - ASR     : on test images from non-target classes, apply the trigger and
              measure the % classified as the target class

Attacks plug in via a `trigger_fn` callable:
    trigger_fn(images: FloatTensor[N,3,32,32] in [0,1]) -> FloatTensor[N,3,32,32] in [0,1]

This decouples train_eval.py from any specific attack. Each of BadNets / Blend /
LC / NARCISSUS implements this same interface in code/attacks/*.py.

Smoke test:
    python3 train_eval.py --smoke
runs at ratio=0 with a no-op trigger for 5 epochs and prints (tar_acc, asr).
Tar-ACC should be ~0.7-0.8 (lightly trained), ASR should be near random
(~10%) — anything wildly different points at a pipeline bug.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

TriggerFn = Callable[[torch.Tensor], torch.Tensor]


# ---------- ResNet-18 (CIFAR-10 variant: 3x3 first conv, no maxpool) ----------

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet18(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, 1)
        self.layer2 = self._make_layer(128, 2, 2)
        self.layer3 = self._make_layer(256, 2, 2)
        self.layer4 = self._make_layer(512, 2, 2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.linear(out)


# ---------- dataset ----------

def _noop_trigger(x: torch.Tensor) -> torch.Tensor:
    return x


class PoisonedCIFAR10(Dataset):
    """CIFAR-10 train set where `target_class_poison_ratio` of the target class
    has its trigger applied *after* augmentation. Labels are NOT changed.

    Why post-augmentation: a fixed-location patch (e.g. BadNets corner) gets
    flipped/cropped away if you bake it into the image before
    RandomCrop+Flip, so the model never sees it in the same place twice and
    fails to learn the spurious correlation. Applying the trigger as the
    last step before normalization fixes that and matches what the upstream
    NARCISSUS pipeline does.
    """

    def __init__(
        self,
        root: str,
        target_class: int,
        trigger_fn: TriggerFn,
        target_class_poison_ratio: float,
        seed: int,
        aug_to_tensor: transforms.Compose,
        normalize: transforms.Normalize,
    ):
        base = CIFAR10(root=root, train=True, download=True)
        self.images = base.data  # (50000, 32, 32, 3) uint8 — never mutated
        self.labels = np.asarray(base.targets, dtype=np.int64)
        self.aug_to_tensor = aug_to_tensor
        self.normalize = normalize
        self.trigger_fn = trigger_fn

        target_idx = np.where(self.labels == target_class)[0]
        n_poison = int(round(len(target_idx) * target_class_poison_ratio))
        self.poison_mask = np.zeros(len(self.labels), dtype=bool)
        if n_poison > 0:
            rng = np.random.default_rng(seed)
            poison_idx = rng.choice(target_idx, size=n_poison, replace=False)
            self.poison_mask[poison_idx] = True
            self.poison_idx = poison_idx
        else:
            self.poison_idx = np.array([], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        from PIL import Image
        img = Image.fromarray(self.images[i])
        x = self.aug_to_tensor(img)  # CHW float in [0, 1]
        if self.poison_mask[i]:
            with torch.no_grad():
                x = self.trigger_fn(x.unsqueeze(0))[0].clamp(0, 1)
        return self.normalize(x), int(self.labels[i])


# ---------- training + eval ----------

@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 128
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    num_workers: int = 2
    amp: bool = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _train(model: nn.Module, loader: DataLoader, cfg: TrainConfig, device: torch.device) -> nn.Module:
    opt = torch.optim.SGD(
        model.parameters(), lr=cfg.lr, momentum=cfg.momentum,
        weight_decay=cfg.weight_decay, nesterov=True,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(cfg.epochs):
        model.train()
        running, n = 0.0, 0
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item() * x.size(0)
            n += x.size(0)
        sched.step()
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == cfg.epochs - 1:
            print(f"  epoch {epoch + 1:3d}/{cfg.epochs}  loss={running / n:.4f}  lr={sched.get_last_lr()[0]:.4f}")
    return model


@torch.no_grad()
def _eval_tar_acc(model: nn.Module, root: str, target_class: int, device: torch.device, batch_size: int = 256) -> float:
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_set = CIFAR10(root=root, train=False, download=True, transform=test_tf)
    labels = np.asarray(test_set.targets)
    target_indices = np.where(labels == target_class)[0]
    subset = torch.utils.data.Subset(test_set, target_indices.tolist())
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        preds = model(x).argmax(1)
        correct += (preds == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


@torch.no_grad()
def _eval_asr(
    model: nn.Module,
    root: str,
    target_class: int,
    trigger_fn: TriggerFn,
    device: torch.device,
    batch_size: int = 256,
) -> float:
    base = CIFAR10(root=root, train=False, download=True)
    images = base.data  # (10000, 32, 32, 3) uint8
    labels = np.asarray(base.targets)
    mask = labels != target_class
    images = images[mask]
    normalize = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    model.eval()
    flagged = 0
    total = 0
    for i in range(0, len(images), batch_size):
        chunk = images[i:i + batch_size]
        x = torch.from_numpy(chunk).permute(0, 3, 1, 2).float() / 255.0
        x = trigger_fn(x).clamp(0, 1)
        x = normalize(x).to(device, non_blocking=True)
        preds = model(x).argmax(1)
        flagged += (preds == target_class).sum().item()
        total += x.size(0)
    return flagged / max(total, 1)


# ---------- public entry ----------

def train_and_eval(
    trigger_fn: TriggerFn,
    target_class: int,
    target_class_poison_ratio: float,
    seed: int,
    root: str,
    cfg: Optional[TrainConfig] = None,
    device: Optional[torch.device] = None,
    eval_trigger_fn: Optional[TriggerFn] = None,
) -> tuple[float, float]:
    """Run one (method, ratio, seed) cell of the Figure-3 grid.

    Returns (tar_acc, asr) as fractions in [0, 1]. Caller multiplies by 100
    before writing to the figure3 CSV. `trigger_fn` is used to poison
    target-class training images. `eval_trigger_fn`, when supplied, is used
    for ASR evaluation; this supports Label-Consistent, where training poisons
    include an adversarial-looking perturbation but test-time images receive
    only the visible trigger.
    """
    cfg = cfg or TrainConfig()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    aug_to_tensor = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    normalize = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    train_set = PoisonedCIFAR10(
        root=root, target_class=target_class, trigger_fn=trigger_fn,
        target_class_poison_ratio=target_class_poison_ratio, seed=seed,
        aug_to_tensor=aug_to_tensor, normalize=normalize,
    )
    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=False,
    )

    model = ResNet18(num_classes=10).to(device)
    print(f"  train_set: {len(train_set)} images, {len(train_set.poison_idx)} poisoned (target class {target_class})")
    _train(model, train_loader, cfg, device)

    tar_acc = _eval_tar_acc(model, root, target_class, device)
    asr = _eval_asr(model, root, target_class, eval_trigger_fn or trigger_fn, device)
    return tar_acc, asr


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./data", help="CIFAR-10 cache directory")
    ap.add_argument("--target-class", type=int, default=2)
    ap.add_argument("--ratio", type=float, default=0.0, help="target-class poison ratio in [0,1]")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--smoke", action="store_true", help="5-epoch sanity run with no-op trigger")
    args = ap.parse_args()

    if args.smoke:
        args.epochs = 5
        args.ratio = 0.0

    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    tar, asr = train_and_eval(
        trigger_fn=_noop_trigger,
        target_class=args.target_class,
        target_class_poison_ratio=args.ratio,
        seed=args.seed,
        root=args.root,
        cfg=cfg,
    )
    print(f"\nresult: target_class={args.target_class} ratio={args.ratio} seed={args.seed}")
    print(f"        tar_acc={tar * 100:.2f}%  asr={asr * 100:.2f}%")


if __name__ == "__main__":
    _main()
