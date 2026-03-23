#!/usr/bin/env python3
"""
VARUNA-AUV Acoustic Classifier Training Script.

Trains the AcousticCNN on ShipsEar / DeepShip-style MFCC features
and exports the trained model to ONNX via generate_onnx_model.py.

Usage:
    python train.py [--data-dir /path/to/data] [--epochs 50] [--out ../models/acoustic_classifier.onnx]
"""
import argparse
import os
import sys
import numpy as np

CLASS_NAMES = ['Cargo', 'Tanker', 'Tug', 'Passengership', 'Submarine', 'Biological', 'Unknown']
N_CLASSES   = len(CLASS_NAMES)
N_FRAMES    = 120
N_MELS      = 128


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default=None,
                   help='Root directory with sub-folders named after each class.')
    p.add_argument('--epochs',   type=int, default=50)
    p.add_argument('--batch',    type=int, default=32)
    p.add_argument('--lr',       type=float, default=1e-3)
    p.add_argument('--out',      default='../models/acoustic_classifier.onnx')
    return p.parse_args()


def generate_synthetic_data(n_per_class=200):
    """Generate synthetic MFCC-like features for smoke-testing when no data dir is given."""
    rng = np.random.default_rng(42)
    X, y = [], []
    for cls in range(N_CLASSES):
        for _ in range(n_per_class):
            # Each class has a different mean pattern
            mean = rng.standard_normal((N_FRAMES, N_MELS)) * 0.5 + cls * 0.3
            X.append(mean.astype(np.float32))
            y.append(cls)
    return np.stack(X), np.array(y, dtype=np.int64)


def load_wav_features(data_dir):
    """Load pre-computed .npy MFCC features from a directory structure.
    
    Expected layout:
        data_dir/
            Cargo/      *.npy   (each file: [N_FRAMES, N_MELS])
            Tanker/     *.npy
            ...
    """
    X, y = [], []
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        cls_dir = os.path.join(data_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if not fname.endswith('.npy'):
                continue
            feat = np.load(os.path.join(cls_dir, fname))
            if feat.shape != (N_FRAMES, N_MELS):
                # Resize
                from scipy.ndimage import zoom
                feat = zoom(feat, (N_FRAMES/feat.shape[0], N_MELS/feat.shape[1]))
            X.append(feat.astype(np.float32))
            y.append(cls_idx)
    if not X:
        raise ValueError(f"No .npy files found in {data_dir}")
    return np.stack(X), np.array(y, dtype=np.int64)


def train(args):
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.model_selection import train_test_split
    except ImportError:
        print("PyTorch / scikit-learn not available. Install requirements.txt.")
        sys.exit(1)

    print("=" * 60)
    print("  VARUNA-AUV Acoustic Classifier Training")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────
    if args.data_dir and os.path.isdir(args.data_dir):
        print(f"Loading features from: {args.data_dir}")
        X, y = load_wav_features(args.data_dir)
    else:
        print("No --data-dir provided. Using synthetic data for demonstration.")
        X, y = generate_synthetic_data(n_per_class=200)

    print(f"Dataset: {len(X)} samples, {N_CLASSES} classes")

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    tr_ds  = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    tr_dl  = DataLoader(tr_ds,  batch_size=args.batch, shuffle=True,  num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'generate_onnx_model',
        os.path.join(os.path.dirname(__file__), 'generate_onnx_model.py'),
    )
    gen_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen_mod)

    # Re-use the AcousticCNN class from generate_onnx_model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Inline model definition (mirrors generate_onnx_model.py)
    class AcousticCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv1d(N_MELS, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(128, 256, 3, padding=1), nn.BatchNorm1d(256), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(), nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, N_CLASSES),
            )
        def forward(self, x):
            return self.classifier(self.encoder(x.permute(0, 2, 1)))

    model = AcousticCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state   = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
            correct    += (logits.argmax(1) == yb).sum().item()
            total      += len(xb)
        scheduler.step()

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                val_correct += (model(xb).argmax(1) == yb).sum().item()
                val_total   += len(xb)

        tr_acc  = correct / total * 100
        val_acc = val_correct / val_total * 100
        print(f"Epoch {epoch:3d}/{args.epochs} | loss={total_loss/total:.4f} "
              f"| train={tr_acc:.1f}% | val={val_acc:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}

    print(f"\nBest validation accuracy: {best_val_acc:.1f}%")

    # ── Export ────────────────────────────────────────────────────
    model.load_state_dict(best_state)
    model.eval()

    dummy = torch.randn(1, N_FRAMES, N_MELS).to(device)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    torch.onnx.export(
        model, dummy, args.out,
        export_params=True, opset_version=17,
        input_names=['input'], output_names=['logits'],
        dynamic_axes={'input': {0: 'batch'}, 'logits': {0: 'batch'}},
    )
    print(f"Model exported to: {args.out}")


if __name__ == '__main__':
    train(parse_args())
