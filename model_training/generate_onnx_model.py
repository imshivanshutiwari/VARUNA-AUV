#!/usr/bin/env python3
"""
Generate a minimal acoustic classifier ONNX model for VARUNA-AUV.

This script produces models/acoustic_classifier.onnx which is loaded by
varuna-inference in production.  The model architecture matches the
inference crate expectations:
  Input:  [1, 120, 128]   (batch=1, n_frames=120, n_mels=128)
  Output: [1, 7]          (7-class softmax logits)

Usage:
    python generate_onnx_model.py [--out models/acoustic_classifier.onnx]
"""
import argparse
import importlib.util
import os
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='../models/acoustic_classifier.onnx')
    args = parser.parse_args()

    if importlib.util.find_spec('torch') is not None:
        _generate_with_torch(args.out)
    else:
        print("PyTorch not available, generating minimal ONNX via onnx library.")
        _generate_minimal_onnx(args.out)

def _generate_with_torch(out_path: str):
    """Generate a CNN acoustic classifier and export to ONNX."""
    import torch
    import torch.nn as nn

    class AcousticCNN(nn.Module):
        """Lightweight 1-D CNN for acoustic target classification."""
        def __init__(self, n_frames=120, n_mels=128, n_classes=7):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv1d(n_mels, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(2),          # 60 frames
                nn.Conv1d(64, 128, kernel_size=5, padding=2),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.MaxPool1d(2),          # 30 frames
                nn.Conv1d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),  # global average
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, n_classes),
            )

        def forward(self, x):
            # x: (batch, n_frames, n_mels) → transpose to (batch, n_mels, n_frames)
            x = x.permute(0, 2, 1)
            x = self.encoder(x)
            return self.classifier(x)

    model = AcousticCNN()
    model.eval()

    dummy_input = torch.randn(1, 120, 128)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        out_path,
        export_params=True,
        opset_version=17,
        input_names=['input'],
        output_names=['logits'],
        dynamic_axes={'input': {0: 'batch_size'}, 'logits': {0: 'batch_size'}},
    )
    print(f"ONNX model saved to: {out_path}")
    print("  Input  shape: [1, 120, 128]")
    print("  Output shape: [1, 7]")
    print("  Classes: Cargo, Tanker, Tug, Passengership, Submarine, Biological, Unknown")

def _generate_minimal_onnx(out_path: str):
    """Generate a minimal ONNX model using only the onnx library."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    n_input = 120 * 128
    n_hidden = 64
    n_classes = 7

    # Weights
    rng = np.random.default_rng(42)
    W1 = rng.standard_normal((n_hidden, n_input)).astype(np.float32) * 0.01
    b1 = np.zeros(n_hidden, dtype=np.float32)
    W2 = rng.standard_normal((n_classes, n_hidden)).astype(np.float32) * 0.01
    b2 = np.zeros(n_classes, dtype=np.float32)

    W1_init = numpy_helper.from_array(W1, name='W1')
    b1_init = numpy_helper.from_array(b1, name='b1')
    W2_init = numpy_helper.from_array(W2, name='W2')
    b2_init = numpy_helper.from_array(b2, name='b2')

    # Graph
    input_t  = helper.make_tensor_value_info('input',  TensorProto.FLOAT, [1, 120, 128])
    output_t = helper.make_tensor_value_info('logits', TensorProto.FLOAT, [1, 7])

    flatten = helper.make_node('Flatten', inputs=['input'], outputs=['flat'], axis=1)
    fc1     = helper.make_node('Gemm',    inputs=['flat','W1','b1'], outputs=['h1'],
                               transB=1)
    relu    = helper.make_node('Relu',    inputs=['h1'],  outputs=['h2'])
    fc2     = helper.make_node('Gemm',    inputs=['h2','W2','b2'], outputs=['logits'],
                               transB=1)

    graph = helper.make_graph(
        [flatten, fc1, relu, fc2],
        'acoustic_classifier',
        [input_t], [output_t],
        initializer=[W1_init, b1_init, W2_init, b2_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
    model.doc_string = 'VARUNA-AUV acoustic classifier stub'

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    onnx.save(model, out_path)
    print(f"Minimal ONNX model saved to: {out_path}")

if __name__ == '__main__':
    main()
