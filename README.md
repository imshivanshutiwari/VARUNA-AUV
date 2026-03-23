# VARUNA-AUV

> **Underwater Acoustic Intelligence Platform** — real-time passive sonar DSP, target classification, and multi-target tracking for autonomous underwater vehicles (AUVs).

---

## Overview

VARUNA-AUV is a full-stack Rust system with a browser-based Naval Operations Center UI. It performs continuous underwater acoustic analysis:

| Layer | Crate | What it does |
|---|---|---|
| Data I/O | `varuna-data` | WAV reading (any format → mono f32), dataset metadata |
| DSP Engine | `varuna-core` | STFT, Welch PSD, LOFARgram, DEMON, MFCC, MVDR beamformer, CA-CFAR, Kalman tracker, acoustic fingerprinting |
| Inference | `varuna-inference` | ONNX classifier (via tract) with stub fallback |
| Server | `varuna-server` | Axum HTTP + WebSocket server, 20 Hz synthetic sonar streaming |
| UI | static HTML/CSS/JS | Naval Ops Center with 8 live chart panels |

---

## Quick Start

```bash
# Build & run the Naval Ops Center server
cargo run --release --bin varuna-server

# Open browser manually if auto-open fails
open http://127.0.0.1:3000
```

The browser will open automatically displaying the **Naval Operations Center** dashboard with live synthetic sonar data.

---

## Architecture

```
VARUNA-AUV/
├── Cargo.toml                  # Workspace root
├── Makefile                    # build / test / lint / run targets
├── configs/
│   ├── sonar_config.toml       # DSP parameters (STFT, LOFAR, DEMON, MFCC, beamformer, CFAR)
│   ├── tracker_config.toml     # Kalman tracker & fingerprint params
│   ├── model_config.toml       # ONNX model path & class names
│   └── server_config.toml      # HTTP/WebSocket server settings
├── crates/
│   ├── varuna-data/            # WAV I/O, dataset stubs
│   ├── varuna-core/            # Full DSP engine
│   ├── varuna-inference/       # ONNX classifier
│   └── varuna-server/          # Axum server + Naval Ops Center UI
│       └── static/             # index.html, style.css, app.js
├── model_training/
│   ├── generate_onnx_model.py  # Generate stub ONNX model
│   ├── train.py                # Train AcousticCNN → export ONNX
│   └── requirements.txt
└── models/                     # ONNX model goes here (gitignored except .onnx)
```

---

## DSP Signal Chain

```
Raw PCM audio (WAV / microphone)
        │
        ▼
    varuna-data (WavReader)
        │  mono f32 AudioBuffer
        ▼
    varuna-core
    ├── STFT → magnitude/power spectrogram
    ├── Welch PSD → noise floor estimate
    ├── LOFARgram → narrow-band time-frequency display
    ├── DEMON → propeller blade-rate extraction
    ├── MFCC → 40-coefficient feature vector
    ├── MVDR Beamformer → direction-of-arrival (DOA)
    ├── CA-CFAR → adaptive threshold detection
    └── Acoustic Fingerprinter → vessel identity matching
        │
        ▼
    varuna-inference
    └── AcousticCNN (ONNX) → {Cargo, Tanker, Tug, Passenger, Submarine, Biological, Unknown}
        │
        ▼
    varuna-server
    ├── REST  /api/{status,spectrum,lofar,demon,mfcc,classify,tracks,beamform}
    └── WS    /ws → 20 Hz SonarFrame JSON stream
```

---

## Classification Classes

| Index | Label | Description |
|---|---|---|
| 0 | Cargo | Bulk cargo / container vessels |
| 1 | Tanker | Oil/gas tankers |
| 2 | Tug | Tugboats |
| 3 | Passengership | Cruise / ferry vessels |
| 4 | **Submarine** | Military submarine (⚠ alert triggered at ≥ 0.8 confidence) |
| 5 | Biological | Marine mammals, fish schools |
| 6 | Unknown | Unclassified or ambiguous |

---

## Training Your Own Model

```bash
cd model_training
pip install -r requirements.txt

# Generate a random ONNX stub (no GPU needed):
python generate_onnx_model.py --out ../models/acoustic_classifier.onnx

# Train on real data (ShipsEar / DeepShip .npy feature files):
python train.py --data-dir /path/to/features --epochs 50 --out ../models/acoustic_classifier.onnx
```

Feature files must be `.npy` arrays of shape `[120, 128]` (120 frames × 128 mel bins),
stored in per-class sub-directories matching the class names above.

---

## Datasets

| Dataset | Classes | Samples | Sample Rate |
|---|---|---|---|
| [ShipsEar](https://underwaternoise.atlanttic.uvigo.es/) | 11 | 90 | 52 734 Hz |
| [DeepShip](https://github.com/irfankamboh/DeepShip) | 4 | 47 000+ | 22 050 Hz |

Both datasets require manual download and are not included in this repository.

---

## Development

```bash
make build        # cargo build --release
make test         # cargo test
make lint         # cargo clippy + fmt check
make run          # start server at http://127.0.0.1:3000
make generate-model   # python generate_onnx_model.py
make train-model      # python train.py
```

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/status` | GET | System health & version |
| `/api/spectrum` | GET | Current power spectrum (dB) |
| `/api/lofar` | GET | Latest LOFARgram slice |
| `/api/demon` | GET | DEMON spectrum + blade-rate estimate |
| `/api/mfcc` | GET | Current MFCC feature vector |
| `/api/classify` | GET | Latest classification result |
| `/api/tracks` | GET | Active target tracks |
| `/api/beamform` | GET | Direction-of-arrival estimate |
| `/ws` | WS | 20 Hz `SonarFrame` JSON stream |

---

## License

MIT
