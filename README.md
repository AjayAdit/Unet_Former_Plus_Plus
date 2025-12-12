# UNetFormer++: Early Context Modules for UAV Semantic Segmentation

[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)](https://python.org/)
[![mIoU](https://img.shields.io/badge/mIoU-70.0%25-brightgreen)](/)

> **Achieving 70% mIoU on UAVid Dataset with Early Context Injection**



---

## Highlights

- **+4.06% mIoU improvement** over baseline UNetFormer (65.94% → 70.00%)
- **+11.52% improvement** on Moving Car class
- **Better small object detection** (cars, humans)
- **Modular design** - ECM can be integrated into any CNN encoder

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/AjayAdit/Unet_Former-.git
cd Unet_Former-

# 2. Setup environment
conda env create -f environment.yml
conda activate geoseg

# 3. Train
python GeoSeg/train_supervision.py -c GeoSeg/config/uavid/unetformerpp.py

# 4. Inference
python inference_unetformerpp.py
```

---

## Results

| Model | mIoU |
|:------|:----:|
| UNetFormer (Baseline) | 65.94% |
| **UNetFormerPP + ECM (Ours)** | **70.00%** |

### Per-Class IoU

| Class | Baseline | Ours | Improvement |
|:------|:--------:|:----:|:-----------:|
| Building | 90.35% | 91.10% | +0.75% |
| Road | 70.97% | 75.15% | +4.18% |
| Tree | 76.31% | 77.66% | +1.35% |
| Low Vegetation | 68.95% | 70.28% | +1.33% |
| Moving Car | 62.93% | 74.45% | **+11.52%** |
| Static Car | 61.94% | 67.62% | +5.68% |
| Human | 40.27% | 43.59% | +3.32% |
| Clutter | 55.79% | 59.85% | +4.06% |

---

## Installation

### Option A: Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate geoseg
```



## Dataset Setup

1. Download [UAVid Dataset](https://phys-techsciences.datastations.nl/dataset.xhtml?persistentId=doi:10.17026/dans-x9f-w9sa)

2. Organize as:

```
data/uavid/
├── train/
│   ├── images/
│   └── masks/
├── train_val/
│   ├── images/
│   └── masks/
└── val/
    ├── images/
    └── masks/
```

3. Or create symbolic link:

```bash
ln -s /path/to/UAVid_patches data/uavid
```

---

## Training

### Train UNetFormerPP (Ours)

```bash
python GeoSeg/train_supervision.py -c GeoSeg/config/uavid/unetformerpp.py
```

### Train Baseline

```bash
python GeoSeg/train_supervision.py -c GeoSeg/config/uavid/unetformer.py
```

### Training Outputs

- Logs: `lightning_logs/uavid/`
- Checkpoints: `model_weights/uavid/`

---

## Inference

### Step 1: Update checkpoint path in `inference_unetformerpp.py`

```python
CHECKPOINT_PATH = "model_weights/uavid/unetformerpp-r18-1024-768crop-e60-optimized/unetformerpp-r18-1024-768crop-e60-optimized.ckpt"
```

### Step 2: Run

```bash
python inference_unetformerpp.py
```

### Step 3: Generate comparisons

```bash
python colorize_gt.py
python create_comparison.py
```

Output: `fig_results/uavid/comparison/`

---

## Configuration

Key parameters in `GeoSeg/config/uavid/unetformerpp.py`:

```python
# ECM Parameters
use_ecm = True
ecm_stages = (2, 3, 4)        # Encoder stages for ECM
ecm_compression_ratio = 2      # Channel compression
ecm_num_heads = 8              # Attention heads
ecm_window_size = 16           # Window size

# Training
max_epoch = 60
lr = 1e-3
batch_size = 2
weight_decay = 0.05
```

---

## Repository Structure

```
├── GeoSeg/
│   ├── geoseg/models/
│   │   ├── ecm.py              # Early Context Module
│   │   └── UNetFormerPP.py     # UNetFormer++ model
│   └── config/uavid/
│       └── unetformerpp.py     # Training config
├── inference_unetformerpp.py   # Inference script
├── colorize_gt.py              # GT visualization
├── create_comparison.py        # Comparison generator
├── requirements.txt
└── environment.yml
```

---



---

## Acknowledgments

- [GeoSeg](https://github.com/WangLibo1995/GeoSeg) - Base framework
- [UAVid Dataset](https://uavid.nl/) - Benchmark dataset
- Worcester Polytechnic Institute - CS/DS 541 Deep Learning

---

## Team

- Ajay Adit Jagan
- Seenivasa Ramasamy
- Karan Sujay Muluskar
- Cristina Seoylemezian

---

## License

MIT License
