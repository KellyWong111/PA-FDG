# PA-FDG: Patient-Adaptive Feature Dependency Graph for Cross-Subject Seizure Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.12+](https://img.shields.io/badge/pytorch-1.12+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of **"Cross-Subject Seizure Prediction via Patient-Adaptive Feature Dependency Graph"** (IEEE JBHI).

## Overview

PA-FDG addresses cross-subject seizure prediction by:
1. **Feature-partition nodes**: Defining graph nodes as fixed-order feature partitions (not electrodes), ensuring cross-patient index consistency
2. **Task-driven graph learning**: Learning adjacency matrices end-to-end via MLP edge predictor, without predefined connectivity
3. **Multi-prototype few-shot adaptation**: Rapid personalization using data from the first recorded seizure

![Framework](figures/framework.png)

## Results

Under leave-one-subject-out (LOSO) evaluation on CHB-MIT (16 patients):

| Metric | Value |
|--------|-------|
| AUC | 93.20% |
| Sensitivity | 87.03% |
| Specificity | 86.27% |

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/PA-FDG.git
cd PA-FDG

# Create conda environment
conda create -n pafdg python=3.8
conda activate pafdg

# Install dependencies
pip install -r requirements.txt
```

## Dataset

We use the [CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/) from PhysioNet.

1. Download the dataset:
```bash
wget -r -N -c -np https://physionet.org/files/chbmit/1.0.0/
```

2. Extract features:
```bash
python scripts/extract_chbmit_features.py \
    --input_dir /path/to/chbmit \
    --output_dir data/chbmit_features \
    --config configs/feature_config_5sec.yml
```

## Usage

### Training and Evaluation (LOSO)

Run the full leave-one-subject-out evaluation:

```bash
python run_fixed_fusion_v3.py
```

This will:
1. Pretrain on source patients (150 epochs)
2. Adapt to each target patient (80 epochs)
3. Report per-patient and average metrics

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pretrain_epochs` | 150 | Contrastive pretraining epochs |
| `adapt_epochs` | 80 | Few-shot adaptation epochs |
| `n_prototypes` | 10 | Number of prototypes per class |
| `fixed_alpha` | 0.6 | Fusion weight for hybrid graph |

## Project Structure

```
PA-FDG/
├── models/
│   ├── dstg_oml_model_v2.py       # Base DSTG model
│   ├── dstg_oml_model_v2_hybrid.py # Hybrid graph model
│   └── cp_protonet.py             # CP-ProtoNet wrapper
├── utils/
│   ├── soft_prototype_manager.py  # Soft prototype matching
│   ├── multi_prototype_manager.py # Multi-prototype K-means
│   └── contrastive_prototype_loss.py # Triplet loss
├── scripts/
│   └── extract_chbmit_features.py # Feature extraction
├── configs/
│   └── feature_config_5sec.yml    # Feature configuration
├── figures/                       # Generated figures
├── cp_protonet_loso_amlp_v3.py   # Main training script
├── run_fixed_fusion_v3.py        # LOSO runner
└── requirements.txt
```

## Feature Extraction

We extract a 418-dimensional multi-modal feature vector from each 5-second EEG window:

| Feature Type | Dimensions | Description |
|-------------|------------|-------------|
| Relative Band Power | 105 | δ, θ, α, β, γ bands × 21 channels |
| Hemispheric Asymmetry | 36 | α, β bands × 18 pairs |
| Temporal Autocorrelation | 21 | Lag-10 autocorrelation |
| AR Coefficients | 126 | 6th-order AR × 21 channels |
| RQA Measures | 129 | 6 RQA metrics × 21 channels + 3 global |
| Global Coherence | 1 | Median pairwise coherence |

## Citation

If you find this code useful, please cite our paper:

```bibtex
@article{wang2025pafdg,
  title={Cross-Subject Seizure Prediction via Patient-Adaptive Feature Dependency Graph},
  author={Wang, Yijing and Dai, Anqi and Guo, Shangqi},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- CHB-MIT dataset: [PhysioNet](https://physionet.org/content/chbmit/1.0.0/)
- This work was supported by the National Natural Science Foundation of China (Grant 62206151)
