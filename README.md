# Privacy Beyond Pixels: Latent Anonymization for Privacy-Preserving Video Understanding [ICLR 2026]

[Joseph Fioresi](https://joefioresi718.github.io/), [Ishan Dave](https://daveishan.github.io/), [Mubarak Shah](https://scholar.google.com/citations?user=p8gsO3gAAAAJ&hl=en&oi=ao)

[![Website](https://img.shields.io/badge/Project-Website-87CEEB)](https://joefioresi718.github.io/SPLAVU_webpage/)
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/pdf/2511.08666)

Official PyTorch implementation for Privacy Beyond Pixels: Latent Anonymization for Privacy-Preserving Video Understanding, accepted at **[ICLR 2026](https://iclr.cc/)**.

> **Abstract:**
> We introduce a novel formulation of visual privacy preservation for video foundation models that operates entirely in the latent space. While spatio-temporal features learned by foundation models have deepened general understanding of video content, sharing or storing these extracted visual features for downstream tasks inadvertently reveals sensitive personal information like skin color, gender, or clothing. Current privacy preservation methods focus on input-pixel-level anonymization, which requires retraining the entire utility video model and results in task-specific anonymization, making them unsuitable for recent video foundational models. To address these challenges, we introduce a lightweight Anonymizing Adapter Module (AAM) that removes private information from video features while retaining general task utility. AAM can be applied in a plug-and-play fashion to frozen video encoders, minimizing the computational burden of finetuning and re-extracting features. Our framework employs three newly designed training objectives: (1) a clip-level self-supervised privacy objective to reduce mutual information between static clips, (2) a co-training objective to retain utility across seen tasks, and (3) a latent consistency loss for generalization on unseen tasks. Our extensive evaluations demonstrate a significant 35% reduction in privacy leakage while maintaining near-baseline utility performance across various downstream tasks: Action Recognition (Kinetics400, UCF101, HMDB51), Temporal Action Detection (THUMOS14), and Anomaly Detection (UCF-Crime). We also provide an analysis on anonymization for sensitive temporal attribute recognition. Additionally, we propose new protocols for assessing gender bias in action recognition models, showing that our method effectively mitigates such biases and promotes more equitable video understanding.


The repository trains a lightweight **Anonymizing Adapter Module (AAM)** on **frozen** video encoder features: clip embeddings are pushed to be less useful for private-attribute inference while staying useful for downstream tasks (action recognition, temporal action localization, weakly supervised anomaly detection). See the paper for more details.

## Usage

1. **Extract features** — Run the scripts under `feature_extraction/` (or your own pipeline) so training sees HDF5/NPY feature stores and label files expected by `feature_dl.py`.
2. **Train the AAM** — `multitask_train_fa.py` with `params/params_fa.py` (or a copy you edit).
3. **Evaluate** — Core training validates action recognition inside the multitask loop. Privacy evaluated with `privacy_eval.py`. Additional evaluations live in separate folders (`mgfn/`, `tridet/`).

## Repository layout

| Location | Purpose |
|----------|---------|
| `multitask_train_fa.py` | Main SPLAVU multitask training (AR + THUMOS TAD + UCF-Crime AD + privacy on paired frame features from the AR dataset). |
| `config.py` | Dataset roots and feature directories (edit for your machine). |
| `train_ft.py` | Train utility classifier heads on features; `--params` (default `params/params_ft.py`). |
| `privacy_eval.py` | Train privacy-attribute head on VISPR features; `--params` (default `params/params_fb.py`). |
| `pretrain_fa.py` | Optional AAM warm-start (reconstruction on features). |
| `feature_dl.py` | PyTorch datasets reading precomputed features for Kinetics, HMDB51, UCF101, VISPR, NTU, Toyota Smarthome, etc. |
| `model_loaders.py` | AAM, fine-tuned heads, TriDet and MGFN loaders used by multitask training. |
| `params/` | Parameter modules: `params_fa.py`, `params_ft.py`, `params_fb.py`. |
| `feature_extraction/` | Clip/feature extraction from video (dataloaders, `feature_extraction_action.py`, long-video / crime–THUMOS scripts). Adds repo root to `sys.path` when run as a script. |
| `tridet/` | Temporal action detection (TriDet, THUMOS14). Entry points: `tridet/train.py`, `tridet/eval.py`; config YAMLs e.g. `tridet/thumos_videoMAE.yaml`. |
| `mgfn/` | Weakly supervised anomaly detection (MGFN, UCF-Crime features). |
| `vp/` | VP-UCF101 and VP-HMDB51 training variants (`train_vp_*.py`) with `vp/params_vp.py`. |
| `bias/` | Perceived gender bias analysis (`evaluate_bias.py`, `params_bias.py`). |
| `pahmdb/` | PA-HMDB sensitive-attribute experiments (`dl_pahmdb.py`, `eval_pahmdb.py`, `params_pahmdb.py`). |
| `run_tridet.sh`, `run_mgfn.sh` | Example commands to train/eval TriDet and train MGFN baselines. |

## Setup

#### Environment creation:
```bash
conda create -n pbp -y python=3.10
conda activate pbp
pip install -r requirements.txt
```

If running feature extraction on UCF-Crime or THUMOS:
```bash
pip install nvidia-dali-cuda120
```
(or similar nvidia-dali install)

## Training

**Multitask SPLAVU:**

```bash
python multitask_train_fa.py --params params/params_fa.py
```

**Privacy evaluation (`privacy_eval.py`):**

```bash
python privacy_eval.py --params params/params_fb.py
```

**TriDet (THUMOS) and MGFN (UCF-Crime) baselines** — adjust YAML/paths and checkpoints as needed:

```bash
bash run_tridet.sh
bash run_mgfn.sh
```

## Third-party code: TriDet and MGFN

The `tridet/` and `mgfn/` trees contain adapted code for temporal action detection and weakly supervised anomaly detection.

| Method | Reference | Original code |
|--------|-----------|----------------|
| **TriDet** | *TriDet: Temporal Action Detection with Relative Boundary Modeling* (CVPR 2023) | [https://github.com/dingfengshi/TriDet](https://github.com/dingfengshi/TriDet) |
| **MGFN** | *MGFN: Magnitude-Contrastive Glance-and-Focus Network for Weakly-Supervised Video Anomaly Detection* (AAAI 2023) | [https://github.com/carolchenyx/MGFN](https://github.com/carolchenyx/MGFN) |

Cite those papers if you use TriDet/MGFN components or their training recipes.

## Citation

If you find our work useful for your research, please consider citing our paper using the following BibTeX:
```bibtex
@inproceedings{fioresi2026privacy,
  title     = {Privacy Beyond Pixels: Latent Anonymization for Privacy-Preserving Video Understanding},
  author    = {Fioresi, Joseph and Dave, Ishan Rajendrakumar and Shah, Mubarak},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026}
}
```
