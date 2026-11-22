<div align="center">

# Automated 3D MRI Quality Control
### A Leak-Free, Shortcut-Audited 3D Deep Learning Benchmark

**Hybrid 3D ResNet · Multi-Head Global-to-Local Cross-Attention · 3D Grad-CAM**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Platform](https://img.shields.io/badge/Platform-Kaggle-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/)
[![GPU](https://img.shields.io/badge/GPU-2×%20NVIDIA%20T4-76B900?logo=nvidia&logoColor=white)](https://www.nvidia.com/)
[![Task](https://img.shields.io/badge/Task-3D%20MRI%20Quality%20Control-6C5CE7)]()

**A reproducible research pipeline for evaluating whether 3D models can distinguish clean-like and synthetically corrupted brain MRI volumes without relying on subject overlap or trivial background cues.**

</div>

---

## Overview

This project implements an end-to-end 3D brain MRI quality-control (QC) research pipeline using BraTS2021 FLAIR volumes. It creates paired clean-like and artifact-corrupted examples, enforces subject-wise data partitioning, audits simple statistical shortcuts, benchmarks dual-CNN baselines against attention-based fusion models, and evaluates robustness under multiple artifact severities.

The proposed model combines two 3D feature pathways with residual refinement and multi-head cross-attention:

- **Local pathway:** extracts features from multiple small 3D brain regions.
- **Global pathway:** encodes a larger central 3D region.
- **Fusion:** uses global-to-local multi-head cross-attention and a learned local-region gate.
- **Interpretability:** produces 3D Grad-CAM visualizations and regional attention distributions.

> **Important scope note:** The notebook uses BraTS2021 **FLAIR MRI** scans. Despite the project’s historical “DTI” naming in some variables and output filenames, this implementation does not process diffusion-tensor imaging (DTI) measurements or diffusion-derived maps. “MRI QC” is the technically accurate modality description for this notebook.

## Research Highlights

| Component | Implementation |
|---|---|
| Data source | BraTS2021 FLAIR NIfTI scans from the referenced Kaggle notebook |
| Cohort used | 150 unique subjects, generating 300 paired volumes |
| Labels | `1 = Good` (clean-like, baseline-noise volume); `0 = Poor` (synthetically corrupted volume) |
| Data partition | 60% train / 20% validation / 20% test, grouped by subject |
| Shortcut audit | Logistic Regression using background and global-intensity summary statistics |
| Baselines | `DTI_CNN1`, `DTI_CNN2`, and two voting configurations |
| Attention model A | Warm-started dual-CNN with self-attention |
| Proposed model B | Hybrid 3D ResNet with multi-head global-to-local cross-attention |
| Cross-validation | Five-fold subject-wise `GroupKFold`, with fold-specific pretraining |
| Explainability | 3D Grad-CAM on `encoder_global.layer3` and regional attention visualization |
| Robustness | Clean, mild, moderate, and severe test-time artifact conditions |

## Benchmark Results

### Hold-out test and cross-validation

| Method | Evaluation | Accuracy | F1-score | ROC-AUC |
|---|---|---:|---:|---:|
| Linear shortcut audit | Subject-wise validation | 55.00% | — | 0.6233 |
| Dual-CNN Voting 1 | Hold-out test | 88.33% | 0.8889 | 0.9200 |
| Dual-CNN Voting 2 | Hold-out test | 86.67% | 0.8667 | — |
| Dual-CNN + Self-Attention (Variant A) | Hold-out test | 88.33% | 0.8889 | 0.9167 |
| **3D ResNet + Cross-Attention (Variant B)** | **Hold-out test** | **90.00%** | **0.9032** | **0.9078** |
| **Variant B — subject-wise 5-fold CV** | **5-fold GroupKFold** | **84.67% ± 5.52%** | **0.8547** | **0.8996** |

### Multi-severity stress test — Proposed Variant B

| Test condition | Number of volumes | Accuracy |
|---|---:|---:|
| Clean-like | 30 | 93.33% |
| Mild artifact | 30 | 76.67% |
| Moderate artifact | 30 | 100.00% |
| Severe artifact | 30 | 86.67% |

| Overall stress-test metric | Result |
|---|---:|
| Balanced accuracy | **90.56%** |
| Raw accuracy | 89.17% |
| F1-score | 0.8116 |
| ROC-AUC | 0.9496 |
| 95% bootstrap CI for balanced accuracy | [84.13%, 95.73%] |

The stress-test confidence interval is estimated using 1,000 bootstrap resamples. Small differences in the displayed balanced-accuracy value may occur from rounding.

### How to interpret these numbers

The hold-out test result is based on 60 volumes from 30 held-out subjects. The five-fold cross-validation result is a separate subject-wise evaluation and should be reported alongside—not substituted for—the single hold-out result. The stress test uses paired clean-like and synthetically corrupted examples derived from the held-out clean-like scans; it is a controlled robustness experiment, not an independent clinical cohort.

The proposed model has the highest reported hold-out accuracy and F1-score in this experiment. Its ROC-AUC (0.9078), however, is slightly lower than Voting 1 (0.9200) and Variant A (0.9167). Therefore, the results do not establish that Variant B is best on every metric.

---

## Methodology

### 1. Dataset construction

The notebook discovers BraTS2021 FLAIR NIfTI files under the `post_process_data` directory, deduplicates scans by subject ID, and selects up to 150 subjects.

For each selected subject, it creates two volumes:

- **Good (`label = 1`):** the normalized source volume with mild baseline Rician noise applied inside the brain mask.
- **Poor (`label = 0`):** a copy of the same source volume with matched baseline Rician noise plus one synthetic artifact condition.

Synthetic artifact types:

- Motion blur
- Nyquist ghosting
- Additional Rician noise
- Slice dropout
- Combined motion and ghosting

The brain mask is defined as `volume > 0`. Artifact operations are restricted to the brain region, and background voxels are reset to exactly `0.0`.

### 2. Subject-wise data split

The 150 subjects are split into:

| Partition | Subjects | Volumes |
|---|---:|---:|
| Train | 90 | 180 |
| Validation | 30 | 60 |
| Test | 30 | 60 |
| **Total** | **150** | **300** |

`GroupShuffleSplit` uses `subject_id` as the grouping variable. Assertions verify that the subject-ID intersections between train, validation, and test are all empty.

### 3. Shortcut audit

Before training the deep models, a Logistic Regression baseline uses only five summary features:

- `zero_frac`
- `fg_mean`
- `fg_std`
- `fg_p50`
- `fg_p95`

The recorded validation result is **55.00% accuracy** and **0.6233 ROC-AUC**. This is a useful sanity check against the specific background/intensity shortcuts tested. It does not prove that every possible shortcut or confounder has been eliminated.

### 4. Multi-scale 3D ROI extraction

The `DTIRoiDataset` creates multi-scale inputs:

- Ten local ROIs of size `45 × 45 × 45`
- One central ROI of size `90 × 90 × 45`

Training uses random local foreground ROIs and synchronized 3D spatial flips. Validation and test use deterministic, stratified anchor locations for reproducible evaluation.

### 5. Model benchmarking

**Baseline — dual CNNs**

`DTI_CNN1` processes local patches, while `DTI_CNN2` processes the central ROI. Two voting configurations provide reference baselines.

**Variant A — self-attention fusion**

The dual-CNN encoders are warm-started from weights trained on the training partition. A self-attention fusion classifier combines local and central representations.

**Variant B — hybrid 3D ResNet + cross-attention**

The proposed architecture adds gated `BasicBlock3D` residual refinement to the two pretrained 3D encoders. A four-head `nn.MultiheadAttention` layer uses the global feature as the query and local ROI features as keys and values. A learned local gate also weights local regions. The fused representation is classified by a fully connected head.

### 6. Checkpoint selection and test protocol

For the hold-out experiments, the best checkpoints are selected using validation data. The test loader is evaluated after training and checkpoint selection; it is not used to choose epochs or checkpoints.

### 7. Five-fold subject-wise cross-validation

The notebook uses `GroupKFold(n_splits=5)` with `subject_id` as the group. In each fold:

1. Dual-CNN backbones are initialized and pretrained from scratch using only that fold’s training subjects.
2. The cross-attention model is trained using that fold’s training subjects.
3. The held-out fold is evaluated once.

Fold-specific pretraining avoids carrying globally pretrained weights across fold boundaries.

### 8. Explainability

The notebook unwraps `nn.DataParallel` before attaching hooks to `encoder_global.layer3`. It computes a 3D Grad-CAM map for the artifact/corrupted class and renders axial, coronal, and sagittal views. It also plots the model’s allocation of attention across the ten local ROIs.

### 9. Stress testing

The stress test starts from clean-like held-out volumes and creates mild, moderate, and severe artifact variants. It evaluates the model across clean and corrupted conditions and reports accuracy by severity, balanced accuracy, F1, ROC-AUC, and a 95% bootstrap confidence interval.

---

## Pipeline at a Glance

```text
BraTS2021 FLAIR NIfTI volumes
              │
              ▼
       Subject discovery
              │
              ▼
     Paired volume generation
       ┌──────┴──────┐
       │             │
   Good / clean   Poor / synthetic
       │             │
       └──────┬──────┘
              ▼
    Subject-wise 60/20/20 split
              │
              ▼
       Shortcut audit
              │
              ▼
      Multi-scale 3D ROIs
              │
      ┌───────┼─────────────────────┐
      ▼       ▼                     ▼
  Dual-CNN  Self-attention     3D ResNet +
  voting    fusion (A)         cross-attention (B)
      │       │                     │
      └───────┴──────────┬──────────┘
                        ▼
              Validation-selected models
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
       Hold-out evaluation   5-fold GroupKFold
              │
       ┌──────┴───────────────┐
       ▼                      ▼
   3D Grad-CAM          Severity stress test
       │                      │
       └──────────┬───────────┘
                  ▼
        Metrics, plots and archive
```

---

## Getting Started

### Recommended environment

The notebook is designed for Kaggle Notebooks and was executed with:

- 2 × NVIDIA Tesla T4 GPUs (14.6 GB VRAM each)
- CUDA-enabled PyTorch
- Mixed-precision training (AMP)
- `nn.DataParallel` when multiple CUDA devices are available

A single-GPU or CPU environment may require smaller batch sizes and longer execution time. The notebook’s two-T4 configuration is the tested target environment; other hardware configurations are not guaranteed to have the same runtime or memory behavior.

### Dataset source

The notebook references the following Kaggle notebook as its BraTS2021 data source:

[**BraTS2021 U-Net Segmentation Model (300–1251 images)**](https://www.kaggle.com/code/victorfernandezalbor/brats2021-u-net-segmentation-model-300-1251-img)

Add the source notebook/dataset as an input to your Kaggle Notebook and ensure the expected `post_process_data` directory containing FLAIR NIfTI files is available.

### Run the notebook

1. Open the project notebook in Kaggle.
2. Attach the referenced BraTS2021 data source.
3. Select a GPU accelerator. For the intended configuration, enable **2 × NVIDIA T4 GPUs** where available.
4. Enable Internet only if required by your Kaggle environment or dependency setup.
5. Run the notebook from top to bottom so that dataset creation, split generation, training, evaluation, and artifact export occur in order.
6. Download the generated output archive from `/kaggle/working/DTI_QC_LeakFree_2xT4_Outputs.zip`.

### Main Python dependencies

The notebook imports the following packages:

```bash
pip install numpy pandas scipy scikit-learn nibabel matplotlib seaborn
```

Install a CUDA-compatible PyTorch build for the selected runtime by following the [official PyTorch installation guide](https://pytorch.org/get-started/locally/). Kaggle environments may already provide PyTorch and the scientific Python packages.

---

## Generated Outputs

The notebook writes its artifacts to `/kaggle/working/`.

| Artifact | Description |
|---|---|
| `dti_qc_leakfree_dashboard.png` | Four-panel benchmark and training dashboard |
| `orthogonal_gradcam_montage.png` | Axial, coronal, and sagittal 3D Grad-CAM montage |
| `attention_distribution.png` | Attention allocation over ten local ROIs |
| `roc_curve_leakfree.png` | Hold-out ROC comparison |
| `final_leakfree_benchmark.csv` | Consolidated benchmark metrics |
| `train_metadata.csv` | Training volume metadata |
| `val_metadata.csv` | Validation volume metadata |
| `test_metadata.csv` | Test volume metadata |
| `best_self_attention_val_selected.pth` | Validation-selected Variant A checkpoint |
| `best_cross_attention_val_selected.pth` | Validation-selected Variant B checkpoint |
| `DTI_QC_LeakFree_2xT4_Outputs.zip` | Packaged output archive |

The archive is created by the final notebook step and contains the listed result files.

---

## Reproducibility and Leakage Safeguards

This implementation includes several safeguards:

- **Subject grouping:** paired volumes from the same subject remain in one partition.
- **Split assertions:** train/validation/test subject overlap is explicitly checked.
- **Brain-masked corruption:** synthetic artifacts are applied within the brain mask; background is reset to zero.
- **Matched baseline noise:** both classes receive baseline Rician noise to reduce noise-floor shortcuts.
- **Statistical shortcut audit:** a simple model checks whether selected global/background statistics trivially predict labels.
- **Validation-only checkpoint selection:** hold-out test results do not select checkpoints.
- **Fold-local pretraining:** each cross-validation fold trains its own CNN backbones from scratch.
- **Deterministic evaluation ROIs:** validation and test ROI anchors are fixed.

These safeguards reduce specific leakage and shortcut risks, but they do not establish the absence of all possible sources of bias. The code, data provenance, preprocessing, and evaluation design should still be independently reviewed before publication.

---

## Limitations and Responsible Interpretation

This project is a **research prototype**, not a clinically validated medical device or a replacement for radiologist review.

1. **Synthetic quality labels:** Good/Poor labels are generated by the notebook’s artifact simulation process. They are not expert-provided clinical quality annotations.
2. **Limited cohort:** the reported experiment uses 150 subjects and 300 derived volumes. The paired examples are not 300 independent subjects.
3. **Single source and modality:** the implementation uses BraTS2021 FLAIR data. Generalization to other MRI sequences, scanners, institutions, acquisition protocols, or DTI data has not been demonstrated.
4. **Artifact realism:** simulated motion, ghosting, noise, and slice dropout are controlled approximations. Their clinical realism and coverage need expert review.
5. **Metric variation:** the best hold-out accuracy does not imply the best ROC-AUC or performance under every severity.
6. **External validation:** independent datasets and expert-labelled real-world scans are required to assess clinical generalization.
7. **Interpretability limits:** Grad-CAM and attention maps are diagnostic visualizations, not proof that the model uses clinically meaningful evidence or that its predictions are causally grounded.

### Recommended next research steps

- Validate on an independent, institutionally distinct MRI cohort.
- Obtain expert quality ratings and evaluate agreement with human reviewers.
- Test across additional MRI sequences and acquisition protocols.
- Compare against 3D medical-imaging architectures and calibrated classical baselines.
- Report subject-level confidence intervals and per-artifact sensitivity/specificity.
- Perform ablations for ROI count, ROI size, cross-attention, local gating, and residual refinement.
- Assess calibration, threshold sensitivity, and subgroup/site robustness.
- Review synthetic artifact parameters with MRI physicists and radiologists.

---

## Citation and Data Attribution

The dataset source referenced by the notebook is:

Fernandezalbor, V. **BraTS2021 U-Net Segmentation Model (300–1251 images)**. Kaggle Notebook.  
https://www.kaggle.com/code/victorfernandezalbor/brats2021-u-net-segmentation-model-300-1251-img

Please review the source dataset’s license, terms of use, and any applicable BraTS data-use requirements before redistribution or publication. Cite the original BraTS challenge/data paper and the dataset source in any academic manuscript. The Kaggle notebook link is the source reference used by this implementation; it is not a substitute for citing the original dataset publication.

## Project Structure

The primary deliverable supplied for this project is a Kaggle notebook. After execution, the notebook generates the following working-directory structure:

```text
/kaggle/working/
├── dti_quality_dataset_clean/
├── train_metadata.csv
├── val_metadata.csv
├── test_metadata.csv
├── best_self_attention_val_selected.pth
├── best_cross_attention_val_selected.pth
├── dti_qc_leakfree_dashboard.png
├── orthogonal_gradcam_montage.png
├── attention_distribution.png
├── roc_curve_leakfree.png
├── final_leakfree_benchmark.csv
└── DTI_QC_LeakFree_2xT4_Outputs.zip
```

---

<div align="center">

**Built for reproducible experimentation in 3D brain MRI quality control.**

*Research prototype · Synthetic artifact benchmark · Not for clinical diagnosis*

</div>

