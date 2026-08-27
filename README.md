# Pancreatic Cancer AI Framework

A research framework for **AI-assisted pancreatic cancer analysis from abdominal CT scans**, developed as part of an undergraduate thesis project.

The project focuses on building a reproducible data-processing and machine-learning pipeline for pancreatic ductal adenocarcinoma (PDAC) analysis using the **PANORAMA public training and development dataset**.

---

## Project Status

The dataset preparation and preprocessing stage is currently complete.

| Item | Current status |
|---|---:|
| PANORAMA cases processed | **2,238** |
| PDAC cases | **676** |
| Non-PDAC cases | **1,562** |
| Processing dimensionality | **3D** |
| Processed image shape | **128 × 160 × 192** |
| Processed mask shape | **128 × 160 × 192** |
| Image dtype | `float32` |
| Image range | `[0, 1]` |
| Mask dtype | `uint8` |
| Mask labels | `0–6` |
| Missing processed images | **0** |
| Missing processed masks | **0** |
| Invalid image shapes | **0** |
| Invalid mask shapes | **0** |
| Invalid image dtypes | **0** |
| Invalid mask dtypes | **0** |
| Invalid image ranges | **0** |
| Invalid mask labels | **0** |
| Duplicate study IDs | **0** |

The project is now moving from **dataset engineering to experimental model development**.

---

## Research Direction

The long-term objective is to investigate whether a 3D deep-learning system can provide useful and trustworthy assistance for pancreatic cancer analysis from CT imaging.

The current research direction includes:

1. Reproducible 3D CT preprocessing
2. Patient/study-level dataset partitioning
3. 3D deep-learning model development
4. PDAC classification and/or segmentation experiments
5. Evaluation using clinically relevant metrics
6. Comparison of different modelling strategies
7. Investigation of model interpretability and reliability
8. Analysis of failure cases and limitations

The exact experimental architecture and final research questions are still being developed and will be documented as experiments are completed.

---

# Dataset

## PANORAMA

The primary dataset used in this project is the **PANORAMA public training and development dataset**.

PANORAMA combines data from multiple sources, including patients without PDAC from the NIH Clinical Center dataset and patients from the Medical Segmentation Decathlon pancreas dataset. The official PANORAMA annotation repository documents the reference standards, annotation process, and label definitions.

Official resources:

- [PANORAMA Challenge](https://panorama.grand-challenge.org/)
- [PANORAMA Labels Repository](https://github.com/DIAGNijmegen/panorama_labels)
- [PANORAMA Study Protocol](https://doi.org/10.5281/zenodo.10599559)

### Dataset composition used in this project

The current processed dataset contains:

- **676 PDAC cases**
- **1,562 non-PDAC cases**
- **2,238 cases total**

The dataset includes both manually and automatically generated lesion annotations. According to the official PANORAMA documentation, 482 of the 676 PDAC cases have manual lesion annotations, while the remaining PDAC cases have automatically generated lesion segmentations.

The pancreas, pancreatic duct, veins, arteries, and common bile duct annotations are automatically generated according to the PANORAMA documentation.

---

# Official PANORAMA Label Mapping

The segmentation masks use the official PANORAMA label definitions:

| Label | Structure |
|---:|---|
| `0` | Background |
| `1` | PDAC lesion |
| `2` | Veins |
| `3` | Arteries |
| `4` | Pancreas parenchyma |
| `5` | Pancreatic duct |
| `6` | Common bile duct |

These definitions are taken directly from the official PANORAMA annotation repository.

---

# Data Processing Pipeline

The project uses a reproducible preprocessing pipeline implemented in `src/preprocessing.py`.

The general pipeline is:

```text
Raw CT
   │
   ▼
CT / Label Loading
   │
   ▼
Geometry Validation
   │
   ▼
Resampling
   │
   ▼
HU Windowing
   │
   ▼
Normalization
   │
   ▼
ROI Cropping
   │
   ▼
Padding
   │
   ▼
3D Processed Volume
   │
   ├── Image
   └── Segmentation Mask
```

### Current preprocessing configuration

```
Target spacing : (1.0, 1.0, 3.0) mm
ROI size       : (128, 160, 192)
HU window      : (-150, 250)
```

The resulting CT volumes are stored as:

```
float32
range: [0, 1]
shape: (128, 160, 192)
```

Segmentation masks are stored as:

```
uint8
shape: (128, 160, 192)
labels: 0–6
```

---

# Dataset Quality Control

Before preprocessing, each dataset batch underwent inspection and eligibility checks.

The checks included:

- CT readability
- Label readability
- CT/label shape compatibility
- CT/label spacing compatibility
- Origin compatibility
- Direction compatibility
- Label-type selection
- Previously processed-case detection
- Held-out-case detection

After preprocessing, the complete dataset was verified for:

- Missing image files
- Missing mask files
- Image shape consistency
- Mask shape consistency
- Image dtype consistency
- Mask dtype consistency
- Image value range
- Valid segmentation labels
- Duplicate study IDs

The final verification currently reports:

```
Processed cases       : 2238
Missing images        : 0
Missing masks         : 0
Invalid image shape   : 0
Invalid mask shape    : 0
Invalid image dtype   : 0
Invalid mask dtype    : 0
Invalid image range   : 0
Invalid mask labels   : 0
Duplicate study IDs   : 0
```

---

# Batch Processing

The dataset was processed incrementally to allow inspection and quality control before committing to the complete dataset.

Current batches:

```
Batch 1
Batch 2
Batch 3
Batch 4
```

Each batch was independently inspected for readability, label availability, and CT/label geometry before preprocessing.

---

# Batch 2 Anomaly Investigation

During Batch 2 processing, case:

```
100936_00001
```

was identified as having a mismatch between the original CT geometry and the corresponding manual label geometry.

The original PANORAMA data was **not modified**.

Instead, a separate repaired copy was created for preprocessing. The repaired label was explicitly injected into the preprocessing pipeline while preserving the original PANORAMA annotation.

The resulting processed case passed the complete dataset validation.

The original annotation remains preserved separately from the repaired preprocessing artifact.

---

# Repository Structure

```
Pancreatic_Cancer_Thesis/
│
├── data/
│   ├── raw_ct/
│   │   └── Raw CT data
│   │
│   ├── labels/
│   │   ├── Automatic_Labels/
│   │   └── Manual_Labels/
│   │
│   └── processed/
│       ├── images/
│       ├── masks/
│       ├── metadata.csv
│       ├── batch*_inventory.csv
│       └── batch*_eligibility.csv
│
├── figures/
├── models/
├── notebooks/
│   ├── 07_dataset_creation.ipynb
│   ├── 08_batch2_inventory.ipynb
│   ├── 09_batch2_preprocessing.ipynb
│   ├── 10_batch3_inventory.ipynb
│   ├── 11_batch3_eligibility.ipynb
│   ├── 12_batch3_preprocessing.ipynb
│   ├── 13_batch4_inventory.ipynb
│   ├── 14_batch4_eligibility.ipynb
│   ├── 15_batch4_preprocessing.ipynb
│   └── 16_final_dataset_audit.ipynb
│
├── src/
│   ├── config.py
│   ├── io.py
│   └── preprocessing.py
│
├── reports/

├── README.md
└── .gitignore
```

Large medical-imaging files and generated processed volumes are intentionally **not stored in the Git repository**.

---

# Reproducibility

The project is organized so that dataset preparation and preprocessing can be reproduced from the source notebooks and preprocessing implementation.

Important preprocessing parameters are centralized in the project configuration and should not be changed casually after the dataset split has been established.

Future experiments should use a fixed dataset partition so that different models can be compared fairly.

---

# Next Research Stage

The dataset preparation stage is complete.

The next stage is:

```
Final Dataset Audit
        │
        ▼
Patient / Study-Level Split
        │
        ▼
Train / Validation / Test Sets
        │
        ▼
3D Baseline Model
        │
        ▼
Training & Evaluation
        │
        ▼
Model Comparison
        │
        ▼
Interpretability / Explainability
        │
        ▼
Failure Analysis
        │
        ▼
Final Experimental Results
```

A critical requirement is that dataset splitting must occur at the **patient/study level**, rather than randomly splitting individual slices or derived samples. This is intended to prevent information leakage between training, validation, and test sets.

The test set will be fixed before model experimentation begins.

---

# Research Philosophy

This project is not intended to focus solely on achieving a high classification or segmentation score.

The research will also investigate:

- Generalization
- Data leakage
- Class imbalance
- Annotation quality
- Model failure cases
- Interpretability
- Robustness
- Clinical plausibility
- Reproducibility

Experimental claims will only be added to this README after they have been empirically evaluated.

---

# Data Privacy and Repository Policy

The repository contains the code, notebooks, metadata, and documentation required to reproduce the research workflow.

Large medical-imaging files are intentionally excluded from Git.

The following types of files are not committed:

```
*.nii
*.nii.gz
*.npy
raw CT volumes
processed image volumes
processed segmentation volumes
local anomaly-investigation artifacts
generated HTML reports
large archives
```

The original dataset should be obtained from its official source and handled according to the applicable dataset terms and research requirements.

---

# Citation

If using the PANORAMA dataset or its annotations, please cite the official PANORAMA publication:

> Alves, N., Schuurmans, M., Rutkowski, D., Yakar, D., Haldorsen, I., Liedenbaum, M., Molven, A., Vendittelli, P., Litjens, G., Hermans, J., & Huisman, H. (2024). *The PANORAMA Study Protocol: Pancreatic Cancer Diagnosis - Radiologists Meet AI.*

DOI:

`https://doi.org/10.5281/zenodo.10599559`

Please also consult the official PANORAMA resources for dataset-specific citation requirements.

---

# Disclaimer

This repository represents an ongoing academic research project.

The models and methods developed here are experimental and are **not intended for clinical diagnosis or treatment decisions**.

Experimental performance should not be interpreted as evidence of clinical effectiveness without appropriate external validation and clinical evaluation.
