# PANORAMA Pancreatic Cancer Segmentation Pipeline

A reproducible preprocessing and dataset preparation pipeline for pancreatic CT image segmentation using the **PANORAMA** dataset.

This repository contains the complete data preparation workflow developed as part of my undergraduate Computer Science thesis. It includes dataset organization, quality verification, preprocessing, visualization, and metadata management required before training deep learning segmentation models.

---

# Overview

Medical image segmentation is one of the most important tasks in computer-aided diagnosis. Before any deep learning model can be trained, CT scans must be standardized into a consistent format.

The objective of this project is to build a reproducible preprocessing pipeline that transforms heterogeneous abdominal CT scans from the PANORAMA dataset into a clean, standardized dataset suitable for deep learning research.

The pipeline performs:

- Dataset organization
- CT and segmentation loading
- Dataset verification
- Image preprocessing
- ROI extraction
- Metadata generation
- Visualization
- Quality control

The repository is designed to make every preprocessing step transparent, reproducible, and easy to validate.

---

# Thesis Objective

The long-term goal of this thesis is to investigate automatic pancreatic cancer segmentation using deep learning.

This repository focuses on the **data preparation stage**, which includes constructing a high-quality training dataset before model development.

Future work will extend this repository with:

- Dataset splitting
- Model training
- Evaluation
- Performance comparison
- Thesis experiments

---

# Dataset

This project uses the **PANORAMA** pancreatic CT dataset.

The dataset contains:

- Abdominal CT volumes (.nii.gz)
- Pancreas and tumor segmentation masks
- Clinical metadata
- Multiple annotation sources

This repository separates the dataset into:

```
data/
│
├── raw_ct/
├── labels/
│   ├── Manual_Labels/
│   └── Automatic_Labels/
│
└── processed/
    ├── images/
    ├── masks/
    └── metadata.csv
```

> **Note:** The PANORAMA dataset is not distributed with this repository due to licensing and storage limitations.

---

# Preprocessing Pipeline

Each CT volume passes through the following preprocessing stages:

1. Load CT scan and segmentation mask
2. Verify image integrity
3. Resample to a uniform voxel spacing
4. Clip Hounsfield Units to a pancreas soft-tissue window
5. Min-max intensity normalization
6. Compute pancreas bounding box
7. Crop around the region of interest
8. Pad volumes to a fixed input size
9. Save processed CT and mask as NumPy arrays
10. Generate processed dataset metadata

The final processed dataset has a fixed shape suitable for deep learning models.

---

# Repository Structure

```
Pancreatic_Cancer_Thesis/
│
├── data/
│   ├── processed/
│   ├── raw_ct/
│   └── labels/
│
├── figures/
├── models/
├── notebooks/
├── outputs/
├── reports/
├── src/
│
├── README.md
├── requirements.txt
└── LICENSE
```

---

# Source Modules

The `src` package contains the core preprocessing pipeline.

| Module | Description |
|---------|-------------|
| `config.py` | Project configuration and paths |
| `io.py` | Dataset loading utilities |
| `preprocessing.py` | Image preprocessing pipeline |
| `validation.py` | Dataset verification and quality checks |
| `visualization.py` | CT and segmentation visualization |
| `utils.py` | Helper utilities |

---

# Jupyter Notebooks

The notebooks document every stage of dataset preparation.

| Notebook | Purpose |
|-----------|---------|
| 01 | Dataset inventory |
| 02 | Metadata analysis |
| 03 | CT and segmentation visualization |
| 04 | Dataset validation |
| 05 | ROI size analysis |
| 06 | Preprocessing pipeline |
| 07 | Processed dataset creation |

Together, these notebooks provide a complete record of the preprocessing workflow.

---

# Installation

Clone the repository

```bash
git clone https://github.com/your-username/Pancreatic_Cancer_Thesis.git
cd Pancreatic_Cancer_Thesis
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate it

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Requirements

Main dependencies include:

- Python 3.11+
- NumPy
- Pandas
- SimpleITK
- NiBabel
- Matplotlib
- tqdm
- OpenPyXL

See `requirements.txt` for the complete list.

---

# Current Progress

✔ Dataset organization

✔ Metadata verification

✔ CT visualization tools

✔ Segmentation visualization

✔ Automatic preprocessing pipeline

✔ Dataset validation

✔ ROI extraction

✔ Metadata generation

✔ Processed dataset creation

⬜ Dataset splitting

⬜ Model training

⬜ Model evaluation

⬜ Thesis experiments

---

# Future Work

Planned extensions include:

- 3D U-Net implementation
- nnU-Net benchmarking
- Transformer-based segmentation models
- Cross-validation experiments
- Performance comparison using Dice Score, IoU, Precision, Recall, and Hausdorff Distance
- Automatic inference pipeline

---

# Citation

If this repository contributes to your research, please cite:

- The PANORAMA dataset
- This repository
- The accompanying undergraduate thesis (to be published)

---

# License

This project is licensed under the MIT License.

See the `LICENSE` file for details.

---

# Author

**Shahriar Ahmed**

Undergraduate Student

Department of Computer Science and Engineering

Rajshahi, Bangladesh

Undergraduate Thesis Project