# PANORAMA Pancreatic Cancer Segmentation Pipeline

A reproducible preprocessing and dataset preparation pipeline for pancreatic CT image segmentation using the **PANORAMA** dataset.

This repository contains the complete data preparation workflow developed as part of an undergraduate Computer Science thesis. It includes dataset organization, quality verification, preprocessing, visualization, anomaly investigation, metadata management, and preparation of a standardized dataset for deep learning segmentation research.

## Overview

Medical image segmentation is an important task in computer-aided diagnosis. Before a deep learning model can be trained, CT scans and segmentation masks must be standardized into a consistent representation.

The objective of this project is to transform heterogeneous abdominal CT scans from the PANORAMA dataset into a clean, standardized dataset suitable for deep learning research while making each preprocessing step transparent, reproducible, and easy to validate.

The pipeline includes:

- Dataset organization and inventory
- CT and segmentation loading
- Readability and geometry validation
- Hounsfield Unit clipping and intensity normalization
- Pancreas ROI extraction and padding
- Metadata generation and quality control
- Visualization and anomaly investigation
- Dataset eligibility and full-dataset verification

## Thesis Objective

The long-term goal of this thesis is to investigate automatic pancreatic cancer segmentation using deep learning. This repository focuses on the data preparation stage and the construction of a high-quality standardized dataset before model development.

Future work will extend the repository with dataset splitting, model training, evaluation, performance comparison, thesis experiments, and model inference.

## Dataset

This project uses the **PANORAMA** pancreatic CT dataset, which contains abdominal CT volumes, pancreas and tumor segmentation masks, clinical metadata, and multiple annotation sources.

The local dataset is organized as:

```text
data/
├── raw_ct/                         # Local raw CT volumes; not committed
├── labels/                         # Local segmentation labels; not committed
│   ├── Manual_Labels/
│   └── Automatic_Labels/
└── processed/
    ├── images/                     # Local processed arrays; not committed
    ├── masks/                      # Local processed masks; not committed
    ├── metadata.csv
    ├── batch2_eligibility.csv
    ├── batch3_inventory.csv
    └── batch3_eligibility.csv
```

The PANORAMA dataset and generated medical-image files are not distributed with this repository because of dataset licensing and storage constraints. Raw CT volumes, segmentation masks, and processed NumPy datasets are intentionally excluded from Git.

## Preprocessing Pipeline

Each CT volume passes through the following stages:

1. Load the CT scan and segmentation mask
2. Verify image and segmentation integrity
3. Verify CT and segmentation spatial geometry
4. Resample to a uniform voxel spacing
5. Clip Hounsfield Units to the configured soft-tissue window
6. Apply min-max intensity normalization
7. Compute the pancreas region of interest
8. Crop around the region of interest
9. Pad to a fixed output size
10. Convert CT and mask arrays to standardized data types
11. Save processed arrays as NumPy files
12. Generate and update processed dataset metadata
13. Run dataset-wide quality verification

The production preprocessing pipeline is implemented in `src/preprocessing.py`.

The standardized processed representation is:

```text
Image: shape (128, 160, 192), dtype float32, range [0, 1]
Mask:  shape (128, 160, 192), dtype uint8
```

## Dataset Quality Control

The validation workflow checks CT and segmentation readability, shape compatibility, voxel spacing, origin, direction, label availability, duplicate study IDs, processed file existence, output shape and dtype, intensity range, and segmentation label validity.

Small floating-point differences in voxel-spacing metadata are handled with a defined numerical tolerance. Cases with substantial spatial inconsistencies are investigated separately rather than silently processed.

## Anomaly Investigation: `100936_00001`

During Batch 2 inspection, case `100936_00001` was identified with a substantial Z-spacing discrepancy between the original CT and manual segmentation. The original CT reported approximately `0.754321 mm` Z spacing, while the manual label reported `0.5 mm`.

The case was investigated using geometry comparison, physical-coordinate analysis, voxel-index comparison, visualization, and resampling experiments. The investigation indicated that the segmentation voxel data was anatomically aligned with the original CT voxel indices, while the label Z-spacing metadata was inconsistent.

A separate repaired label was created by preserving the original segmentation voxel values and shape, changing only the spatial metadata to match the CT geometry, and converting the representation to `uint8`. The original PANORAMA label was not modified. The repaired case was processed through the standard pipeline and integrated into the processed dataset.

This investigation is documented in `notebooks/100936_00001_inspection.ipynb`. Local investigation artifacts are excluded from Git.

## Workflow Diagram

```text
PANORAMA dataset
        |
        v
Dataset inventory
        |
        v
Readability and geometry validation
        |
        v
Eligibility validation
        |
        v
Preprocessing and ROI extraction
        |
        v
Processed NumPy arrays
        |
        v
Metadata generation
        |
        v
Dataset-wide verification
```

## Current Dataset Status

The current processed dataset contains **1,703 verified cases**:

| Source | Cases |
| --- | ---: |
| Batch 1 | 1,123 |
| Batch 2 | 565 |
| Batch 2 repaired anomaly (`100936_00001`) | 1 |
| Batch 3 | 580 |
| **Total** | **1,703** |

The final verification reported zero missing images, missing masks, invalid shapes, invalid dtypes, invalid image ranges, invalid mask labels, and duplicate study IDs.

## Batch Workflows

### Batch 2

Batch 2 was inventoried and validated before preprocessing. The workflow checked CT and label availability, readability, spatial compatibility, and cases requiring review. The `100936_00001` anomaly was investigated and repaired separately.

Relevant notebooks:

- `notebooks/08_batch2_inventory.ipynb`
- `notebooks/09_batch2_preprocessing.ipynb`
- `notebooks/100936_00001_inspection.ipynb`

### Batch 3

The Batch 3 inventory contained 580 CT cases, 447 automatic labels, and 133 manual labels. All 580 cases were readable, geometrically compatible, and eligible for preprocessing. Batch 3 preprocessing increased the processed dataset from 1,123 to 1,703 cases.

Relevant notebooks:

- `notebooks/10_batch3_inventory.ipynb`
- `notebooks/11_batch3_eligibility.ipynb`
- `notebooks/12_batch3_preprocessing.ipynb`

## Repository Structure

```text
Pancreatic_Cancer_Thesis/
├── data/
│   ├── raw_ct/                    # Local dataset, not committed
│   ├── labels/                    # Local labels, not committed
│   └── processed/
│       ├── images/                # Local processed dataset, not committed
│       ├── masks/                 # Local processed masks, not committed
│       ├── metadata.csv
│       ├── batch2_eligibility.csv
│       ├── batch3_inventory.csv
│       └── batch3_eligibility.csv
├── figures/                       # Figures and visual outputs
├── models/                        # Local model artifacts, not committed
├── notebooks/                     # Research workflow notebooks
├── outputs/                       # Summary outputs
├── reports/                       # Generated reports, not committed
├── src/                           # Core Python package
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Source Modules

| Module | Description |
| --- | --- |
| `config.py` | Project configuration and paths |
| `io.py` | Dataset loading utilities |
| `preprocessing.py` | Image preprocessing pipeline |
| `validation.py` | Dataset verification and quality checks |
| `visualization.py` | CT and segmentation visualization |
| `utils.py` | Helper utilities |

## Jupyter Notebooks

| Notebook | Purpose |
| --- | --- |
| `01_dataset_inventory.ipynb` | Dataset inventory |
| `02_batch1_metadata.ipynb` | Metadata analysis |
| `03_ct_segmentation_visualization.ipynb` | CT and segmentation visualization |
| `04_dataset_validation.ipynb` | Dataset validation |
| `05_roi_size_analysis.ipynb` | ROI size analysis |
| `06_preprocessing_pipeline.ipynb` | Preprocessing demonstration |
| `07_dataset_creation.ipynb` | Processed dataset creation |
| `08_batch2_inventory.ipynb` | Batch 2 inventory and inspection |
| `09_batch2_preprocessing.ipynb` | Batch 2 preprocessing |
| `100936_00001_inspection.ipynb` | Batch 2 spatial anomaly investigation |
| `10_batch3_inventory.ipynb` | Batch 3 inventory and inspection |
| `11_batch3_eligibility.ipynb` | Batch 3 preprocessing eligibility |
| `12_batch3_preprocessing.ipynb` | Batch 3 preprocessing |

## Installation

```bash
git clone https://github.com/imacreepnweirdo/pancreatic-ai-framework.git
cd pancreatic-ai-framework
python -m venv .venv
```

Activate the environment:

```bash
# Windows
.venv\Scripts\activate

# Linux or macOS
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Requirements

The project uses Python 3.11 or newer and the packages listed in `requirements.txt`, including NumPy, Pandas, SimpleITK, NiBabel, Matplotlib, tqdm, and OpenPyXL.

## Reproducing Dataset Preparation

1. Obtain the PANORAMA dataset through its official distribution channel.
2. Place raw CT volumes and labels in the local `data/` directories.
3. Run the relevant inventory notebook.
4. Review readability and spatial-geometry validation.
5. Run the corresponding eligibility notebook.
6. Process eligible cases with the preprocessing notebook.
7. Verify the generated arrays and metadata.
8. Run full processed-dataset verification.

The notebooks provide the research record for the preparation workflow. Generated medical-image files remain local and are not committed.

## Current Progress

- [x] Dataset organization
- [x] Metadata verification
- [x] CT and segmentation visualization
- [x] CT/segmentation geometry validation
- [x] Eligibility validation
- [x] ROI extraction
- [x] Intensity normalization
- [x] Metadata generation
- [x] Batch 1 preprocessing
- [x] Batch 2 preprocessing and anomaly investigation
- [x] Batch 3 inventory, eligibility validation, and preprocessing
- [x] Final 1,703-case dataset verification
- [ ] Final dataset splitting
- [ ] Model training
- [ ] Model evaluation
- [ ] Thesis experiments

## Roadmap and Future Work

- Add reproducible train/validation/test splitting
- Implement and evaluate a 3D U-Net baseline
- Benchmark nnU-Net and transformer-based segmentation models
- Add cross-validation experiments
- Compare Dice score, IoU, precision, recall, and Hausdorff distance
- Add an automatic inference pipeline
- Add automated unit tests and a command-line workflow

## Results

Quantitative model results will be added after the training and evaluation experiments are completed.

## Acknowledgements

This work is part of an undergraduate thesis and builds on the PANORAMA dataset and open medical imaging research practices.

## Citation

If this repository contributes to your research, please cite the PANORAMA dataset, this repository, and the accompanying undergraduate thesis when it is published. The formal citation details will be added here.

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Author

**Shahriar Ahmed**

Undergraduate Student

Department of Computer Science and Engineering

Rajshahi, Bangladesh
