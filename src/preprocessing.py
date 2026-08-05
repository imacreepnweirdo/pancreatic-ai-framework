"""
preprocessing.py

Image preprocessing utilities for the PANORAMA dataset.

Contents
--------
• HU clipping
• Intensity normalization
• Image resampling
• Bounding-box utilities
• Fixed-size pancreas ROI extraction

Author: Shahriar Ahmed
"""

import SimpleITK as sitk
import numpy as np
import pandas as pd
from .io import list_ct_cases, load_mask
from src.config import (
    PROCESSED_DIR,
)


# ============================================================
# Default Parameters
# ============================================================

DEFAULT_HU_WINDOW = (-150, 250)

DEFAULT_TARGET_SPACING = (1.0, 1.0, 3.0)

DEFAULT_ROI_SIZE = (128, 160, 192)

def clip_hu(
    image: np.ndarray,
    lower: float = -150,
    upper: float = 250
) -> np.ndarray:
    """
    Clip CT intensities to a pancreas soft-tissue window.

    Parameters
    ----------
    image : np.ndarray
        CT volume in Hounsfield Units.

    lower : float
        Lower clipping limit.

    upper : float
        Upper clipping limit.

    Returns
    -------
    np.ndarray
        Clipped CT volume.
    """

    if lower >= upper:
        raise ValueError("lower must be smaller than upper")

    return np.clip(image, lower, upper)


# ============================================================
# Min-Max Normalization
# ============================================================

def normalize_minmax(image):
    """
    Normalize image to the range [0, 1].

    Parameters
    ----------
    image : numpy.ndarray

    Returns
    -------
    numpy.ndarray
    """

    image = image.astype(np.float32)

    minimum = image.min()
    maximum = image.max()

    if maximum == minimum:
        return np.zeros_like(image, dtype=np.float32)

    return (image - minimum) / (maximum - minimum)


# ============================================================
# Z-score Normalization
# ============================================================

def normalize_zscore(image):
    """
    Normalize image using z-score normalization.

    Parameters
    ----------
    image : numpy.ndarray

    Returns
    -------
    numpy.ndarray
    """

    image = image.astype(np.float32)

    mean = image.mean()
    std = image.std()

    if std == 0:
        return np.zeros_like(image, dtype=np.float32)

    return (image - mean) / std


# ============================================================
# Resampling
# ============================================================

def resample_image(
    image,
    target_spacing=DEFAULT_TARGET_SPACING,
    interpolator=sitk.sitkLinear,
):
    """
    Resample a SimpleITK image to a new voxel spacing.

    Parameters
    ----------
    image : sitk.Image

    target_spacing : tuple
        Desired voxel spacing in mm.

    interpolator :
        sitk.sitkLinear for CT
        sitk.sitkNearestNeighbor for masks

    Returns
    -------
    sitk.Image
    """

    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_size = [

        int(round(
            original_size[i] *
            (original_spacing[i] / target_spacing[i])
        ))

        for i in range(3)

    ]

    resampler = sitk.ResampleImageFilter()

    resampler.SetInterpolator(interpolator)

    resampler.SetOutputSpacing(target_spacing)

    resampler.SetSize(new_size)

    resampler.SetOutputDirection(
        image.GetDirection()
    )

    resampler.SetOutputOrigin(
        image.GetOrigin()
    )

    resampler.SetDefaultPixelValue(0)

    return resampler.Execute(image)

def resample_ct(
    ct_image,
    target_spacing=DEFAULT_TARGET_SPACING,
):
    """
    Resample a CT image using linear interpolation.
    """

    return resample_image(
        ct_image,
        target_spacing,
        sitk.sitkLinear,
    )

def resample_mask(
    mask_image,
    target_spacing=DEFAULT_TARGET_SPACING,
):
    """
    Resample a segmentation mask using
    nearest-neighbor interpolation.
    """

    return resample_image(
        mask_image,
        target_spacing,
        sitk.sitkNearestNeighbor,
    )


def resample_case(
    case,
    target_spacing=DEFAULT_TARGET_SPACING,
):
    """
    Resample a complete case.

    Parameters
    ----------
    case : dict

    target_spacing : tuple

    Returns
    -------
    dict
    """

    ct = resample_ct(
        case["ct_image"],
        target_spacing,
    )

    mask = resample_mask(
        case["mask_image"],
        target_spacing,
    )

    new_case = case.copy()

    new_case["ct_image"] = ct
    new_case["mask_image"] = mask

    new_case["ct_array"] = sitk.GetArrayFromImage(ct)
    new_case["mask_array"] = sitk.GetArrayFromImage(mask)

    new_case["spacing"] = ct.GetSpacing()
    new_case["origin"] = ct.GetOrigin()
    new_case["direction"] = ct.GetDirection()

    return new_case

# ============================================================
# Bounding Box Utilities
# ============================================================




def get_structure_bbox(mask_array, label):
    """
    Compute the bounding box of a structure in a segmentation mask.

    Parameters
    ----------
    mask_array : np.ndarray
        Segmentation mask (Z, Y, X)

    label : int
        Structure label.

    Returns
    -------
    tuple or None

    (z_min, z_max,
     y_min, y_max,
     x_min, x_max)

    Returns None if the label is absent.
    """

    coords = np.argwhere(mask_array == label)

    if len(coords) == 0:
        return None

    z_min, y_min, x_min = coords.min(axis=0)
    z_max, y_max, x_max = coords.max(axis=0)

    return (
        z_min,
        z_max,
        y_min,
        y_max,
        x_min,
        x_max,
    )

def bbox_size(bbox):
    """
    Compute the size of a bounding box.

    Parameters
    ----------
    bbox : tuple

    Returns
    -------
    (depth, height, width)
    """

    z0, z1, y0, y1, x0, x1 = bbox

    return (
        z1 - z0 + 1,
        y1 - y0 + 1,
        x1 - x0 + 1,
    )


def bbox_center(bbox):
    """
    Return the center of a bounding box.
    """

    z0, z1, y0, y1, x0, x1 = bbox

    return (
        (z0 + z1) // 2,
        (y0 + y1) // 2,
        (x0 + x1) // 2,
    )

# ============================================================
# Fixed ROI Extraction
# ============================================================

def crop_volume(
    volume,
    center,
    output_size,
    constant=0,
):
    """
    Extract a fixed-size ROI centered on a point.

    If the requested ROI extends beyond the image boundary,
    the missing region is padded automatically.

    Parameters
    ----------
    volume : ndarray
        Input volume with shape (D, H, W).

    center : tuple
        Center of the ROI as (z, y, x).

    output_size : tuple
        Desired output size as (depth, height, width).

    constant : int or float, optional
        Padding value.

    Returns
    -------
    ndarray
        Cropped (and padded if necessary) volume with the same dtype
        as the input volume.
    """

    D, H, W = volume.shape
    out_D, out_H, out_W = output_size
    cz, cy, cx = center

    # ---------------------------------------------------------
    # Desired ROI coordinates
    # ---------------------------------------------------------

    z0 = cz - out_D // 2
    z1 = z0 + out_D

    y0 = cy - out_H // 2
    y1 = y0 + out_H

    x0 = cx - out_W // 2
    x1 = x0 + out_W

    # ---------------------------------------------------------
    # Clip ROI to image boundaries
    # ---------------------------------------------------------

    src_z0 = max(z0, 0)
    src_z1 = min(z1, D)

    src_y0 = max(y0, 0)
    src_y1 = min(y1, H)

    src_x0 = max(x0, 0)
    src_x1 = min(x1, W)

    cropped = volume[
        src_z0:src_z1,
        src_y0:src_y1,
        src_x0:src_x1,
    ]

    # ---------------------------------------------------------
    # Compute padding
    # ---------------------------------------------------------

    pad_before = (
        max(0, -z0),
        max(0, -y0),
        max(0, -x0),
    )

    pad_after = (
        max(0, z1 - D),
        max(0, y1 - H),
        max(0, x1 - W),
    )

    # ---------------------------------------------------------
    # Pad if necessary
    # ---------------------------------------------------------

    if any(pad_before) or any(pad_after):

        cropped = np.pad(
            cropped,
            (
                (pad_before[0], pad_after[0]),
                (pad_before[1], pad_after[1]),
                (pad_before[2], pad_after[2]),
            ),
            mode="constant",
            constant_values=constant,
        )

    return cropped.astype(volume.dtype, copy=False)
# ============================================================
# Crop Entire Case
# ============================================================

def crop_case(
    case,
    crop_size=DEFAULT_ROI_SIZE,
    label=4,
):
    """
    Extract a fixed-size ROI centered on the pancreas.

    Parameters
    ----------
    case : dict
        Case dictionary containing CT and mask arrays.

    crop_size : tuple, optional
        Desired output size (depth, height, width).

    label : int, optional
        Label value corresponding to the pancreas.
        Default is 4 for the PANORAMA dataset.

    Returns
    -------
    dict
        Cropped case dictionary.
    """

    new_case = case.copy()

    # ---------------------------------------------------------
    # Compute pancreas bounding box
    # ---------------------------------------------------------

    bbox = get_structure_bbox(
        case["mask_array"],
        label,
    )

    if bbox is None:
        raise ValueError(
            "Pancreas label not found in mask."
        )

    center = bbox_center(bbox)

    # ---------------------------------------------------------
    # Crop CT
    # ---------------------------------------------------------

    new_case["ct_array"] = crop_volume(
        case["ct_array"],
        center,
        crop_size,
        constant=0.0,
    )

    # ---------------------------------------------------------
    # Crop mask
    # ---------------------------------------------------------

    new_case["mask_array"] = crop_volume(
        case["mask_array"],
        center,
        crop_size,
        constant=0,
    )

    return new_case

# ============================================================
# Padding
# ============================================================

def pad_volume(volume, output_size, constant=0):
    """
    Pad a 3D volume to a fixed size.

    Parameters
    ----------
    volume : np.ndarray
        (D, H, W)

    output_size : tuple
        Desired output size (D, H, W)

    constant : number
        Padding value.

    Returns
    -------
    np.ndarray
    """

    D, H, W = volume.shape
    target_D, target_H, target_W = output_size

    if D > target_D or H > target_H or W > target_W:
        raise ValueError(
            "Volume larger than requested output size."
        )

    pad_D = target_D - D
    pad_H = target_H - H
    pad_W = target_W - W

    pad_before = (
        pad_D // 2,
        pad_H // 2,
        pad_W // 2,
    )

    pad_after = (
        pad_D - pad_before[0],
        pad_H - pad_before[1],
        pad_W - pad_before[2],
    )

    return np.pad(
        volume,
        (
            (pad_before[0], pad_after[0]),
            (pad_before[1], pad_after[1]),
            (pad_before[2], pad_after[2]),
        ),
        mode="constant",
        constant_values=constant,
    )


def pad_case(
    case,
    output_size=DEFAULT_ROI_SIZE,
):
    """
    Pad a cropped case to a fixed size.

    Parameters
    ----------
    case : dict

    output_size : tuple

    Returns
    -------
    dict
    """

    new_case = case.copy()

    new_case["ct_array"] = pad_volume(
        case["ct_array"],
        output_size,
        constant=0,
    )

    new_case["mask_array"] = pad_volume(
        case["mask_array"],
        output_size,
        constant=0,
    )

    return new_case


def preprocess_case(
    case,
    target_spacing=DEFAULT_TARGET_SPACING,
    crop_size=DEFAULT_ROI_SIZE,
    hu_window=DEFAULT_HU_WINDOW,
):
    """
    Complete preprocessing pipeline for one PANORAMA case.

    Processing order
    ----------------
    1. Resample CT and mask
    2. Clip CT Hounsfield Units
    3. Normalize CT intensities
    4. Crop fixed pancreas ROI
    5. Pad to fixed output size

    Parameters
    ----------
    case : dict
        Output of ``load_case()``.

    target_spacing : tuple
        Target voxel spacing (z, y, x).

    crop_size : tuple
        Desired ROI size (depth, height, width).

    hu_window : tuple
        (lower, upper) HU clipping window.

    Returns
    -------
    dict
        Fully preprocessed case.
    """

    new_case = case.copy()

    # ---------------------------------------------------------
    # Step 1: Resample
    # ---------------------------------------------------------

    new_case = resample_case(
        new_case,
        target_spacing=target_spacing,
    )

    print(
        "After Resample   :",
        new_case["ct_array"].min(),
        new_case["ct_array"].max(),
    )

    # ---------------------------------------------------------
    # Step 2: HU clipping
    # ---------------------------------------------------------

    lower, upper = hu_window

    new_case["ct_array"] = clip_hu(
        new_case["ct_array"],
        lower=lower,
        upper=upper,
    )

    print(
        "After Clip       :",
        new_case["ct_array"].min(),
        new_case["ct_array"].max(),
    )

    # ---------------------------------------------------------
    # Step 3: Normalization
    # ---------------------------------------------------------

    new_case["ct_array"] = normalize_minmax(
        new_case["ct_array"]
    )

    print(
        "After Normalize  :",
        new_case["ct_array"].min(),
        new_case["ct_array"].max(),
    )

    # ---------------------------------------------------------
    # Step 4: Crop
    # ---------------------------------------------------------

    new_case = crop_case(
        new_case,
        crop_size=crop_size,
    )

    print(
        "After Crop       :",
        new_case["ct_array"].min(),
        new_case["ct_array"].max(),
    )

    # ---------------------------------------------------------
    # Step 5: Pad
    # ---------------------------------------------------------

    new_case = pad_case(
        new_case,
        output_size=crop_size,
    )

    print(
        "After Pad        :",
        new_case["ct_array"].min(),
        new_case["ct_array"].max(),
    )

    # ---------------------------------------------------------
    # Final data types
    # ---------------------------------------------------------

    new_case["ct_array"] = new_case["ct_array"].astype(np.float32)
    new_case["mask_array"] = new_case["mask_array"].astype(np.uint8)

    print(
        "Final            :",
        new_case["ct_array"].min(),
        new_case["ct_array"].max(),
    )

    return new_case

# ============================================================
# Dataset Creation
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.config import (
    PROCESSED_DIR,
    PROCESSED_IMAGES_DIR,
    PROCESSED_MASKS_DIR,
)

from src.io import (
    list_ct_cases,
    load_case,
    load_metadata,
)


def save_case(
    case: dict,
    image_dir: Path = PROCESSED_IMAGES_DIR,
    mask_dir: Path = PROCESSED_MASKS_DIR,
) -> dict:
    """
    Save a preprocessed CT case as NumPy arrays.

    Parameters
    ----------
    case : dict
        Preprocessed case dictionary returned by ``preprocess_case()``.
    image_dir : Path, optional
        Directory where processed CT volumes will be saved.
    mask_dir : Path, optional
        Directory where processed masks will be saved.

    Returns
    -------
    dict
        Technical metadata describing the saved case.
    """

    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    study_id = case["study_id"]

    image_path = image_dir / f"{study_id}.npy"
    mask_path = mask_dir / f"{study_id}.npy"

    # Ensure consistent data types
    ct_array = case["ct_array"].astype(np.float32)
    mask_array = case["mask_array"].astype(np.uint8)

    np.save(image_path, ct_array)
    np.save(mask_path, mask_array)

    return {
        "study_id": study_id,

        # File locations
        "image_path": str(image_path.relative_to(PROCESSED_DIR)),
        "mask_path": str(mask_path.relative_to(PROCESSED_DIR)),

        # Image information
        "image_shape": ",".join(map(str, ct_array.shape)),
        "mask_shape": ",".join(map(str, mask_array.shape)),

        "image_dtype": str(ct_array.dtype),
        "mask_dtype": str(mask_array.dtype),

        "mask_labels": ",".join(
            map(str, np.unique(mask_array).astype(int))
        ),

        # Preprocessing configuration
        "roi_size": ",".join(map(str, DEFAULT_ROI_SIZE)),
        "target_spacing": ",".join(map(str, DEFAULT_TARGET_SPACING)),
        "hu_window": ",".join(map(str, DEFAULT_HU_WINDOW)),
    }


def merge_clinical_metadata(
    record,
    metadata_row,
):
    """
    Merge clinical metadata into a processed case record.

    Parameters
    ----------
    record : dict
        Metadata record produced by save_case().

    metadata_row : pandas.Series
        One row from the clinical metadata DataFrame.

    Returns
    -------
    dict
        Updated metadata record containing both technical and
        clinical information.
    """

    record = record.copy()

    # --------------------------------------------------------
    # Patient ID
    # --------------------------------------------------------

    record["patient_id"] = int(
        metadata_row["PANORAMA_patient_id"]
    )

    # --------------------------------------------------------
    # Patient age
    # --------------------------------------------------------

    age = metadata_row["patient_age"]

    if pd.isna(age):

        record["patient_age"] = None

    else:

        age = str(age).strip()

        if age.endswith("Y"):
            age = age[:-1]

        try:

            record["patient_age"] = int(age)

        except ValueError:

            record["patient_age"] = None

    # --------------------------------------------------------
    # Patient sex
    # --------------------------------------------------------

    record["patient_sex"] = metadata_row["patient_sex"]

    # --------------------------------------------------------
    # Scanner
    # --------------------------------------------------------

    record["scanner"] = metadata_row["scanner"]

    # --------------------------------------------------------
    # Diagnosis
    # --------------------------------------------------------

    record["diagnosis"] = metadata_row["label"]

    # --------------------------------------------------------
    # Diagnosis source
    # --------------------------------------------------------

    record["diagnosis_source"] = metadata_row["level"]

    return record

def update_metadata(
    records,
    metadata_path=PROCESSED_DIR / "metadata.csv",
):
    """
    Update the processed dataset metadata.

    This function appends newly processed case records to the existing
    metadata file, removes duplicate study IDs, and saves the updated
    metadata to disk.

    Parameters
    ----------
    records : list of dict
        Metadata records returned by ``save_case()``.

    metadata_path : pathlib.Path, optional
        Path to the metadata CSV file.

    Returns
    -------
    pandas.DataFrame
        Updated metadata DataFrame.
    """

    if len(records) == 0:
        raise ValueError("No metadata records were provided.")

    new_df = pd.DataFrame(records)

    # Load existing metadata if available
    if metadata_path.exists():

        existing_df = pd.read_csv(metadata_path)

        metadata_df = pd.concat(
            [existing_df, new_df],
            ignore_index=True,
        )

        metadata_df = metadata_df.drop_duplicates(
            subset="study_id",
            keep="last",
        )

    else:

        metadata_df = new_df

    metadata_df = metadata_df.sort_values(
        by="study_id"
    ).reset_index(drop=True)

    metadata_df.to_csv(
        metadata_path,
        index=False,
    )

    return metadata_df


def repair_metadata(
    metadata_path=PROCESSED_DIR / "metadata.csv",
):
    """
    Repair metadata.csv by rebuilding missing metadata records
    from already processed .npy files.

    Returns
    -------
    pandas.DataFrame
        Updated metadata dataframe.
    """

    # --------------------------------------------------------
    # Existing processed metadata
    # --------------------------------------------------------

    if metadata_path.exists():

        metadata_df = pd.read_csv(metadata_path)

    else:

        metadata_df = pd.DataFrame()

    existing_ids = set()

    if len(metadata_df) > 0:

        existing_ids = set(metadata_df["study_id"])

    # --------------------------------------------------------
    # Clinical metadata
    # --------------------------------------------------------

    clinical = load_metadata().copy()

    clinical = clinical.rename(
        columns={
            "PANORAMA_study_id": "study_id",
        }
    )

    clinical = clinical.set_index("study_id")

    # --------------------------------------------------------
    # Scan processed images
    # --------------------------------------------------------

    records = []

    image_files = sorted(
        PROCESSED_IMAGES_DIR.glob("*.npy")
    )

    for image_path in tqdm(
        image_files,
        desc="Repairing metadata",
    ):

        study_id = image_path.stem

        if study_id in existing_ids:
            continue

        mask_path = (
            PROCESSED_MASKS_DIR /
            image_path.name
        )

        if not mask_path.exists():

            print(
                f"Missing mask for {study_id}"
            )
            continue

        # --------------------------------------------
        # Load processed arrays
        # --------------------------------------------

        image = np.load(image_path)

        mask = np.load(mask_path)

        record = {

            "study_id": study_id,

            "image_path": str(
                image_path.relative_to(PROCESSED_DIR)
            ),

            "mask_path": str(
                mask_path.relative_to(PROCESSED_DIR)
            ),

            "image_shape": ",".join(
                map(str, image.shape)
            ),

            "mask_shape": ",".join(
                map(str, mask.shape)
            ),

            "image_dtype": str(image.dtype),

            "mask_dtype": str(mask.dtype),

            "mask_labels": ",".join(
                map(
                    str,
                    np.unique(mask).astype(int),
                )
            ),

            "roi_size": ",".join(
                map(str, DEFAULT_ROI_SIZE)
            ),

            "target_spacing": ",".join(
                map(str, DEFAULT_TARGET_SPACING)
            ),

            "hu_window": ",".join(
                map(str, DEFAULT_HU_WINDOW)
            ),
        }

        # --------------------------------------------
        # Merge clinical metadata
        # --------------------------------------------

        if study_id in clinical.index:

            record = merge_clinical_metadata(
                record,
                clinical.loc[study_id],
            )

        records.append(record)

    # --------------------------------------------------------
    # Nothing to repair
    # --------------------------------------------------------

    if len(records) == 0:

        print("Metadata already complete.")

        return metadata_df

    # --------------------------------------------------------
    # Append repaired records
    # --------------------------------------------------------

    repaired = update_metadata(records)

    print()

    print("=" * 60)
    print(f"Recovered {len(records)} missing metadata records.")
    print("=" * 60)

    return repaired


from tqdm.auto import tqdm

from src.io import (
    list_ct_cases,
    load_case,
    load_metadata,
)

def verify_dataset(
    metadata_path=PROCESSED_DIR / "metadata.csv",
):
    """
    Verify the integrity of the processed dataset.

    Checks
    ------
    • Metadata file exists
    • Image/mask files exist
    • Correct image shape
    • Correct mask shape
    • Correct dtypes
    • Image intensity range
    • Valid segmentation labels

    Returns
    -------
    tuple
        (summary, report)

        summary : dict
            Dataset statistics.

        report : pandas.DataFrame
            One row per study describing any detected issues.
    """

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}"
        )

    metadata = pd.read_csv(metadata_path)

    expected_shape = DEFAULT_ROI_SIZE
    expected_image_dtype = np.float32
    expected_mask_dtype = np.uint8

    valid_labels = {0, 1, 2, 3, 4, 5, 6}

    eps = 1e-6

    summary = {
        "num_cases": len(metadata),
        "missing_images": 0,
        "missing_masks": 0,
        "invalid_image_shape": 0,
        "invalid_mask_shape": 0,
        "invalid_image_dtype": 0,
        "invalid_mask_dtype": 0,
        "invalid_image_range": 0,
        "invalid_mask_labels": 0,
        "duplicate_study_ids": metadata["study_id"].duplicated().sum(),
    }

    report_rows = []

    for _, row in tqdm(
        metadata.iterrows(),
        total=len(metadata),
        desc="Verifying dataset",
    ):

        study_id = row["study_id"]

        image_path = PROCESSED_DIR / row["image_path"]
        mask_path = PROCESSED_DIR / row["mask_path"]

        issues = []

        # --------------------------------------------------
        # Check file existence
        # --------------------------------------------------

        if not image_path.exists():
            summary["missing_images"] += 1
            issues.append("missing_image")

        if not mask_path.exists():
            summary["missing_masks"] += 1
            issues.append("missing_mask")

        if issues:
            report_rows.append(
                {
                    "study_id": study_id,
                    "status": "FAILED",
                    "issues": "; ".join(issues),
                }
            )
            continue

        # --------------------------------------------------
        # Load arrays
        # --------------------------------------------------

        image = np.load(image_path)
        mask = np.load(mask_path)

        # --------------------------------------------------
        # Shape
        # --------------------------------------------------

        if image.shape != expected_shape:
            summary["invalid_image_shape"] += 1
            issues.append("image_shape")

        if mask.shape != expected_shape:
            summary["invalid_mask_shape"] += 1
            issues.append("mask_shape")

        # --------------------------------------------------
        # Dtype
        # --------------------------------------------------

        if image.dtype != expected_image_dtype:
            summary["invalid_image_dtype"] += 1
            issues.append("image_dtype")

        if mask.dtype != expected_mask_dtype:
            summary["invalid_mask_dtype"] += 1
            issues.append("mask_dtype")

        # --------------------------------------------------
        # Intensity range
        # --------------------------------------------------

        img_min = float(image.min())
        img_max = float(image.max())

        if img_min < -eps or img_max > 1.0 + eps:
            summary["invalid_image_range"] += 1
            issues.append(
                f"image_range ({img_min:.6f}, {img_max:.6f})"
            )

        # --------------------------------------------------
        # Mask labels
        # --------------------------------------------------

        labels = set(np.unique(mask).astype(int))

        if not labels.issubset(valid_labels):
            summary["invalid_mask_labels"] += 1
            issues.append(
                f"mask_labels={sorted(labels)}"
            )

        # --------------------------------------------------
        # Report row
        # --------------------------------------------------

        report_rows.append(
            {
                "study_id": study_id,
                "status": "PASS" if len(issues) == 0 else "FAILED",
                "issues": "; ".join(issues),
                "image_min": img_min,
                "image_max": img_max,
                "mask_labels": ",".join(map(str, sorted(labels))),
            }
        )

    report = pd.DataFrame(report_rows)

    print("=" * 60)
    print("Processed Dataset Verification")
    print("=" * 60)

    for key, value in summary.items():
        print(f"{key:<25}: {value}")

    print("=" * 60)

    if (report["status"] == "FAILED").any():
        print("\nFailed cases:")
        print(report.loc[report["status"] == "FAILED"])
    else:
        print("\n✓ All processed cases passed verification.")

    return summary, report


def process_dataset(
    study_ids=None,
    overwrite=False,
):
    """
    Process and save a batch of CT studies.

    Parameters
    ----------
    study_ids : list of str, optional
        Study IDs to process. If None, all currently available CT
        studies in ``raw_ct`` are processed.

    overwrite : bool, optional
        Whether to overwrite already processed cases.

    Returns
    -------
    pandas.DataFrame
        Updated metadata DataFrame.
    """

    # --------------------------------------------------------
    # Load clinical metadata
    # --------------------------------------------------------

    metadata = load_metadata().copy()

    metadata = metadata.rename(
        columns={
            "PANORAMA_study_id": "study_id",
        }
    )

    metadata = metadata.set_index("study_id")

    # --------------------------------------------------------
    # Determine studies to process
    # --------------------------------------------------------

    if study_ids is None:
        study_ids = sorted(list_ct_cases())

    records = []

    # --------------------------------------------------------
    # Process each study
    # --------------------------------------------------------

    for study_id in tqdm(
        study_ids,
        desc="Processing dataset",
    ):

        image_path = PROCESSED_IMAGES_DIR / f"{study_id}.npy"
        mask_path = PROCESSED_MASKS_DIR / f"{study_id}.npy"

        if (
            image_path.exists()
            and mask_path.exists()
            and not overwrite
        ):
            print(f"Skipping {study_id}")
            continue

        try:

            # --------------------------------------------
            # Load raw case
            # --------------------------------------------

            case = load_case(study_id)

            # --------------------------------------------
            # Preprocess
            # --------------------------------------------

            case = preprocess_case(case)

            # --------------------------------------------
            # Save processed arrays
            # --------------------------------------------

            record = save_case(case)

            # --------------------------------------------
            # Merge clinical metadata
            # --------------------------------------------

            if study_id in metadata.index:

                record = merge_clinical_metadata(
                    record,
                    metadata.loc[study_id],
                )

            else:

                print(
                    f"Warning: Clinical metadata not found for {study_id}"
                )

            records.append(record)

        except Exception as e:

            print(f"{study_id}: {e}")

    # --------------------------------------------------------
    # Save metadata.csv
    # --------------------------------------------------------

    return update_metadata(records)



# ============================================================
# Dataset Analysis
# ============================================================


def analyze_roi_sizes(
    label=4,
    target_spacing=DEFAULT_TARGET_SPACING,
    progress_every=25,
):
    """
    Analyze bounding-box sizes of a structure across the dataset.

    Parameters
    ----------
    label : int
        Structure label.
        4 = pancreas
        1 = tumor

    target_spacing : tuple
        Spacing used before measuring the ROI.

    progress_every : int
        Print progress every N cases.

    Returns
    -------
    pandas.DataFrame
        Columns:
        study_id
        depth
        height
        width
        volume
    """

    results = []

    study_ids = list_ct_cases()

    total = len(study_ids)

    for i, study_id in enumerate(study_ids, start=1):

        if i % progress_every == 0 or i == total:
            print(f"{i}/{total}")

        try:

            mask_image, _, _, _ = load_mask(study_id)

            mask_image = resample_mask(
                mask_image,
                target_spacing,
            )

            mask_array = sitk.GetArrayFromImage(mask_image)

            bbox = get_structure_bbox(
                mask_array,
                label,
            )

            if bbox is None:
                continue

            depth, height, width = bbox_size(bbox)

            results.append({

                "study_id": study_id,

                "depth": depth,

                "height": height,

                "width": width,

                "volume": depth * height * width,

            })

        except Exception as e:

            print(f"Skipped {study_id}: {e}")

    return pd.DataFrame(results)





__all__ = [
    "DEFAULT_TARGET_SPACING",
    "DEFAULT_HU_WINDOW",
    "DEFAULT_ROI_SIZE",

    "clip_hu",
    "normalize_minmax",
    "normalize_zscore",

    "resample_image",
    "resample_ct",
    "resample_mask",
    "resample_case",

    "get_structure_bbox",
    "bbox_center",
    "bbox_size",

    "crop_volume",
    "crop_case",

    "pad_volume",
    "pad_case",

    "preprocess_case",
    "preprocess_array",

    "analyze_roi_sizes",
]