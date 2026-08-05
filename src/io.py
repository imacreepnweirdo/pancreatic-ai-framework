"""
io.py

Input/output utilities for the PANORAMA pancreatic CT dataset.

Author: Shahriar Ahmed
"""

import random

import pandas as pd
import SimpleITK as sitk

from .config import (
    RAW_CT_DIR,
    MANUAL_LABEL_DIR,
    AUTO_LABEL_DIR,
    METADATA_DIR,
)


# ============================================================
# Helper Functions
# ============================================================

def normalize_study_id(study_id):
    """
    Normalize PANORAMA study IDs.

    Examples
    --------
    100000_00001
    100000_00001_0000
    100000_00001.nii.gz
    100000_00001_0000.nii.gz

    Returns
    -------
    str
        Example:
        100000_00001
    """

    study_id = str(study_id)

    if study_id.endswith(".nii.gz"):
        study_id = study_id[:-7]

    if study_id.endswith("_0000"):
        study_id = study_id[:-5]

    return study_id


# ============================================================
# Dataset Listing
# ============================================================

def list_ct_cases():
    """
    Return all CT study IDs.
    """

    return sorted(
        p.name.replace("_0000.nii.gz", "")
        for p in RAW_CT_DIR.glob("*.nii.gz")
    )


def list_manual_cases():
    """
    Return all manually annotated study IDs.
    """

    return sorted(
        normalize_study_id(p.name)
        for p in MANUAL_LABEL_DIR.glob("*.nii.gz")
    )


def list_automatic_cases():
    """
    Return all automatically annotated study IDs.
    """

    return sorted(
        normalize_study_id(p.name)
        for p in AUTO_LABEL_DIR.glob("*.nii.gz")
    )


# ============================================================
# Metadata
# ============================================================

def load_metadata():
    """
    Load clinical metadata.
    """

    metadata_path = METADATA_DIR / "clinical_information.xlsx"

    if not metadata_path.exists():

        raise FileNotFoundError(
            f"Metadata not found:\n{metadata_path}"
        )

    return pd.read_excel(metadata_path)


# ============================================================
# Paths
# ============================================================

def get_ct_path(study_id):
    """
    Return CT file path.
    """

    study_id = normalize_study_id(study_id)

    return RAW_CT_DIR / f"{study_id}_0000.nii.gz"


def get_mask_path(study_id):
    """
    Return segmentation path.

    Priority
    --------
    Manual
    Automatic

    Returns
    -------
    tuple
        (path, mask_type)
    """

    study_id = normalize_study_id(study_id)

    manual = MANUAL_LABEL_DIR / f"{study_id}.nii.gz"

    if manual.exists():
        return manual, "Manual"

    automatic = AUTO_LABEL_DIR / f"{study_id}.nii.gz"

    if automatic.exists():
        return automatic, "Automatic"

    return None, None


# ============================================================
# Image Loading
# ============================================================

def load_image(path):
    """
    Load a NIfTI image.

    Returns
    -------
    tuple
        (SimpleITK.Image, ndarray)
    """

    image = sitk.ReadImage(str(path))

    array = sitk.GetArrayFromImage(image)

    return image, array


def load_ct(study_id):
    """
    Load CT scan.
    """

    ct_path = get_ct_path(study_id)

    if not ct_path.exists():

        raise FileNotFoundError(
            f"CT not found:\n{ct_path}"
        )

    return load_image(ct_path)


def load_mask(study_id):
    """
    Load segmentation mask.

    Returns
    -------
    tuple
        (
            image,
            array,
            mask_type,
            path
        )
    """

    mask_path, mask_type = get_mask_path(study_id)

    if mask_path is None:

        raise FileNotFoundError(
            f"No segmentation found for {study_id}"
        )

    image, array = load_image(mask_path)

    return image, array, mask_type, mask_path


# ============================================================
# Case Loading
# ============================================================

def load_case(study_id):
    """
    Load a complete PANORAMA case.

    Returns
    -------
    dict
    """

    study_id = normalize_study_id(study_id)

    ct_path = get_ct_path(study_id)

    ct_image, ct_array = load_ct(study_id)

    (
        mask_image,
        mask_array,
        mask_type,
        mask_path,
    ) = load_mask(study_id)

    return {

        "study_id": study_id,

        "ct_path": ct_path,

        "mask_path": mask_path,

        "mask_type": mask_type,

        "ct_image": ct_image,

        "mask_image": mask_image,

        "ct_array": ct_array,

        "mask_array": mask_array,

        "spacing": ct_image.GetSpacing(),

        "origin": ct_image.GetOrigin(),

        "direction": ct_image.GetDirection(),

    }


# ============================================================
# Convenience Functions
# ============================================================

def case_exists(study_id):
    """
    Check whether a CT exists.
    """

    return get_ct_path(study_id).exists()


def random_case(label=None):
    """
    Load a random case.

    Parameters
    ----------
    label : str or None

        None
        "PDAC"
        "non-PDAC"

    Returns
    -------
    dict
    """

    metadata = load_metadata()

    if label is not None:

        metadata = metadata[
            metadata["label"] == label
        ]

    study_ids = set(list_ct_cases())

    metadata = metadata[
        metadata["PANORAMA_study_id"].astype(str).isin(study_ids)
    ]

    if len(metadata) == 0:

        raise ValueError(
            "No matching cases found."
        )

    study_id = random.choice(
        metadata["PANORAMA_study_id"].astype(str).tolist()
    )

    return load_case(study_id)


def print_dataset_summary():
    """
    Print a dataset summary.
    """

    print("=" * 40)
    print("PANORAMA Dataset")
    print("=" * 40)

    print(f"CT volumes          : {len(list_ct_cases())}")
    print(f"Manual labels       : {len(list_manual_cases())}")
    print(f"Automatic labels    : {len(list_automatic_cases())}")
    print(f"Metadata rows       : {len(load_metadata())}")

    print("=" * 40)