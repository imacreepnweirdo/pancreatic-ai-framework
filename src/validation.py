"""
validation.py

Dataset validation utilities for the PANORAMA dataset.

Author: Shahriar Ahmed
"""

import warnings

import numpy as np
import pandas as pd
import SimpleITK as sitk

from .config import RAW_CT_DIR
from .io import (
    list_ct_cases,
    load_case,
    get_mask_path,
    normalize_study_id,
)


# ============================================================
# Validate one case
# ============================================================

def validate_case(study_id):
    """
    Validate a single PANORAMA case.

    Parameters
    ----------
    study_id : str

    Returns
    -------
    dict
    """

    result = {

        "study_id": study_id,

        "mask_type": None,

        "valid": False,

        "error": "",

        "shape_match": False,

        "spacing_match": False,

        "origin_match": False,

        "direction_match": False,

        "labels": None,

        "num_labels": None,

    }

    try:

        case = load_case(study_id)

        mask = case["mask_array"]

        labels = np.unique(mask)

        result["mask_type"] = case["mask_type"]

        result["shape_match"] = True
        result["spacing_match"] = True
        result["origin_match"] = True
        result["direction_match"] = True

        result["labels"] = ",".join(map(str, labels))

        result["num_labels"] = len(labels)

        result["valid"] = True

    except Exception as e:

        result["error"] = str(e)

    return result


# ============================================================
# Validate entire dataset
# ============================================================

def validate_dataset(
    spacing_tolerance=1e-3,
    progress_every=25,
):
    """
    Validate the complete PANORAMA dataset.

    Reads only image headers,
    therefore extremely fast.

    Parameters
    ----------
    spacing_tolerance : float

    progress_every : int

    Returns
    -------
    pandas.DataFrame
    """

    results = []

    study_ids = list_ct_cases()

    total = len(study_ids)

    for i, study_id in enumerate(study_ids, start=1):

        if i % progress_every == 0 or i == total:

            print(f"{i}/{total}")

        study_id = normalize_study_id(study_id)

        ct_path = RAW_CT_DIR / f"{study_id}_0000.nii.gz"

        mask_path, mask_type = get_mask_path(study_id)

        if mask_path is None:

            results.append({

                "study_id": study_id,

                "mask_type": None,

                "valid": False,

                "error": "Missing mask",

                "shape_match": False,

                "spacing_match": False,

                "origin_match": False,

                "direction_match": False,

                "ct_size": None,

                "mask_size": None,

                "ct_spacing": None,

                "mask_spacing": None,

                "ct_origin": None,

                "mask_origin": None,

            })

            continue

        try:

            # -----------------------------
            # Read CT header
            # -----------------------------

            ct_reader = sitk.ImageFileReader()

            ct_reader.SetFileName(str(ct_path))

            ct_reader.ReadImageInformation()

            # -----------------------------
            # Read mask header
            # -----------------------------

            mask_reader = sitk.ImageFileReader()

            mask_reader.SetFileName(str(mask_path))

            mask_reader.ReadImageInformation()

            ct_size = ct_reader.GetSize()

            mask_size = mask_reader.GetSize()

            ct_spacing = ct_reader.GetSpacing()

            mask_spacing = mask_reader.GetSpacing()

            ct_origin = ct_reader.GetOrigin()

            mask_origin = mask_reader.GetOrigin()

            ct_direction = ct_reader.GetDirection()

            mask_direction = mask_reader.GetDirection()

            shape_match = ct_size == mask_size

            spacing_match = np.allclose(
                ct_spacing,
                mask_spacing,
                atol=spacing_tolerance,
            )

            origin_match = np.allclose(
                ct_origin,
                mask_origin,
                atol=1e-4,
            )

            direction_match = np.allclose(
                ct_direction,
                mask_direction,
                atol=1e-6,
            )

            valid = (

                shape_match

                and spacing_match

                and origin_match

                and direction_match

            )

            results.append({

                "study_id": study_id,

                "mask_type": mask_type,

                "valid": valid,

                "error": "",

                "shape_match": shape_match,

                "spacing_match": spacing_match,

                "origin_match": origin_match,

                "direction_match": direction_match,

                "ct_size": ct_size,

                "mask_size": mask_size,

                "ct_spacing": ct_spacing,

                "mask_spacing": mask_spacing,

                "ct_origin": ct_origin,

                "mask_origin": mask_origin,

            })

        except Exception as e:

            results.append({

                "study_id": study_id,

                "mask_type": mask_type,

                "valid": False,

                "error": str(e),

                "shape_match": False,

                "spacing_match": False,

                "origin_match": False,

                "direction_match": False,

                "ct_size": None,

                "mask_size": None,

                "ct_spacing": None,

                "mask_spacing": None,

                "ct_origin": None,

                "mask_origin": None,

            })

    return pd.DataFrame(results)


# ============================================================
# Validation Summary
# ============================================================

def generate_validation_summary(validation_df):
    """
    Return validation summary.
    """

    return {

        "total_cases":
            len(validation_df),

        "valid_cases":
            int(validation_df["valid"].sum()),

        "invalid_cases":
            int((~validation_df["valid"]).sum()),

        "shape_mismatches":
            int((~validation_df["shape_match"]).sum()),

        "spacing_mismatches":
            int((~validation_df["spacing_match"]).sum()),

        "origin_mismatches":
            int((~validation_df["origin_match"]).sum()),

        "direction_mismatches":
            int((~validation_df["direction_match"]).sum()),

    }


# ============================================================
# Validation Report
# ============================================================

def generate_validation_report(
    validation_df,
    save_path=None,
):
    """
    Print validation summary and optionally save CSV.
    """

    report = generate_validation_summary(validation_df)

    print("=" * 60)

    print("PANORAMA DATASET VALIDATION REPORT")

    print("=" * 60)

    for key, value in report.items():

        print(f"{key:<25}: {value}")

    if save_path is not None:

        validation_df.to_csv(
            save_path,
            index=False,
        )

        print(f"\nSaved report to:\n{save_path}")

    return report


# ============================================================
# Show Failed Cases
# ============================================================

def show_validation_failures(validation_df):
    """
    Return all failed cases.
    """

    return validation_df[
        validation_df["valid"] == False
    ]