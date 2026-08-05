"""
Visualization utilities for the PANORAMA CT dataset.

Author: Shahriar Ahmed
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import PROCESSED_DIR

from .io import random_case
from .utils import (
    describe_case,
    find_tumor_slices,
)


# ============================================================
# Windowing
# ============================================================

def window_image(image, level=40, width=400):
    """
    Apply CT windowing.

    Parameters
    ----------
    image : ndarray
        CT volume.

    level : int
        Window level.

    width : int
        Window width.

    Returns
    -------
    ndarray
        Windowed image scaled to [0, 1].
    """

    lower = level - width / 2
    upper = level + width / 2

    image = np.clip(image, lower, upper)

    image = (image - lower) / width

    return image


# ============================================================
# Slice visualization
# ============================================================

def show_slice(
    image,
    slice_index,
    cmap="gray",
    figsize=(6, 6),
    title=None,
):
    """
    Display a single slice.
    """

    plt.figure(figsize=figsize)

    plt.imshow(image[slice_index], cmap=cmap)

    if title is not None:
        plt.title(title)

    plt.axis("off")

    plt.show()


# ============================================================
# Overlay visualization
# ============================================================

def overlay_mask(
    image,
    mask,
    slice_index,
    alpha=0.4,
):
    """
    Overlay segmentation mask on CT slice.
    """

    img = window_image(image)

    plt.figure(figsize=(7, 7))

    plt.imshow(img[slice_index], cmap="gray")

    plt.imshow(
        mask[slice_index],
        cmap="jet",
        alpha=alpha,
    )

    plt.title(f"Slice {slice_index}")

    plt.axis("off")

    plt.show()


# ============================================================
# Case visualization
# ============================================================

def show_middle_slice(case):
    """
    Display the middle CT slice.
    """

    image = case["ct_array"]

    middle = image.shape[0] // 2

    show_slice(
        window_image(image),
        middle,
        title=case["study_id"],
    )


def show_case(
    case,
    slice_index=None,
    overlay=True,
):
    """
    Display one CT case.

    Parameters
    ----------
    case : dict
        Case dictionary containing at least
        "ct_array", "mask_array", and "study_id".

    slice_index : int, optional
        Slice to display. If None, an informative
        slice is selected automatically.

    overlay : bool
        Whether to overlay the segmentation mask.
    """

    image = case["ct_array"]
    mask = case["mask_array"]

    # --------------------------------------------------
    # Automatically choose a good slice
    # --------------------------------------------------

    if slice_index is None:

        # Prefer pancreas (label 4)
        pancreas = mask == 4

        if pancreas.any():

            areas = pancreas.sum(axis=(1, 2))
            slice_index = int(np.argmax(areas))

        else:

            # Otherwise use the slice with the
            # largest segmented region.
            foreground = mask > 0

            if foreground.any():

                areas = foreground.sum(axis=(1, 2))
                slice_index = int(np.argmax(areas))

            else:

                # Fallback
                slice_index = image.shape[0] // 2

    # --------------------------------------------------
    # Display
    # --------------------------------------------------

    if overlay:

        overlay_mask(
            image,
            mask,
            slice_index,
        )

    else:

        show_slice(
            window_image(image),
            slice_index,
            title=case["study_id"],
        )



def show_random_case(label=None):
    """
    Display a random case.

    If a tumor exists, the middle tumor slice is shown.
    Otherwise the middle CT slice is displayed.
    """

    case = random_case(label)

    describe_case(case)

    tumor_slices = find_tumor_slices(
        case["mask_array"]
    )

    if len(tumor_slices):

        show_case(
            case,
            slice_index=tumor_slices[len(tumor_slices) // 2],
        )

    else:

        show_middle_slice(case)


# ============================================================
# Saving
# ============================================================

def save_figure(
    filename,
    dpi=300,
):
    """
    Save the current matplotlib figure.
    """

    plt.savefig(
        filename,
        dpi=dpi,
        bbox_inches="tight",
    )



# ============================================================
# Print Case Information
# ============================================================

def print_case_info(case):
    """
    Print basic information about a case.
    """

    print("=" * 40)
    print(case["study_id"])
    print("=" * 40)

    print(f"Mask type : {case['mask_type']}")
    print(f"Shape     : {case['ct_array'].shape}")
    print(f"Spacing   : {case['spacing']}")
    print(f"Origin    : {case['origin']}")



def load_processed_case(
    study_id,
    metadata_path=PROCESSED_DIR / "metadata.csv",
):
    """
    Load a processed case from metadata.csv.
    """

    metadata = pd.read_csv(metadata_path)

    row = metadata.loc[
        metadata["study_id"] == study_id
    ]

    if len(row) == 0:
        raise ValueError(f"{study_id} not found.")

    row = row.iloc[0]

    return {
        "study_id": study_id,
        "ct_array": np.load(
            PROCESSED_DIR / row["image_path"]
        ),
        "mask_array": np.load(
            PROCESSED_DIR / row["mask_path"]
        ),
    }


def show_processed_case(
    study_id,
    metadata_path=PROCESSED_DIR / "metadata.csv",
):
    """
    Display a processed case.

    Automatically selects the slice containing
    the largest pancreas region.
    """

    case = load_processed_case(
        study_id,
        metadata_path,
    )

    mask = case["mask_array"]

    pancreas = mask == 4

    if pancreas.any():

        areas = pancreas.sum(axis=(1, 2))

        slice_index = int(np.argmax(areas))

    else:

        slice_index = mask.shape[0] // 2

    image = case["ct_array"]

    plt.figure(figsize=(7, 7))

    plt.imshow(
        image[slice_index],
        cmap="gray",
    )

    plt.imshow(
        np.ma.masked_where(
            mask[slice_index] == 0,
            mask[slice_index],
        ),
        cmap="jet",
        alpha=0.4,
    )

    plt.title(study_id)

    plt.axis("off")

    plt.show()