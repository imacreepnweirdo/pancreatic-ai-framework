"""
General utility functions for the PANORAMA dataset.

Author: Shahriar Ahmed
"""

import numpy as np


# ============================================================
# Slice search
# ============================================================

def find_structure(mask, label):
    """
    Find all slices containing a given structure.

    Parameters
    ----------
    mask : ndarray
        Segmentation mask.

    label : int
        Label value.

    Returns
    -------
    ndarray
        Slice indices.
    """

    return np.where(
        np.any(mask == label, axis=(1, 2))
    )[0]


def find_pancreas_slices(mask):
    """
    Return slices containing the pancreas.
    """

    return find_structure(mask, 4)


def find_tumor_slices(mask):
    """
    Return slices containing PDAC.
    """

    return find_structure(mask, 1)


# ============================================================
# Bounding boxes
# ============================================================

def get_structure_bbox(mask, label=4):
    """
    Compute the 3D bounding box of a structure.

    Parameters
    ----------
    mask : ndarray

    label : int

    Returns
    -------
    tuple or None

    (zmin, zmax,
     ymin, ymax,
     xmin, xmax)
    """

    coords = np.argwhere(mask == label)

    if len(coords) == 0:
        return None

    zmin, ymin, xmin = coords.min(axis=0)
    zmax, ymax, xmax = coords.max(axis=0)

    return (
        zmin, zmax,
        ymin, ymax,
        xmin, xmax,
    )


def bbox_size(bbox):
    """
    Compute bounding-box size.

    Returns
    -------
    tuple

    (depth, height, width)
    """

    zmin, zmax, ymin, ymax, xmin, xmax = bbox

    return (
        zmax - zmin + 1,
        ymax - ymin + 1,
        xmax - xmin + 1,
    )


def bbox_center(bbox):
    """
    Compute bounding-box center.

    Returns
    -------
    tuple

    (z, y, x)
    """

    zmin, zmax, ymin, ymax, xmin, xmax = bbox

    return (
        (zmin + zmax) // 2,
        (ymin + ymax) // 2,
        (xmin + xmax) // 2,
    )


# ============================================================
# Cropping
# ============================================================

def crop_to_bbox(volume, bbox):
    """
    Crop a 3D volume to a bounding box.

    Parameters
    ----------
    volume : ndarray

    bbox : tuple

    Returns
    -------
    ndarray
    """

    zmin, zmax, ymin, ymax, xmin, xmax = bbox

    return volume[
        zmin:zmax + 1,
        ymin:ymax + 1,
        xmin:xmax + 1,
    ]


# ============================================================
# Padding
# ============================================================

def pad_volume(volume, target_shape, value=0):
    """
    Pad a volume symmetrically to a target shape.

    Parameters
    ----------
    volume : ndarray

    target_shape : tuple
        (depth, height, width)

    value : number

    Returns
    -------
    ndarray
    """

    current = volume.shape

    pad_width = []

    for cur, tgt in zip(current, target_shape):

        diff = max(tgt - cur, 0)

        before = diff // 2
        after = diff - before

        pad_width.append((before, after))

    return np.pad(
        volume,
        pad_width,
        mode="constant",
        constant_values=value,
    )


# ============================================================
# Case information
# ============================================================

def describe_case(case):
    """
    Print useful information about a case.
    """

    print("=" * 50)
    print(f"Study ID   : {case['study_id']}")
    print("=" * 50)

    print(f"Mask type  : {case['mask_type']}")
    print(f"Shape      : {case['ct_array'].shape}")
    print(f"Spacing    : {case['spacing']}")
    print(f"Origin     : {case['origin']}")
    print(f"Direction  : {case['direction']}")

    validation = case.get("validation")

    if validation is not None:

        print("\nValidation")

        for key, value in validation.items():
            print(f"  {key:<18}: {value}")