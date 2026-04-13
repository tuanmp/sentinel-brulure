import numpy as np


def compute_nbr(bands: np.ndarray):
    """
    NBR = (NIR - SWIR2) / (NIR + SWIR2)
    bands shape: (H, W, 7) — band order: B02,B03,B04,B08,B11,B12,mask
    """
    nir   = bands[3].astype(float)  # B08
    swir2 = bands[5].astype(float)  # B12
    mask  = bands[6]                # dataMask

    nbr = np.where(
        (nir + swir2) > 0,
        (nir - swir2) / (nir + swir2),
        np.nan  # avoid division by zero / no-data pixels
    )
    nbr[mask == 0] = np.nan  # mask out invalid pixels
    return nbr
