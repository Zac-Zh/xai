"""Simple overlay utilities for detection and segmentation."""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np


def overlay_detection(image: np.ndarray, bbox: Iterable[int], mask: Optional[np.ndarray]) -> np.ndarray:
    """Overlay bbox rectangle and binary mask onto the RGB image.

    Args:
        image: HxWx3 uint8 RGB.
        bbox: [x1,y1,x2,y2]
        mask: HxW bool or None
    Returns:
        Annotated image as uint8 array.
    """
    out = image.copy()
    h, w = int(out.shape[0]), int(out.shape[1])
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = int(np.clip(x1, 0, w - 1))
    y1 = int(np.clip(y1, 0, h - 1))
    x2 = int(np.clip(x2, 0, w - 1))
    y2 = int(np.clip(y2, 0, h - 1))
    # Rectangle in green
    out[y1 : y1 + 2, x1:x2] = [30, 200, 30]
    out[y2 - 2 : y2, x1:x2] = [30, 200, 30]
    out[y1:y2, x1 : x1 + 2] = [30, 200, 30]
    out[y1:y2, x2 - 2 : x2] = [30, 200, 30]
    # Mask with alpha
    if mask is not None:
        alpha = 0.3
        m = mask.astype(bool)
        overlay = out.copy()
        overlay[m] = [200, 30, 30]
        out = (alpha * overlay + (1 - alpha) * out).astype(np.uint8)
    return out

