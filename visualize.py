"""visualize.py - Visualization utilities for Go board detection."""

import cv2
import numpy as np


def annotate_original(
    image: np.ndarray,
    mask: np.ndarray,
    corners: np.ndarray,
) -> np.ndarray:
    """Overlay the segmentation mask and board corners on the original image.

    Draws:
      - A semi-transparent green tint over the segmented board region.
      - The board contour outline in green.
      - The four corner points as red dots with TL/TR/BR/BL labels.

    Args:
        image:   Original BGR image.
        mask:    Binary segmentation mask from :func:`board.segment_board`.
        corners: (4, 2) float32 array of board corners [TL, TR, BR, BL].

    Returns:
        Annotated copy of *image*.
    """
    out = image.copy()

    overlay = out.copy()
    overlay[mask == 255] = (overlay[mask == 255] * 0.5 + np.array([0, 120, 0]) * 0.5).clip(0, 255).astype(np.uint8)
    cv2.addWeighted(overlay, 0.4, out, 0.6, 0, out)

    contour_pts = corners.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(out, [contour_pts], isClosed=True, color=(0, 255, 0), thickness=3)

    for label, (x, y) in zip(("TL", "TR", "BR", "BL"), corners):
        cv2.circle(out, (int(x), int(y)), 10, (0, 0, 255), -1)
        cv2.putText(
            out, label,
            (int(x) + 12, int(y) - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA,
        )

    return out


def annotate_grid(
    warped: np.ndarray,
    h_lines: np.ndarray,
    v_lines: np.ndarray,
    color: tuple[int, int, int] = (0, 200, 0),
    thickness: int = 1,
) -> np.ndarray:
    """Overlay detected grid lines on the rectified board image.

    Args:
        warped:    Rectified BGR board image.
        h_lines:   Y-coordinates of horizontal lines (from detect_grid_lines).
        v_lines:   X-coordinates of vertical lines (from detect_grid_lines).
        color:     BGR line color.
        thickness: Line thickness in pixels.

    Returns:
        Annotated copy of *warped*.
    """
    out = warped.copy()
    h, w = out.shape[:2]
    for y in h_lines:
        cv2.line(out, (0, int(y)), (w, int(y)), color, thickness, cv2.LINE_AA)
    for x in v_lines:
        cv2.line(out, (int(x), 0), (int(x), h), color, thickness, cv2.LINE_AA)
    return out
