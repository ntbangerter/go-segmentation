"""board.py - Go board segmentation and perspective rectification."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class DetectedBoard:
    """Result of a board detection pass."""
    warped: np.ndarray     # Rectified board image (output_size × output_size)
    homography: np.ndarray # 3×3 homography mapping original → warped space
    corners: np.ndarray    # (4, 2) float32 corner points in the original image
    mask: np.ndarray       # Binary segmentation mask (255=board, 0=background)


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def segment_board(
    image: np.ndarray,
    blur_ksize: int = 11,
    close_ksize: int = 61,
    open_ksize: int = 31,
) -> np.ndarray:
    """Segment the board from the background using color-based thresholding.

    Uses the LAB color space 'b' channel (blue-yellow axis), which gives strong
    contrast between warm wood tones (high b) and cool gray/green backgrounds
    (low b).  Otsu's method finds the threshold automatically.

    After thresholding, a morphological close fills the dark holes left by
    black stones, and an open removes small noise blobs outside the board.

    Args:
        image:       BGR input image.
        blur_ksize:  Gaussian blur kernel size (must be odd).
        close_ksize: Morphological close kernel size.
        open_ksize:  Morphological open kernel size.

    Returns:
        Single-channel binary mask: 255 inside the board, 0 outside.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    b_channel = lab[:, :, 2]  # blue-yellow axis: warm wood >> cool background
    blurred = cv2.GaussianBlur(b_channel, (blur_ksize, blur_ksize), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (close_ksize, close_ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)

    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (open_ksize, open_ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)

    return mask


# ---------------------------------------------------------------------------
# Corner finding
# ---------------------------------------------------------------------------

def find_board_corners(mask: np.ndarray) -> np.ndarray:
    """Find the four corners of the board from a binary segmentation mask.

    Takes the largest contour (the board), computes its convex hull, then
    simplifies it to exactly four vertices via ``approxPolyDP``.  If the
    polygon simplification never lands on four points, the four extremal hull
    vertices (min/max of x+y and x-y) are used as a fallback.

    Args:
        mask: Binary mask as returned by :func:`segment_board`.

    Returns:
        (4, 2) float32 array ordered [top-left, top-right, bottom-right,
        bottom-left].

    Raises:
        ValueError: If no contours are found in the mask.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in segmentation mask.")

    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    peri = cv2.arcLength(hull, True)

    # Increase epsilon until approxPolyDP gives exactly 4 vertices.
    for eps_factor in [0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20]:
        approx = cv2.approxPolyDP(hull, eps_factor * peri, True)
        if len(approx) == 4:
            return _order_corners(approx.reshape(4, 2).astype(np.float32))

    # Fallback: four extremal hull points.
    pts = hull.reshape(-1, 2).astype(np.float32)
    corners = np.array([
        pts[np.argmin(pts[:, 0] + pts[:, 1])],  # top-left:     min x+y
        pts[np.argmax(pts[:, 0] - pts[:, 1])],  # top-right:    max x-y
        pts[np.argmax(pts[:, 0] + pts[:, 1])],  # bottom-right: max x+y
        pts[np.argmin(pts[:, 0] - pts[:, 1])],  # bottom-left:  min x-y
    ], dtype=np.float32)
    return corners


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order four points as [top-left, top-right, bottom-right, bottom-left]."""
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]          # TL: smallest x+y
    rect[2] = pts[np.argmax(s)]          # BR: largest  x+y
    diff = pts[:, 1] - pts[:, 0]         # y - x
    rect[1] = pts[np.argmin(diff)]       # TR: smallest y-x
    rect[3] = pts[np.argmax(diff)]       # BL: largest  y-x
    return rect


# ---------------------------------------------------------------------------
# Rectification
# ---------------------------------------------------------------------------

def rectify_board(
    image: np.ndarray,
    corners: np.ndarray,
    output_size: int = 800,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp *image* so that *corners* maps to a square of side *output_size*.

    Args:
        image:       Original BGR (or grayscale) image.
        corners:     (4, 2) float32 array of board corners (any order).
        output_size: Side length in pixels of the output square.

    Returns:
        (warped, H): The rectified image and the 3×3 homography matrix.
    """
    src = _order_corners(corners)
    dst = np.array([
        [0,               0],
        [output_size - 1, 0],
        [output_size - 1, output_size - 1],
        [0,               output_size - 1],
    ], dtype=np.float32)
    H, _ = cv2.findHomography(src, dst)
    warped = cv2.warpPerspective(image, H, (output_size, output_size))
    return warped, H


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def detect_board(
    image: np.ndarray,
    output_size: int = 800,
    blur_ksize: int = 11,
    close_ksize: int = 61,
    open_ksize: int = 31,
) -> DetectedBoard:
    """Detect a Go board in *image* and return a perspective-corrected view.

    Pipeline:
        1. Segment the board from the background via Otsu threshold +
           morphological close (fills stone holes) + open (removes noise).
        2. Find the four board corners from the largest contour.
        3. Apply a perspective homography to produce a square output image.

    Args:
        image:       BGR image as returned by ``cv2.imread``.
        output_size: Side length in pixels for the rectified output image.
        blur_ksize:  Gaussian blur kernel size applied before thresholding.
        close_ksize: Morphological close kernel size (should exceed stone size).
        open_ksize:  Morphological open kernel size (noise removal).

    Returns:
        :class:`DetectedBoard` with the warped image, homography, corner
        points, and segmentation mask.

    Raises:
        ValueError: Propagated from :func:`find_board_corners` if no board
        contour is found.
    """
    # Scale morphological kernels proportionally to image resolution.
    # Default values were tuned for ~1400px wide images.
    scale = min(image.shape[:2]) / 1400
    def _scaled_odd(k: int) -> int:
        v = max(3, round(k * scale))
        return v if v % 2 == 1 else v + 1

    mask = segment_board(
        image,
        _scaled_odd(blur_ksize),
        _scaled_odd(close_ksize),
        _scaled_odd(open_ksize),
    )
    corners = find_board_corners(mask)
    warped, H = rectify_board(image, corners, output_size)
    return DetectedBoard(warped=warped, homography=H, corners=corners, mask=mask)
