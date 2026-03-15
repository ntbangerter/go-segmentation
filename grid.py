"""grid.py - Grid line detection for rectified Go board images."""

import numpy as np


def detect_grid_lines(
    warped,
    n: int = 19,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect the n horizontal and n vertical grid lines in a rectified board image.

    Computes row-wise and column-wise mean intensity projections on the inverted
    grayscale image (dark grid lines become bright peaks), then fits a regular
    grid via NMS peak detection and linear regression.

    Args:
        warped: Rectified BGR board image (square, as from rectify_board).
        n:      Lines per direction (19 for a standard Go board).

    Returns:
        (h_lines, v_lines): int32 arrays of y- and x-coordinates, sorted ascending.
    """
    import cv2
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    inv = 255.0 - gray.astype(np.float32)

    h_lines = _detect_linreg(inv.mean(axis=1), n)
    v_lines = _detect_linreg(inv.mean(axis=0), n)
    return h_lines, v_lines


def _detect_linreg(profile: np.ndarray, n: int) -> np.ndarray:
    """Detect n grid line positions in a 1D projection profile.

    1. Estimate the line spacing (period) from NMS peak median spacing.
    2. Collect NMS peaks and assign each one a grid index based on period.
    3. Fit a line (position = a + b * index) through the (index, peak) pairs.
    4. Project onto n positions from the fitted line.

    Falls back to a comb sweep if too few peaks are found.
    """
    N = len(profile)
    k = max(3, N // 200)  # narrow kernel → sharper peaks
    smoothed = np.convolve(profile, np.ones(k) / k, mode="same")
    norm = smoothed - smoothed.min()

    period = _estimate_period(smoothed, norm, n, N)
    min_dist = max(3, int(period * 0.4))
    nms = np.sort(_nms_peaks(smoothed, n + 6, min_dist=min_dist))

    if len(nms) < 3:
        return _detect_comb(profile, n)

    # Assign each peak a grid index using a median-based anchor.
    anchor = float(np.median(nms)) - (n // 2) * period
    indices = np.round((nms - anchor) / period).astype(int)

    mask = (indices >= 0) & (indices < n)
    nms, indices = nms[mask], indices[mask]

    # If two peaks share an index, keep the one closest to the expected position.
    unique_idx, unique_pos = [], []
    for idx in np.unique(indices):
        candidates = nms[indices == idx]
        expected = anchor + idx * period
        best = candidates[np.argmin(np.abs(candidates - expected))]
        unique_idx.append(idx)
        unique_pos.append(best)

    if len(unique_idx) < 3:
        return _detect_comb(profile, n)

    b, a = np.polyfit(np.array(unique_idx, dtype=float), np.array(unique_pos, dtype=float), 1)

    # Shift start index so no positions fall below zero.
    i_start = int(np.ceil(-a / b)) if (b > 0 and a < 0) else 0
    positions = np.round(a + b * (i_start + np.arange(n))).astype(int)
    return np.clip(positions, 0, N - 1).astype(np.int32)


def _detect_comb(profile: np.ndarray, n: int) -> np.ndarray:
    """Fallback: fit an equally-spaced comb by sweeping all possible offsets."""
    N = len(profile)
    k = max(3, N // 80)
    smoothed = np.convolve(profile, np.ones(k) / k, mode="same")
    norm = smoothed - smoothed.min()

    period = _estimate_period(smoothed, norm, n, N)

    best_score, best_offset = -1.0, 0
    for offset in range(int(np.ceil(period))):
        positions = np.round(offset + np.arange(n) * period).astype(int)
        valid = positions[(positions >= 0) & (positions < N)]
        if len(valid) < n - 1:
            continue
        score = float(norm[valid].sum())
        if score > best_score:
            best_score = score
            best_offset = offset

    positions = np.round(best_offset + np.arange(n) * period).astype(int)
    return np.clip(positions, 0, N - 1).astype(np.int32)


def _nms_peaks(smoothed: np.ndarray, max_peaks: int, min_dist: int) -> np.ndarray:
    """Return up to max_peaks local maxima via greedy non-maximum suppression."""
    candidates = [
        (smoothed[i], i)
        for i in range(1, len(smoothed) - 1)
        if smoothed[i] > smoothed[i - 1] and smoothed[i] > smoothed[i + 1]
    ]
    candidates.sort(reverse=True)
    selected: list[int] = []
    for _, idx in candidates:
        if all(abs(idx - s) >= min_dist for s in selected):
            selected.append(idx)
        if len(selected) == max_peaks:
            break
    return np.array(selected, dtype=np.int32)


def _estimate_period(smoothed: np.ndarray, norm: np.ndarray, n: int, N: int) -> float:
    """Estimate grid line spacing from NMS median spacing, with fallbacks."""
    nms = _nms_peaks(smoothed, n + 4, min_dist=N // (n * 2))
    if len(nms) >= 3:
        period = float(np.median(np.diff(np.sort(nms))))
    else:
        period = _autocorr_period(norm, n)

    if not (N * 0.5 < period * (n - 1) < N * 0.95):
        period = _autocorr_period(norm, n)
    if not (N * 0.5 < period * (n - 1) < N * 0.95):
        period = N / n

    return period


def _autocorr_period(norm: np.ndarray, n: int) -> float:
    """Estimate line spacing from the autocorrelation of the projection."""
    N = len(norm)
    centered = norm - norm.mean()
    autocorr = np.correlate(centered, centered, mode="full")[N - 1 :]
    min_p = max(5, int(N * 0.6 / n))
    max_p = min(N // 2, int(N * 1.4 / n))
    return float(min_p + int(np.argmax(autocorr[min_p : max_p + 1])))
