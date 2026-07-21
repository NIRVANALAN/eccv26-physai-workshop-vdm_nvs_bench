"""Video PSNR metric for the Syn4D NVS Kaggle challenge.

This uses the same tabular metric interface as the Syn4D 3D point-tracking
challenge and performs no file I/O. Video preprocessing is deliberately baked
into the private ``solution.csv`` and each participant ``submission.csv``:

1. standardize the decoded video to 288×512 (height × width) and frames 0–48;
2. apply ``cv2.INTER_AREA`` 8×8 downsampling to a dense 36×64 RGB grid;
3. retain frames ``0, 8, 16, 24, 32, 40, 48``.

The hidden ``solution`` dataframe has columns ``sequence, valid, R, G, B``.
Each row represents one sampled-grid RGB pixel at one retained frame. Pixels
are enumerated row-major as ``q = y * 64 + x`` and row ids look like
``og-antiquity-seq_000000_0-q0042-f008``. Thus every sequence has
``7 × 36 × 64 = 16,128`` rows. This is a dense 36×64 grid, not the tracking
challenge's 512-query-point sampling.

The leaderboard metric is mean PSNR (dB, capped) over valid pixels in each
sequence, then averaged over sequences.

Run the tests with: ``python -m doctest metric_video_psnr.py``
"""

import numpy as np
import pandas as pd
import pandas.api.types

# Pixel value range is auto-detected per call from the ground-truth data
# itself (see _infer_max_pixel_value), so the metric accepts either [0, 1]
# floats or [0, 255] RGB values. The official solution uses [0, 255]. Set this
# to a float (for example 255.0) to override automatic detection.
MAX_PIXEL_VALUE = None

_AUTO_DETECT_THRESHOLD = 1.5
PSNR_CAP = 60.0
_MSE_EPS = 1e-10


class ParticipantVisibleError(Exception):
    # Kaggle displays only this exception type to participants, preventing
    # accidental leakage of private solution data through host-side errors.
    pass


def _infer_max_pixel_value(all_gt: np.ndarray) -> float:
    """Infer whether colors are [0, 1] or [0, 255] scaled from GT alone."""
    if MAX_PIXEL_VALUE is not None:
        return MAX_PIXEL_VALUE
    max_val = float(np.max(all_gt)) if all_gt.size else 0.0
    return 1.0 if max_val <= _AUTO_DETECT_THRESHOLD else 255.0


def _sequence_psnr(gt: np.ndarray, pred: np.ndarray, max_pixel_value: float) -> float:
    """PSNR (dB, capped) for one sequence's sampled pixel/frame rows.

    ``gt`` and ``pred`` are [M, 3] float arrays over valid rows only.
    """
    mse = float(np.mean((gt - pred) ** 2))
    if mse < _MSE_EPS:
        return PSNR_CAP
    psnr = 10.0 * np.log10((max_pixel_value ** 2) / mse)
    return min(psnr, PSNR_CAP)


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """Compute the official sampled-video PSNR leaderboard score.

    Rows already encode the 288×512 → 36×64 (8×8 ``INTER_AREA``) and
    49-frame → 7-frame (every eighth frame) preprocessing described above.
    The callback compares the submission's ``R, G, B`` values with the hidden
    solution; it does not decode or resize videos.

    Scoring is macro-averaged across sequences:

    1. Ignore rows for which the hidden ``valid`` flag is zero.
    2. Compute capped PSNR over all remaining sampled pixels and frames in a
       sequence.
    3. Return the arithmetic mean of the per-sequence PSNR values.

    Examples
    --------
    >>> import pandas as pd
    >>> sol = pd.DataFrame({
    ...     'id': range(4),
    ...     'sequence': ['s0'] * 4,
    ...     'valid': [1, 1, 1, 1],
    ...     'R': [10.0, 20.0, 30.0, 40.0],
    ...     'G': [50.0, 60.0, 70.0, 80.0],
    ...     'B': [90.0, 100.0, 110.0, 120.0],
    ... })
    >>> perfect = sol[['id', 'R', 'G', 'B']].copy()
    >>> score(sol.copy(), perfect.copy(), 'id')
    60.0

    A small uniform color error gives a finite, lower PSNR:

    >>> noisy = perfect.copy()
    >>> noisy[['R', 'G', 'B']] += 10.0
    >>> round(score(sol.copy(), noisy.copy(), 'id'), 2)
    28.13

    Rows with ``valid == 0`` are ignored:

    >>> sol2 = sol.copy(); sol2.loc[3, 'valid'] = 0
    >>> junk = perfect.copy(); junk.loc[3, ['R', 'G', 'B']] = 9e9
    >>> score(sol2.copy(), junk.copy(), 'id')
    60.0
    """
    del solution[row_id_column_name]
    del submission[row_id_column_name]

    color_cols = ["R", "G", "B"]
    missing = [column for column in color_cols if column not in submission.columns]
    if missing:
        raise ParticipantVisibleError(f"Submission is missing required column(s): {missing}")
    if len(submission) != len(solution):
        raise ParticipantVisibleError(
            f"Submission has {len(submission)} rows; expected {len(solution)}"
        )
    for column in color_cols:
        if not pandas.api.types.is_numeric_dtype(submission[column]):
            raise ParticipantVisibleError(f"Submission column {column} must be a number")
    pred = submission[color_cols].to_numpy(dtype=np.float64)
    if not np.isfinite(pred).all():
        raise ParticipantVisibleError("Submission contains NaN or infinite values")

    gt = solution[color_cols].to_numpy(dtype=np.float64)
    valid = solution["valid"].to_numpy() == 1
    sequences = solution["sequence"].to_numpy()
    max_pixel_value = _infer_max_pixel_value(gt[valid])

    sequence_scores = []
    for sequence in pd.unique(sequences):
        mask = (sequences == sequence) & valid
        if not mask.any():
            continue
        sequence_scores.append(_sequence_psnr(gt[mask], pred[mask], max_pixel_value))
    if not sequence_scores:
        raise ParticipantVisibleError("No scorable rows found")
    return float(np.mean(sequence_scores))
