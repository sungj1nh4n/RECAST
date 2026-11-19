"""RECAST selection from cached positive-match probabilities (§III-C)."""
import math
import torch


def rosin_threshold(scores):
    """Return the histogram corner (§III-C1). See reproduction notes for binning."""
    # [N] -> [100]. Inputs to this helper are validated by select_frames.
    hist = torch.histc(scores.double(), bins=100, min=0, max=1)
    peak = int(hist.argmax())
    tail = int(hist.nonzero()[-1].item())
    if scores.min() == scores.max() or tail <= peak + 1:
        return scores.min()
    x = torch.arange(peak, tail + 1, device=scores.device)
    # [tail-peak+1]. The perpendicular-distance denominator is constant.
    distances = ((tail - peak) * (hist[x] - hist[peak])
                 - (hist[tail] - hist[peak]) * (x - peak)).abs()
    if distances.max() == 0:
        return scores.min()
    return (peak + int(distances.argmax()) + 0.5) / 100


def select_frames(scores, k, relevance, sigma):
    """Return chronological row offsets into a dense 1-FPS score vector (§III-C)."""
    scores = torch.as_tensor(scores).to(dtype=torch.float64)
    if scores.ndim != 1 or not torch.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Expected a one-dimensional vector of probabilities in [0, 1]")
    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise ValueError("k must be a nonnegative integer")
    if not 0 <= relevance <= 1 or not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("Expected relevance in [0, 1] and positive finite sigma")
    n = scores.numel()
    if k == 0 or n <= k:
        return torch.arange(min(k, n), device=scores.device)

    keep = scores >= rosin_threshold(scores)  # [N], §III-C1 inclusive cut
    if int(keep.sum()) < k:
        keep[scores.argsort(descending=True, stable=True)[:k]] = True
    positions = keep.nonzero().flatten()  # [C], original timeline
    values = scores[positions]  # [C]
    choice = values.argmax()
    used = torch.zeros_like(values, dtype=torch.bool)
    redundancy = torch.zeros_like(values)
    for step in range(k):
        used[choice] = True
        if step + 1 < k:
            delta = (positions - positions[choice]).double() / sigma  # [C]
            redundancy = torch.maximum(redundancy, torch.exp(-0.5 * delta.square()))
            objective = relevance * values - (1 - relevance) * redundancy  # Eq. 1, [C]
            choice = objective.masked_fill(used, -torch.inf).argmax()
    return positions[used]  # [k], chronological


def recast(question, options, frames, generate, score, *, k, relevance, sigma):
    """Run §§III-A–C on chronological 1-FPS frames. Return q*, scores, offsets."""
    from .reformulation import reformulate
    description = reformulate(question, options, generate)
    scores = torch.as_tensor(score(frames, description))
    if scores.shape != (len(frames),):
        raise ValueError("The scorer must return one probability per frame")
    return description, scores, select_frames(scores, k, relevance, sigma)
