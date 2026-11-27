# GPU declicker using GrooveClean detection and autoregressive reconstruction
import numpy as np
import torch
import torch.nn.functional as F
from grooveclean.detect import Detector

AR_ORDER = 16
CHUNK_SIZE = 256
RIDGE = 1e-3

device = "cuda" if torch.cuda.is_available() else "cpu"

detector = None


def get_detector():
    global detector

    if detector is None:
        detector = Detector.load(device=device)

    return detector


def predict(context, valid, steps):

    order = AR_ORDER

    # Build autoregressive training samples
    windows = context.unfold(1, order + 1, 1)
    valid_windows = valid.unfold(1, order + 1, 1).all(dim=-1)

    X = windows[:, :, :-1]
    y = windows[:, :, -1:]

    X = X * valid_windows.unsqueeze(-1)
    y = y * valid_windows.unsqueeze(-1)

    Xt = X.transpose(1, 2)

    A = Xt @ X
    b = Xt @ y

    scale = A.diagonal(dim1=1, dim2=2).mean(dim=1)
    scale = scale.clamp_min(1e-8)

    identity = torch.eye(order, device=context.device, dtype=context.dtype)

    A = A + RIDGE * scale[:, None, None] * identity

    coefficients, info = torch.linalg.solve_ex(A, b)
    coefficients = coefficients.squeeze(-1)

    usable = (
        (info == 0)
        & (valid_windows.sum(dim=1) >= order * 2)
        & valid[:, -order:].all(dim=1)
        & torch.isfinite(coefficients).all(dim=1)
    )

    coefficients = torch.nan_to_num(coefficients, nan=0.0, posinf=0.0, neginf=0.0)

    state = context[:, -order:].clone()

    limit = torch.where(valid, context.abs(), 0.0).amax(dim=1) * 1.5
    limit = limit.clamp_min(1e-6)

    predictions = []

    for _ in range(steps):

        sample = (state * coefficients).sum(dim=1)

        sample = torch.maximum(torch.minimum(sample, limit), -limit)

        predictions.append(sample)

        state = torch.cat([state[:, 1:], sample.unsqueeze(1)], dim=1)

    return torch.stack(predictions, dim=1), usable


@torch.no_grad()
def reconstruct(waveforms, sample_rate, strength, mask):

    device = waveforms.device
    batch_size, length = waveforms.shape

    # Scale reconstruction parameters with sample rate
    context = max(AR_ORDER * 3, round(256 * sample_rate / 22050))
    max_gap = max(1, round(220 * sample_rate / 22050))
    expansion = 2 * round(2 * sample_rate / 22050) + 1

    mask = F.max_pool1d(
        mask.float().unsqueeze(1),
        kernel_size=expansion,
        stride=1,
        padding=expansion // 2
    ).squeeze(1).bool()

    mask[:, 0] = False
    mask[:, -1] = False

    # Identify damaged regions
    previous = F.pad(mask[:, :-1], (1, 0), value=False)
    following = F.pad(mask[:, 1:], (0, 1), value=False)

    starts = torch.nonzero(mask & ~previous)
    ends = torch.nonzero(mask & ~following)

    if len(starts) == 0:
        return waveforms.clone()

    gap_lengths = ends[:, 1] - starts[:, 1] + 1

    keep = gap_lengths <= max_gap

    starts = starts[keep]
    ends = ends[keep]
    gap_lengths = gap_lengths[keep]

    restored = waveforms.clone()

    offsets = torch.arange(context, device=device)

    # Process damaged regions in chunks
    for i in range(0, len(starts), CHUNK_SIZE):

        start = starts[i:i + CHUNK_SIZE]
        end = ends[i:i + CHUNK_SIZE]
        gaps = gap_lengths[i:i + CHUNK_SIZE]

        batch = start[:, 0]
        first = start[:, 1]
        last = end[:, 1]

        steps = int(gaps.max().item())

        # Context before the damaged region
        left_indices = first[:, None] - context + offsets

        left_valid = left_indices >= 0
        left_indices = left_indices.clamp(0, length - 1)

        left_context = waveforms[batch[:, None], left_indices]

        left_valid = left_valid & ~mask[batch[:, None], left_indices]

        # Context after the damaged region, reversed
        right_indices = last[:, None] + context - offsets

        right_valid = right_indices < length
        right_indices = right_indices.clamp(0, length - 1)

        right_context = waveforms[batch[:, None], right_indices]

        right_valid = right_valid & ~mask[batch[:, None], right_indices]

        forward, forward_valid = predict(left_context, left_valid, steps)

        backward, backward_valid = predict(right_context, right_valid, steps)

        positions = torch.arange(steps, device=device).unsqueeze(0).expand(len(start), -1)

        reverse_positions = (gaps[:, None] - 1 - positions).clamp_min(0)

        backward = backward.gather(1, reverse_positions)

        fraction = (positions + 1) / (gaps[:, None] + 1)

        # Blend forward and backward predictions
        prediction = forward * (1 - fraction) + backward * fraction

        # Linear interpolation fallback
        left_value = waveforms[batch, first - 1]
        right_value = waveforms[batch, last + 1]

        linear = (
            left_value[:, None] * (1 - fraction)
            + right_value[:, None] * fraction
        )

        prediction = torch.where(
            forward_valid[:, None] & ~backward_valid[:, None],
            forward,
            prediction
        )

        prediction = torch.where(
            backward_valid[:, None] & ~forward_valid[:, None],
            backward,
            prediction
        )

        prediction = torch.where(
            ~forward_valid[:, None] & ~backward_valid[:, None],
            linear,
            prediction
        )

        valid_positions = positions < gaps[:, None]

        output_indices = first[:, None] + positions

        batch_indices = batch[:, None].expand_as(output_indices)

        restored[
            batch_indices[valid_positions],
            output_indices[valid_positions]
        ] = prediction[valid_positions]

    return waveforms + strength * (restored - waveforms)


@torch.no_grad()
def batch_gpu_declick(waveforms, sample_rate, actions):

    waveforms = torch.as_tensor(waveforms, dtype=torch.float32, device=device)

    actions = torch.as_tensor(actions, dtype=torch.float32, device=device).clamp(0, 1)

    strength = actions[:, 0:1]
    sensitivities = actions[:, 1].cpu().tolist()

    mask = torch.zeros_like(waveforms, dtype=torch.bool)

    click_detector = get_detector()

    # Detect clicks using GrooveClean
    for i, sensitivity in enumerate(sensitivities):

        events = click_detector.events(
            waveforms[i:i + 1],
            sample_rate,
            sensitivity=sensitivity
        )

        starts, ends, confidence = events[0]

        for start, end in zip(starts, ends):
            mask[i, int(start):int(end)] = True

    return reconstruct(
        waveforms,
        sample_rate,
        strength,
        mask
    )


@torch.no_grad()
def apply(waveform, sample_rate, sensitivity=1.0, strength=1.0):

    actions = np.array([[strength, sensitivity]])

    restored = batch_gpu_declick(
        np.asarray(waveform)[None, :],
        sample_rate,
        actions
    )

    return restored[0].cpu().numpy()