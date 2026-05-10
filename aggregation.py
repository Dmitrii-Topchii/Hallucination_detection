from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _real_positions(attention_mask: torch.Tensor, device: torch.device) -> torch.Tensor:
    positions = torch.nonzero(attention_mask > 0, as_tuple=False).flatten()
    if positions.numel() == 0:
        positions = torch.arange(attention_mask.numel())
    return positions.to(device=device, dtype=torch.long)


def _layer_indices(n_layers: int) -> list[int]:
    raw = [
        max(1, n_layers // 3),
        max(1, n_layers // 2),
        max(1, (2 * n_layers) // 3),
        max(1, n_layers - 5),
        max(1, n_layers - 3),
        n_layers - 1,
    ]
    return sorted(dict.fromkeys(min(n_layers - 1, i) for i in raw))


def _tail_positions(positions: torch.Tensor, size: int) -> torch.Tensor:
    return positions[-min(size, int(positions.numel())) :]


def _cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a, b, dim=0, eps=1e-6)


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    positions = _real_positions(attention_mask, hidden_states.device)
    layers = _layer_indices(hidden_states.shape[0])
    last_pos = int(positions[-1].item())
    tail32 = _tail_positions(positions, 32)
    tail8 = _tail_positions(positions, 8)
    dim_scale = math.sqrt(float(hidden_states.shape[-1]))
    seq_scale = torch.tensor(
        [
            float(positions.numel()) / 512.0,
            math.log1p(float(positions.numel())) / math.log1p(512.0),
            float(tail32.numel()) / 32.0,
            float(tail8.numel()) / 8.0,
        ],
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    values = [seq_scale]
    for layer in layers:
        tokens = hidden_states[layer, positions, :]
        tail_tokens = hidden_states[layer, tail32, :]
        last = hidden_states[layer, last_pos, :]
        mean = tokens.mean(dim=0)
        tail_mean = tail_tokens.mean(dim=0)
        stats = torch.stack(
            [
                last.norm() / dim_scale,
                mean.norm() / dim_scale,
                tail_mean.norm() / dim_scale,
                tokens.var(dim=0, unbiased=False).mean().sqrt(),
                tail_tokens.var(dim=0, unbiased=False).mean().sqrt(),
                _cosine(last, mean),
                _cosine(last, tail_mean),
            ]
        )
        values.append(stats)
    for left, right in zip(layers[1:], layers[:-1]):
        a = hidden_states[left, last_pos, :]
        b = hidden_states[right, last_pos, :]
        values.append(
            torch.stack(
                [
                    (a - b).norm() / dim_scale,
                    _cosine(a, b),
                ]
            )
        )
    return torch.cat(values).float()


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    positions = _real_positions(attention_mask, hidden_states.device)
    layers = _layer_indices(hidden_states.shape[0])
    last_pos = int(positions[-1].item())
    tail32 = _tail_positions(positions, 32)
    tail8 = _tail_positions(positions, 8)
    tail16 = _tail_positions(positions, 16)
    late_layers = layers[-3:]
    last_vectors = hidden_states[layers][:, last_pos, :].reshape(-1)
    all_mean = hidden_states[late_layers][:, positions, :].mean(dim=1).reshape(-1)
    tail32_mean = hidden_states[late_layers][:, tail32, :].mean(dim=1).reshape(-1)
    tail8_mean = hidden_states[late_layers][:, tail8, :].mean(dim=1).reshape(-1)
    weights = torch.linspace(
        1.0,
        2.0,
        steps=int(tail16.numel()),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    ).view(1, -1, 1)
    weighted_tail16 = (
        hidden_states[late_layers][:, tail16, :] * weights
    ).sum(dim=1).reshape(-1) / weights.sum()
    final_last = hidden_states[-1, last_pos, :]
    final_all_mean = hidden_states[-1, positions, :].mean(dim=0)
    final_tail_mean = hidden_states[-1, tail32, :].mean(dim=0)
    deltas = []
    first = layers[0]
    middle = layers[min(1, len(layers) - 1)]
    previous = layers[-2] if len(layers) > 1 else layers[-1]
    last = layers[-1]
    for left, right in [(last, previous), (last, first), (previous, middle)]:
        deltas.append(hidden_states[left, last_pos, :] - hidden_states[right, last_pos, :])
    deltas.extend([final_last - final_all_mean, final_last - final_tail_mean])
    return torch.cat(
        [
            last_vectors,
            all_mean,
            tail32_mean,
            tail8_mean,
            weighted_tail16,
            torch.cat(deltas),
            extract_geometric_features(hidden_states, attention_mask),
        ],
        dim=0,
    ).float()


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    features = aggregate(hidden_states, attention_mask)
    if use_geometric:
        features = torch.cat(
            [features, extract_geometric_features(hidden_states, attention_mask)],
            dim=0,
        )
    return features
