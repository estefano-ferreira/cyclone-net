from __future__ import annotations

"""CycloneNet — auxiliary interpretability utilities.

Important scientific note
-------------------------
Integrated gradients in this module are provided only as auxiliary model
inspection tools. They are not part of the released spatial validation claims.
The paper's localisation discussion is based on FuelMap outputs, not on gradient-
based attribution maps.
"""

from typing import Literal

import torch


TargetName = Literal["ri_logit", "dv12", "dv24"]


def _extract_scalar_target(output: object, target: TargetName, batch_index: int) -> torch.Tensor:
    if isinstance(output, dict):
        if target not in output:
            raise KeyError(f"Requested target '{target}' not found in model output keys: {list(output.keys())}")
        value = output[target]
    elif torch.is_tensor(output):
        value = output
    else:
        raise TypeError("Model output must be either a dict or a torch.Tensor.")

    if not torch.is_tensor(value):
        raise TypeError(f"Target '{target}' is not a tensor.")

    if value.ndim == 0:
        return value
    if value.ndim == 1:
        return value[batch_index]
    raise ValueError(
        f"Target '{target}' must be scalar-like per sample. Got tensor with shape {tuple(value.shape)}"
    )


def integrated_gradients(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    baseline: torch.Tensor | None = None,
    steps: int = 50,
    target: TargetName = "ri_logit",
    batch_index: int = 0,
    use_prior_map_t0: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute integrated gradients for a selected scalar model target.

    Parameters
    ----------
    model
        Model returning either a dict with keys such as ``ri_logit``, ``dv12``,
        and ``dv24``, or a tensor directly.
    input_tensor
        Input tensor with shape ``(B, C, T, H, W)``.
    baseline
        Reference tensor. If omitted, a zero baseline is used.
    steps
        Number of line-integral steps.
    target
        Scalar output target to explain.
    batch_index
        Batch element to explain.
    use_prior_map_t0
        Optional prior map passed to models whose forward signature accepts it.

    Returns
    -------
    torch.Tensor
        Integrated gradients tensor with the same shape as ``input_tensor``.
    """
    if input_tensor.ndim != 5:
        raise ValueError(f"Expected input_tensor with shape (B,C,T,H,W), got {tuple(input_tensor.shape)}")
    if steps <= 0:
        raise ValueError("steps must be a positive integer")

    model.eval()

    if baseline is None:
        baseline = torch.zeros_like(input_tensor)
    if baseline.shape != input_tensor.shape:
        raise ValueError(
            f"Baseline shape mismatch: expected {tuple(input_tensor.shape)}, got {tuple(baseline.shape)}"
        )

    grads = []
    for alpha in torch.linspace(0.0, 1.0, steps + 1, device=input_tensor.device, dtype=input_tensor.dtype):
        scaled = baseline + alpha * (input_tensor - baseline)
        scaled = scaled.clone().detach().requires_grad_(True)

        model.zero_grad(set_to_none=True)
        if use_prior_map_t0 is not None:
            output = model(scaled, prior_map_t0=use_prior_map_t0)
        else:
            output = model(scaled)

        scalar = _extract_scalar_target(output, target=target, batch_index=batch_index)
        scalar.backward()

        if scaled.grad is None:
            raise RuntimeError("Integrated gradients failed because no gradient was produced for the input tensor.")
        grads.append(scaled.grad.detach().clone())

    avg_grad = torch.mean(torch.stack(grads, dim=0), dim=0)
    return (input_tensor - baseline) * avg_grad