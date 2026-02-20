"""Interpretability utilities (integrated gradients)."""

import torch


def integrated_gradients(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    baseline: torch.Tensor = None,
    steps: int = 50
) -> torch.Tensor:
    """Compute integrated gradients for the given input and model."""
    if baseline is None:
        baseline = torch.zeros_like(input_tensor)

    scaled_inputs = [
        baseline + (i / steps) * (input_tensor - baseline)
        for i in range(steps + 1)
    ]

    grads = []
    for scaled_input in scaled_inputs:
        scaled_input = scaled_input.clone().detach().requires_grad_(True)
        logit = model(scaled_input)
        if logit.dim() == 0:
            logit = logit.unsqueeze(0)
        model.zero_grad()
        logit.backward()
        grad = scaled_input.grad.clone()
        grads.append(grad)

    avg_grad = torch.mean(torch.stack(grads), dim=0)
    ig = (input_tensor - baseline) * avg_grad
    return ig
