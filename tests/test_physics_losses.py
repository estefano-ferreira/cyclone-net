"""
Tests for the physics-guided loss wiring.

Guards the central project claim: the model is physics-guided ONLY when the
training.physics.* weights are non-zero. These tests fail if a future refactor
silently disconnects the physics terms again (the exact regression this codebase
already suffered).
"""
import numpy as np
import torch

from src.physics.physics_guided_losses import (
    equation_consistency_loss,
    vort_div_from_uv,
)
from src.training.trainer import _physics_loss


def _batch(B=3, H=12, W=12):
    prior = torch.rand(B, 1, H, W)
    prior = prior / prior.sum(dim=(2, 3), keepdim=True)
    return {
        "x": torch.randn(B, 4, 2, H, W),  # only used for .device
        "dv24": torch.randn(B) * 15.0,
        "dv24_mask": torch.ones(B),
        "prior_map_t0": prior,
        "eq_mask": torch.ones(B),
        "u10_t0": torch.randn(B, 1, H, W),
        "v10_t0": torch.randn(B, 1, H, W),
        "vort_t0": torch.randn(B, 1, H, W) * 1e-4,
        "div_t0": torch.randn(B, 1, H, W) * 1e-4,
        "dx_m": torch.full((B,), 25000.0),
        "dy_m": torch.full((B,), 27000.0),
    }


def _outputs(B=3, H=12, W=12):
    return {
        "fuelmap_logits": torch.randn(B, 1, H, W, requires_grad=True),
        "dv24_forward_hat": torch.randn(B, requires_grad=True),
    }


def _cfg(prior=1.0, fwd=0.5, tv=1e-3, l1=1e-4, cons=0.0):
    return {"training": {"physics": {
        "lambda_prior_align": prior, "lambda_forward": fwd,
        "lambda_tv": tv, "lambda_l1": l1, "lambda_consistency": cons,
    }}}


def test_physics_loss_is_zero_when_all_weights_zero():
    loss = _physics_loss(_cfg(0, 0, 0, 0, 0), _batch(), _outputs())
    assert float(loss) == 0.0


def test_physics_loss_is_positive_and_differentiable_when_enabled():
    out = _outputs()
    loss = _physics_loss(_cfg(), _batch(), out)
    assert float(loss.detach()) > 0.0
    assert loss.requires_grad
    loss.backward()
    assert out["fuelmap_logits"].grad is not None


def test_forward_term_requires_forward_head_output():
    # No dv24_forward_hat -> forward term contributes nothing, only regularizers/prior.
    out = {"fuelmap_logits": torch.randn(3, 1, 12, 12)}
    loss_no_fwd = _physics_loss(_cfg(prior=0, fwd=0.5, tv=0, l1=0, cons=0), _batch(), out)
    assert float(loss_no_fwd) == 0.0


def test_vort_div_from_uv_recovers_solid_body_rotation():
    omega, dx, dy, N = 3e-5, 25000.0, 27000.0, 24
    j = torch.arange(N).float()
    i = torch.arange(N).float()
    X = j.view(1, -1).repeat(N, 1) * dx
    Y = i.view(-1, 1).repeat(1, N) * dy
    xc, yc = X.mean(), Y.mean()
    u = (-omega * (Y - yc)).view(1, 1, N, N)
    v = (omega * (X - xc)).view(1, 1, N, N)
    vort, div = vort_div_from_uv(u, v, dx=dx, dy=dy)
    assert torch.allclose(vort[:, :, 2:-2, 2:-2], torch.full_like(vort[:, :, 2:-2, 2:-2], 2 * omega), atol=1e-9)
    assert torch.allclose(div[:, :, 2:-2, 2:-2], torch.zeros_like(div[:, :, 2:-2, 2:-2]), atol=1e-9)


def test_equation_consistency_is_near_zero_for_self_consistent_fields():
    # If vort/div channels ARE the derivatives of u/v, the loss must be ~0.
    # This is exactly why the term is documented as weak/representational.
    dx, dy, N = 25000.0, 27000.0, 16
    u = torch.randn(2, 1, N, N)
    v = torch.randn(2, 1, N, N)
    vort, div = vort_div_from_uv(u, v, dx=dx, dy=dy)
    loss = equation_consistency_loss(u, v, vort, div, dx=dx, dy=dy)
    assert float(loss) < 1e-6
