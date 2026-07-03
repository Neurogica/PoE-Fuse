"""Unbalanced entropic optimal-transport correspondence (the headline novelty).

``SinkhornMatch`` replaces the single soft-cross-attention ``_temporal_match`` in
:class:`~poe_fuse.fusion.SpatialChangeAgreementFusion` with a
**mass-conserving** transport plan between the two aligned frames on a *learned
ground metric*.  Unlike row-softmax attention (which lets one t0 cell explain
arbitrarily many t1 cells), the (approximately doubly-stochastic) transport plan
distinguishes content that *moved/persisted* from content that *appeared/
disappeared*:

* ``matched``  : ``N * P @ a0`` -- the t0 content transported to each t1 cell, so
  ``a1 - matched`` is a correspondence-aware change (handles displacement);
* ``tau``      : the per-cell Kantorovich transport cost ``sum_j P_ij C_ij`` --
  the **transport-residual change field**.  LOW tau = corroborated/transported =
  pseudo-change (registration / illumination / season); used to suppress;
* ``unmatched``: the row marginal deficit under **unbalanced** OT -- mass at a t1
  cell that found no t0 match = *appeared* (and symmetrically, *disappeared*).
  This term is the real-change signal and is **only non-zero under unbalanced OT**
  (balanced OT with uniform marginals is mass-conserving in both directions and
  structurally cannot represent appearance/destruction).

Design notes:
* ``iters=0`` reduces EXACTLY to row-softmax attention with score ``-C/eps`` (the
  ablation that recovers the incumbent matcher); the column marginal + unmatched
  mass are the genuine added structure.
* ``rho -> inf`` recovers *balanced* OT (the ablation that kills the appearance
  term); unbalanced (finite rho) is the default and the core of the novelty.
* eps / gamma / rho are learned positive scalars (softplus), warm-started from
  config so the operator is calibratable.
* Sinkhorn iterations run in float32 for stability, regardless of cached dtype.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _inv_softplus(y: float) -> float:
    """x such that softplus(x) == y (so a param warm-starts at value ``y``)."""
    y = max(float(y), 1e-4)
    return math.log(math.expm1(y)) if y < 20 else y


class SinkhornMatch(nn.Module):
    """Unbalanced entropic-OT matcher between two ``(B, N=G*G, d)`` grids."""

    def __init__(
        self,
        d_model: int,
        *,
        grid: int,
        iters: int = 4,
        eps_init: float = 0.1,
        rho_init: float = 4.0,
        gamma_init: float = 0.1,
        decouple_lam: bool = False,
        lam_init: float = 0.7,
    ):
        super().__init__()
        self.G = int(grid)
        self.N = self.G * self.G
        self.iters = int(iters)
        # Content-dependent ground cost (the fix for the static-query resampler).
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        # Learned positive scalars (softplus): entropy eps, position-cost weight
        # gamma, unbalanced marginal weight rho.
        self.raw_eps = nn.Parameter(torch.tensor(_inv_softplus(eps_init)))
        self.raw_gamma = nn.Parameter(torch.tensor(_inv_softplus(gamma_init)))
        self.raw_rho = nn.Parameter(torch.tensor(_inv_softplus(rho_init)))
        # Sinkhorn damping lam in (0,1): ->1 balanced, ->0 no transport.  When
        # coupled (legacy) lam = rho/(rho+eps), which at moderate rho pins lam~=1
        # (balanced) so the marginal *deficit* collapses to ~0 -- this is exactly
        # why v10b's "unmatched mass" never appeared.  decouple_lam learns lam
        # directly (sigmoid) so partial transport / mass creation is reachable.
        self.decouple_lam = bool(decouple_lam)
        lam0 = min(max(float(lam_init), 1e-3), 1 - 1e-3)
        self.raw_lam = nn.Parameter(torch.tensor(math.log(lam0 / (1 - lam0))))
        # Normalised grid coordinates -> squared-distance locality prior.
        ys, xs = torch.meshgrid(torch.arange(self.G), torch.arange(self.G), indexing="ij")
        coords = torch.stack([ys.reshape(-1), xs.reshape(-1)], dim=-1).float()
        coords = coords / max(1, self.G - 1)
        self.register_buffer("coords", coords, persistent=False)  # (N, 2)

    # ---- shared entropic-OT plan -----------------------------------------
    def _plan(self, a_q: Tensor, a_k: Tensor):
        """Return ``(P, C, row, col)`` for the (unbalanced) entropic-OT plan
        transporting ``a_q`` mass onto ``a_k`` on the shared grid.  All in fp32."""
        B, N, _d = a_q.shape
        eps = F.softplus(self.raw_eps) + 1e-4
        gamma = F.softplus(self.raw_gamma)
        q = F.normalize(self.q_proj(a_q), dim=-1).float()
        k = F.normalize(self.k_proj(a_k), dim=-1).float()
        cos = torch.bmm(q, k.transpose(1, 2))  # (B,N,N) in [-1,1]
        d2 = torch.cdist(self.coords[None], self.coords[None]).pow(2)  # (1,N,N)
        C = (1.0 - cos) + gamma.float() * d2  # ground cost (B,N,N)
        K = -C / eps.float()  # log-kernel
        log_marg = -math.log(N)
        if self.decouple_lam:
            lam = torch.sigmoid(self.raw_lam).float()
        else:
            rho = F.softplus(self.raw_rho) + 1e-4
            lam = (rho / (rho + eps)).float()
        f = torch.zeros(B, N, device=a_q.device, dtype=torch.float32)
        g = torch.zeros(B, N, device=a_q.device, dtype=torch.float32)
        for _ in range(self.iters):
            f = lam * (log_marg - torch.logsumexp(g[:, None, :] + K, dim=2))
            g = lam * (log_marg - torch.logsumexp(f[:, :, None] + K, dim=1))
        logP = f[:, :, None] + g[:, None, :] + K  # (B,N,N)
        P = logP.exp()
        row = P.sum(dim=2, keepdim=True)  # (B,N,1) mass leaving q_i
        col = P.sum(dim=1, keepdim=True).transpose(1, 2)  # (B,N,1) mass into k_j
        return P, C, row, col

    def forward_cost(self, a_q: Tensor, a_k: Tensor) -> tuple[Tensor, Tensor]:
        """Variant A: per-cell transport COST tau and row-entropy H (no matched).

        ``tau_j`` LOW = content cheaply transported from a nearby t0 cell
        (registration / illumination pseudo-change); HIGH = no cheap
        correspondent = genuine appearance.  ``H_j`` (normalised row entropy)
        disambiguates a confident sharp match from a degenerate diffuse plan.
        Crucially never forms the barycentric ``matched`` (no blur)."""
        in_dtype = a_q.dtype
        P, C, row, _col = self._plan(a_q, a_k)
        tau = (P * C).sum(dim=2, keepdim=True) / (row + 1e-8)  # (B,N,1)
        Pbar = P / (row + 1e-8)
        H = -(Pbar * Pbar.clamp_min(1e-12).log()).sum(dim=2, keepdim=True)
        H = H / math.log(P.shape[1])  # normalise to [0,1]
        return tau.to(in_dtype), H.to(in_dtype)

    def forward_marginals(self, a0: Tensor, a1: Tensor) -> Tensor:
        """Variant B: bidirectional unbalanced-OT mass deficits as (B,N,2).

        Plan transports t1 mass onto t0.  ``appeared_i`` = relu(1/N - row_i)*N
        (t1 cell with no cheap t0 correspondent) ; ``disappeared_j`` =
        relu(1/N - col_j)*N (t0 cell nothing matched).  Returns
        ``[appeared+disappeared, appeared-disappeared]``."""
        in_dtype = a0.dtype
        N = a1.shape[1]
        _P, _C, row, col = self._plan(a1, a0)
        appeared = (1.0 / N - row).clamp_min(0.0) * N  # (B,N,1)
        disappeared = (1.0 / N - col).clamp_min(0.0) * N  # (B,N,1)
        out = torch.cat([appeared + disappeared, appeared - disappeared], dim=-1)
        return out.to(in_dtype)

    def readout(self, a0: Tensor, a1: Tensor):
        """v12 head-aware readout: ONE plan, return the 4 per-cell functionals
        ``(appeared, disappeared, tau, H)`` each ``(B,N,1)``.

        ``appeared``/``disappeared`` = unbalanced-OT row/col mass deficits
        (created / destroyed content); ``tau`` = per-cell transport cost
        (LOW=cheap correspondence=pseudo-change, HIGH=genuine change); ``H`` =
        normalised row entropy (match confidence).  Consumed two ways: a spatial
        decoder into the dense change field, and a pooled summary for the
        classifier (see fusion.forward)."""
        in_dtype = a1.dtype
        P, C, row, col = self._plan(a1, a0)
        N = a1.shape[1]
        appeared = (1.0 / N - row).clamp_min(0.0) * N
        disappeared = (1.0 / N - col).clamp_min(0.0) * N
        tau = (P * C).sum(dim=2, keepdim=True) / (row + 1e-8)
        Pbar = P / (row + 1e-8)
        H = -(Pbar * Pbar.clamp_min(1e-12).log()).sum(dim=2, keepdim=True) / math.log(N)
        return (appeared.to(in_dtype), disappeared.to(in_dtype), tau.to(in_dtype), H.to(in_dtype))

    def forward(self, a_q: Tensor, a_k: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Match each query (t1) cell to key (t0) content.

        Returns ``(matched, tau, unmatched)`` with shapes ``(B,N,d)``,
        ``(B,N,1)``, ``(B,N,1)``.
        """
        B, N, _d = a_q.shape
        in_dtype = a_q.dtype
        eps = F.softplus(self.raw_eps) + 1e-4
        gamma = F.softplus(self.raw_gamma)
        rho = F.softplus(self.raw_rho) + 1e-4

        q = F.normalize(self.q_proj(a_q), dim=-1).float()
        k = F.normalize(self.k_proj(a_k), dim=-1).float()
        cos = torch.bmm(q, k.transpose(1, 2))  # (B,N,N) in [-1,1]
        d2 = torch.cdist(self.coords[None], self.coords[None]).pow(2)  # (1,N,N)
        C = (1.0 - cos) + gamma.float() * d2  # ground cost (B,N,N)
        K = -C / eps.float()  # log-kernel

        log_marg = -math.log(N)  # uniform log-marginal
        lam = (rho / (rho + eps)).float()  # ->1 balanced, ->0 no transport
        f = torch.zeros(B, N, device=a_q.device, dtype=torch.float32)
        g = torch.zeros(B, N, device=a_q.device, dtype=torch.float32)
        for _ in range(self.iters):
            f = lam * (log_marg - torch.logsumexp(g[:, None, :] + K, dim=2))
            g = lam * (log_marg - torch.logsumexp(f[:, :, None] + K, dim=1))

        logP = f[:, :, None] + g[:, None, :] + K  # (B,N,N)
        P = logP.exp()
        row = P.sum(dim=2, keepdim=True)  # (B,N,1)
        matched = torch.bmm(P, a_k.float()) / (row + 1e-8)  # (B,N,d)
        tau = (P * C).sum(dim=2, keepdim=True) / (row + 1e-8)  # (B,N,1)
        # Row marginal deficit vs uniform 1/N, scaled to [0,1] = "appeared" mass.
        unmatched = (1.0 / N - row).clamp_min(0.0) * N  # (B,N,1)
        return matched.to(in_dtype), tau.to(in_dtype), unmatched.to(in_dtype)


__all__ = ["SinkhornMatch"]
