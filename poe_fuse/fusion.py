"""Cross-expert change fusion.

This module is the trainable heart of the change-detection variant of the
PoEFuse codec.  It consumes the *per-expert, per-timestep* projected tokens
produced by the codec's per-branch ``Linear`` layers and assembles the token
sequence that is then run through the Mamba-3 stack.

Inputs (per forward):

* ``proj``: ``dict[str, Tensor]`` mapping each expert key (``"dino"``,
  ``"sam3"``, ``"gemma"``) to a ``(B, T, N_e, d)`` tensor of projected
  features (already in the shared ``d_model`` space).
* ``image_mask``: ``(B, T)`` bool marking valid time steps.

What it produces (``forward`` returns ``(tokens, token_mask)``):

* base per-timestep tokens (optionally tagged with expert + segment
  embeddings);
* per-expert signed difference tokens ``D_e = p_e(t_last) - p_e(t0)`` and
  optional ``|D_e|`` (the cheap "explicit difference" path -- treated as an
  internal mechanism, not the headline);
* a learned per-expert **change gate** computed from a *cross-expert
  attention* over each expert's pooled change summary (the cross-expert
  gate: heterogeneous frozen experts corroborate each other's change
  evidence); and
* a handful of global **agreement tokens** (the refined cross-expert
  summaries) prepended to the sequence.

Every piece is individually toggled by :class:`FusionConfig`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import FusionConfig

# Segment-embedding row indices.
_SEG_T0 = 0
_SEG_T1 = 1
_SEG_DIFF = 2
_SEG_ABS = 3
_SEG_AGREE = 4
_N_SEG = 5


class CrossExpertChangeFusion(nn.Module):
    def __init__(self, cfg: FusionConfig, *, d_model: int, branch_keys: tuple[str, ...]):
        super().__init__()
        self.cfg = cfg
        self.d_model = d_model
        self.branch_keys = tuple(branch_keys)
        E = len(self.branch_keys)
        self.n_experts = E

        # Identify each expert / temporal role so the (permutation-agnostic)
        # Mamba mixer and the heads can disambiguate the concatenated stream.
        self.expert_emb = nn.Parameter(torch.randn(E, d_model) * 0.02)
        if cfg.add_segment_embeddings:
            self.seg_emb = nn.Parameter(torch.randn(_N_SEG, d_model) * 0.02)
        else:
            self.register_parameter("seg_emb", None)

        self._needs_summary = cfg.cross_expert_gate or cfg.agreement_tokens
        if self._needs_summary:
            self.summary_in_norm = nn.LayerNorm(d_model)
            self.cross_attn = nn.MultiheadAttention(
                d_model, num_heads=cfg.n_heads, batch_first=True
            )
            self.summary_out_norm = nn.LayerNorm(d_model)
        if cfg.cross_expert_gate:
            self.gate_mlp = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, 1),
            )

    # --------------------------------------------------------------- utils
    def _seg(self, idx: int, ref: Tensor) -> Tensor:
        if self.seg_emb is None:
            return ref.new_zeros(self.d_model)
        return self.seg_emb[idx]

    def forward(self, proj: dict[str, Tensor], image_mask: Tensor) -> tuple[Tensor, Tensor]:
        any_expert = proj[self.branch_keys[0]]
        B, T = any_expert.shape[0], any_expert.shape[1]
        device = any_expert.device
        image_mask = image_mask.to(torch.bool)

        # Per-expert signed change (last vs first frame) and its summary.
        diffs: dict[str, Tensor] = {}
        summaries = []  # (B, d) per expert, used for cross-expert attention
        for e, key in enumerate(self.branch_keys):
            p = proj[key]  # (B, T, N_e, d)
            d_e = p[:, -1] - p[:, 0]  # (B, N_e, d)
            diffs[key] = d_e
            summaries.append(d_e.mean(dim=1))  # (B, d)

        gate = None
        agree_tokens = None
        if self._needs_summary:
            C = torch.stack(summaries, dim=1)  # (B, E, d)
            C = self.summary_in_norm(C)
            attn_out, _ = self.cross_attn(C, C, C, need_weights=False)
            C2 = self.summary_out_norm(C + attn_out)  # (B, E, d)
            if self.cfg.cross_expert_gate:
                gate = torch.sigmoid(self.gate_mlp(C2))  # (B, E, 1)
            if self.cfg.agreement_tokens:
                agree = C2 + self.expert_emb.unsqueeze(0)  # (B, E, d)
                agree = agree + self._seg(_SEG_AGREE, agree)
                agree_tokens = agree

        tok_parts: list[Tensor] = []
        mask_parts: list[Tensor] = []

        # (1) global agreement tokens (always valid).
        if agree_tokens is not None:
            tok_parts.append(agree_tokens)
            mask_parts.append(torch.ones(B, self.n_experts, dtype=torch.bool, device=device))

        # (2) base per-timestep tokens, expert/segment tagged + gated.
        for e, key in enumerate(self.branch_keys):
            p = proj[key]  # (B, T, N_e, d)
            N_e = p.shape[2]
            g_e = gate[:, e].unsqueeze(1) if gate is not None else None  # (B,1,1)
            for t in range(T):
                seg_idx = _SEG_T0 if t == 0 else _SEG_T1
                tok = p[:, t] + self.expert_emb[e] + self._seg(seg_idx, p)
                if g_e is not None:
                    tok = tok * g_e
                tok_parts.append(tok)
                valid = image_mask[:, t].unsqueeze(1).expand(B, N_e)
                mask_parts.append(valid)

        # (3) difference tokens (signed and optional abs), gated.
        both_valid = image_mask[:, 0] & image_mask[:, -1]  # (B,)
        if self.cfg.diff_tokens:
            for e, key in enumerate(self.branch_keys):
                d_e = diffs[key]  # (B, N_e, d)
                N_e = d_e.shape[1]
                g_e = gate[:, e].unsqueeze(1) if gate is not None else None
                signed = d_e + self.expert_emb[e] + self._seg(_SEG_DIFF, d_e)
                if g_e is not None:
                    signed = signed * g_e
                tok_parts.append(signed)
                mask_parts.append(both_valid.unsqueeze(1).expand(B, N_e))
                if self.cfg.diff_abs:
                    absd = d_e.abs() + self.expert_emb[e] + self._seg(_SEG_ABS, d_e)
                    if g_e is not None:
                        absd = absd * g_e
                    tok_parts.append(absd)
                    mask_parts.append(both_valid.unsqueeze(1).expand(B, N_e))

        tokens = torch.cat(tok_parts, dim=1).contiguous()  # (B, L, d)
        token_mask = torch.cat(mask_parts, dim=1).contiguous()  # (B, L)
        return tokens, token_mask


class SpatialChangeAgreementFusion(nn.Module):
    """Spatial Cross-Expert Change-Agreement (the headline ACCV mechanism).

    Unlike :class:`CrossExpertChangeFusion`, which pools each expert's change
    into a single global vector, this module keeps *spatial* structure and the
    two contributions are made first-class:

    **(2) Heterogeneous expert -> shared spatial field.**  DINOv3 patches (196),
    SAM 3 detection queries (200) and Gemma soft tokens (~256) live in
    different token spaces with no common geometry.  A learned set of
    ``G*G`` spatial query tokens cross-attends into each expert (per time step)
    to *resample* every expert onto the **same** ``G x G`` grid, giving
    ``aligned[e]: (B, T, G*G, d)`` that are spatially comparable across experts.

    **(1) Per-location change agreement.**  On that shared grid we form each
    expert's signed change ``D_e = aligned_e(t_last) - aligned_e(t0)`` and read
    off, *per location*, how much the experts agree on the change *direction*::

        u_e        = D_e / ||D_e||                      (unit change directions)
        agreement  = || mean_e u_e ||  in [0, 1]        (1=corroborated, 0=cancel)

    ``agreement`` is high where heterogeneous experts independently see the same
    change (likely a true semantic change) and low where they disagree (likely
    illumination / seasonal / registration pseudo-change).  We use it to gate
    the fused change map, suppressing pseudo-change.

    Emitted token stream (all in shared ``d``):

    * per-timestep, expert-fused **aligned tokens** (``T * G*G``);
    * the agreement-weighted **change map** (``G*G``); and
    * a few global **agreement summary** tokens.

    The per-location ``agreement`` map is stashed on ``self.last_agreement``
    (detached) for analysis / visualisation.
    """

    def __init__(self, cfg: FusionConfig, *, d_model: int, branch_keys: tuple[str, ...]):
        super().__init__()
        self.cfg = cfg
        self.d_model = d_model
        self.branch_keys = tuple(branch_keys)
        E = len(self.branch_keys)
        self.n_experts = E
        self.G = int(cfg.align_grid)
        self.n_grid = self.G * self.G
        # DINOv3 patches are spatially ordered and equal the grid size, so use
        # that expert as the canonical grid (identity) instead of resampling.
        self.identity_align = tuple(k == "dino" for k in self.branch_keys)

        # Shared spatial query grid (also acts as a learned 2D positional code).
        self.grid_query = nn.Parameter(torch.randn(self.n_grid, d_model) * 0.02)
        self.expert_emb = nn.Parameter(torch.randn(E, d_model) * 0.02)
        if cfg.add_segment_embeddings:
            self.seg_emb = nn.Parameter(torch.randn(_N_SEG, d_model) * 0.02)
        else:
            self.register_parameter("seg_emb", None)

        # One resampler cross-attention per expert (keys/values = expert tokens).
        self.key_norm = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(E))
        self.resampler = nn.ModuleList(
            nn.MultiheadAttention(d_model, num_heads=cfg.align_heads, batch_first=True)
            for _ in range(E)
        )
        self.aligned_norm = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(E))

        # Optional learned per-expert change gate (cross-expert attention over
        # each expert's pooled change summary), reused from the global path.
        if cfg.cross_expert_gate:
            self.summary_in_norm = nn.LayerNorm(d_model)
            self.cross_attn = nn.MultiheadAttention(
                d_model, num_heads=cfg.n_heads, batch_first=True
            )
            self.summary_out_norm = nn.LayerNorm(d_model)
            self.gate_mlp = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, 1),
            )

        # Bidirectional agreement modulation: a zero-mean (over space) gate so
        # high-agreement locations are *amplified* and low-agreement ones
        # *suppressed* without globally shrinking the change signal.  The
        # learnable scale lets the model decide how aggressively to trust it.
        self.agree_scale = nn.Parameter(torch.ones(()))

        # --- Bi-temporal cross-alignment (novel temporal correspondence) ------
        # Each t1 grid location cross-attends into the t0 grid (and vice versa)
        # so the signed change is computed against the *matched* content rather
        # than the co-located cell.  Learned 2D relative position bias gives the
        # attention a soft locality prior (matches should be near, not global).
        self.bitemporal = bool(getattr(cfg, "bitemporal_align", False))
        if self.bitemporal:
            self.temporal_q_norm = nn.LayerNorm(d_model)
            self.temporal_kv_norm = nn.LayerNorm(d_model)
            self.temporal_attn = nn.MultiheadAttention(
                d_model,
                num_heads=int(getattr(cfg, "bitemporal_heads", 8)),
                batch_first=True,
            )
            self.temporal_out_norm = nn.LayerNorm(d_model)
            # learned scalar mixing the matched-content change with the raw
            # co-located change.  Initialised to a large negative value so
            # sigmoid(mix)~=0 => training STARTS at the proven naive-difference
            # path and only blends in correspondence-aware change as it helps.
            self.temporal_mix = nn.Parameter(
                torch.tensor(float(getattr(cfg, "temporal_mix_init", -4.0)))
            )
            self.rel_pos_bias = nn.Parameter(torch.zeros(self.n_grid, self.n_grid))

        # --- Unbalanced optimal-transport temporal correspondence (headline) ---
        # Selects an entropic-OT plan instead of soft cross-attention for the
        # t1<->t0 matcher, and exposes the transport-residual change field.
        self.transport = bool(getattr(cfg, "transport_align", False))
        # Variant selectors (mutually exclusive; default = legacy v10b matched path).
        self.transport_cost_only = bool(getattr(cfg, "transport_cost_only", False))
        self.transport_marginal = bool(getattr(cfg, "transport_marginal", False))
        self.transport_pool = bool(getattr(cfg, "transport_pool", False))
        self.transport_dense_on = bool(getattr(cfg, "transport_dense", True))
        if self.transport:
            if not self.bitemporal:
                raise ValueError("fusion.transport_align=true requires bitemporal_align=true")
            from .transport import SinkhornMatch

            self.sinkhorn = SinkhornMatch(
                d_model,
                grid=self.G,
                iters=int(getattr(cfg, "sinkhorn_iters", 4)),
                eps_init=float(getattr(cfg, "sinkhorn_eps", 0.1)),
                rho_init=float(getattr(cfg, "unbalanced_rho", 4.0)),
                decouple_lam=bool(getattr(cfg, "decouple_lam", False)),
                lam_init=float(getattr(cfg, "lam_init", 0.7)),
            )
            if self.transport_pool:
                # (v12) ONE plan, head-aware readouts. Dense: spatial conv decoder
                # of the 4 OT functionals -> change_grid (zero-init last conv =>
                # baseline at init, td_alpha live). Classifier reads a pooled
                # summary in its OWN head (built there), never touching fused_change.
                if self.transport_dense_on:
                    self.td = nn.Sequential(
                        nn.Conv2d(4, 4, kernel_size=3, padding=1, groups=4),
                        nn.GELU(),
                        nn.Conv2d(4, 64, kernel_size=1),
                        nn.GELU(),
                        nn.Conv2d(64, d_model, kernel_size=1),
                    )
                    nn.init.zeros_(self.td[-1].weight)
                    nn.init.zeros_(self.td[-1].bias)
                    self.td_alpha = nn.Parameter(
                        torch.tensor(float(getattr(cfg, "td_alpha_init", 0.1)))
                    )
            elif self.transport_cost_only:
                # (A) lift [tau_z, H_z] -> d_model residual, ADD to fused_change.
                # Live-gradient warm-start: alpha>0 + tiny non-zero weight (zero-init
                # both => dead gradient, the v10 "never engaged" trap).
                self.cost_lift = nn.Linear(2, d_model)
                nn.init.zeros_(self.cost_lift.bias)
                nn.init.normal_(self.cost_lift.weight, std=1e-3)
                self.cost_alpha = nn.Parameter(
                    torch.tensor(float(getattr(cfg, "cost_alpha_init", 0.05)))
                )
            elif self.transport_marginal:
                # (B) per-expert gated [m, s] -> d_model residual added into d_e.
                self.evidence_proj = nn.Linear(2, d_model, bias=False)
                from .transport import _inv_softplus

                # softplus(beta)~=0.05 at init: small (baseline ~recovered) but a
                # LIVE gradient up into lam/evidence_proj (vs near-0 dead gradient).
                self.transport_beta = nn.Parameter(torch.full((E,), float(_inv_softplus(0.05))))
            else:
                # legacy v10b: tau/unmatched fold via cost_scale (matched path).
                self.cost_scale = nn.Parameter(
                    torch.tensor(float(getattr(cfg, "cost_scale_init", 0.0)))
                )
        self.last_transport: dict | None = None
        self.last_transport_summary: Tensor | None = None

        # --- PoE-Fuse: precision-weighted Product-of-Experts combine -----------
        # Per-expert per-cell-per-channel log-precision MLP; posterior mean
        # (inverse-variance weighting) replaces the gated sum at both combine
        # sites.  Zero-init last layer + prec_scale_init=0 => Lambda_e == const at
        # init => recovers the incumbent combiner (up to a per-cell scalar absorbed
        # by the downstream RMSNorm/LayerNorm) with a LIVE gradient (prec_scale
        # multiplies a non-zero MLP output).
        self.poe_fuse = bool(getattr(cfg, "poe_fuse", False))
        # Per-site PoE toggles (ablation): apply the precision combine at the dense
        # (fused_change) and/or classifier (fused_t) site; a disabled site falls
        # back to the incumbent gated sum.  Default both on == headline model.
        self.poe_dense = bool(getattr(cfg, "poe_dense", True))
        self.poe_cls = bool(getattr(cfg, "poe_cls", True))
        if self.poe_fuse:
            from .transport import _inv_softplus

            E2 = self.n_experts
            self.logprec = nn.ModuleList(
                nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, d_model),
                    nn.GELU(),
                    nn.Linear(d_model, d_model),
                )
                for _ in range(E2)
            )
            for m in self.logprec:
                nn.init.zeros_(m[-1].weight)
                nn.init.zeros_(m[-1].bias)
            self.prec_bias = nn.Parameter(torch.full((E2,), float(_inv_softplus(1.0))))
            # poe_static: freeze prec_scale=0 => Lambda_e = softplus(prec_bias_e),
            # a learned but input-independent per-expert precision (ablation).
            self._poe_static = bool(getattr(cfg, "poe_static", False))
            init_scale = 0.0 if self._poe_static else float(getattr(cfg, "poe_scale_init", 0.0))
            self.prec_scale = nn.Parameter(
                torch.tensor(init_scale), requires_grad=not self._poe_static
            )
            # B (difference-precision corollary) on by default; off = single-timestep.
            self.poe_diff_corollary = bool(getattr(cfg, "poe_diff_corollary", True))
        self.last_poe_weights: Tensor | None = None

        self.last_agreement: Tensor | None = None
        self.last_change_grid: Tensor | None = None
        # Token-stream layout of the last forward (consumed by the codec's
        # location-major "Change-Anchored Scan" permutation).
        self.last_layout: dict | None = None

    def _seg(self, idx: int, ref: Tensor) -> Tensor:
        if self.seg_emb is None:
            return ref.new_zeros(self.d_model)
        return self.seg_emb[idx]

    def _align(self, e: int, frame: Tensor) -> Tensor:
        """Resample one expert's ``(B, N_e, d)`` tokens onto the shared grid.

        A spatially-ordered expert whose token count already equals the grid
        (DINOv3 patches, 14x14=196) is used as the **canonical grid via
        identity** -- this gives the shared field real spatial structure from
        step 0 so the dense decoder can learn immediately; the unordered
        experts (SAM 3 detection queries, Gemma soft tokens) are cross-attention
        resampled onto that same grid.
        """
        if self.identity_align[e] and frame.shape[1] == self.n_grid:
            return self.aligned_norm[e](frame + self.expert_emb[e])
        B = frame.shape[0]
        q = (self.grid_query + self.expert_emb[e]).unsqueeze(0).expand(B, -1, -1)
        k = self.key_norm[e](frame)
        attn_out, _ = self.resampler[e](q, k, k, need_weights=False)
        return self.aligned_norm[e](q + attn_out)  # (B, G*G, d)

    def _precision(self, e: int, a: Tensor) -> Tensor:
        """Per-cell-per-channel positive precision Lambda_e for expert ``e``.

        ``softplus(prec_bias_e + prec_scale * MLP_e(a))``; at init (prec_scale=0,
        zero-init MLP last layer) this is the constant ``softplus(prec_bias)=1``,
        so the PoE combine reduces to the (gated) uniform mean = incumbent."""
        return F.softplus(self.prec_bias[e] + self.prec_scale * self.logprec[e](a))

    def _temporal_match(self, a0: Tensor, a1: Tensor) -> Tensor:
        """Cross-attend each t1 location into the t0 grid; return matched t0.

        ``a0``/``a1`` are ``(B, G*G, d)``.  Returns the t0 content matched to
        each t1 location, so ``a1 - matched_t0`` is a correspondence-aware
        change (handles displacement) rather than a co-located subtraction.
        """
        q = self.temporal_q_norm(a1)
        kv = self.temporal_kv_norm(a0)
        attn_out, _ = self.temporal_attn(
            q,
            kv,
            kv,
            need_weights=False,
            attn_mask=self.rel_pos_bias,
        )
        return self.temporal_out_norm(attn_out)

    def forward(self, proj: dict[str, Tensor], image_mask: Tensor) -> tuple[Tensor, Tensor]:
        any_expert = proj[self.branch_keys[0]]
        B, T = any_expert.shape[0], any_expert.shape[1]
        image_mask = image_mask.to(torch.bool)

        # (2) resample every expert onto the shared G*G grid, per timestep.
        aligned: dict[str, Tensor] = {}
        for e, key in enumerate(self.branch_keys):
            p = proj[key]  # (B, T, N_e, d)
            frames = [self._align(e, p[:, t]) for t in range(T)]
            aligned[key] = torch.stack(frames, dim=1)  # (B, T, G*G, d)
        self.last_aligned = {k: v.detach() for k, v in aligned.items()}  # diagnostic (figures)

        # (1) per-location signed change + agreement across experts.
        eps = 1e-6
        diffs = []
        units = []
        tau_list: list[Tensor] = []
        unmatched_list: list[Tensor] = []
        H_list: list[Tensor] = []
        pool_ap: list[Tensor] = []
        pool_dis: list[Tensor] = []
        pool_tau: list[Tensor] = []
        pool_H: list[Tensor] = []
        additive = self.transport and (self.transport_cost_only or self.transport_marginal)
        v12 = self.transport and self.transport_pool
        for e_idx, key in enumerate(self.branch_keys):
            a0 = aligned[key][:, 0]
            a1 = aligned[key][:, -1]
            if v12 and T >= 2:
                # (v12) co-located difference untouched; collect OT functionals.
                d_e = a1 - a0
                ap_e, dis_e, tau_e, H_e = self.sinkhorn.readout(a0, a1)
                pool_ap.append(ap_e)
                pool_dis.append(dis_e)
                pool_tau.append(tau_e)
                pool_H.append(H_e)
            elif additive and T >= 2:
                # v11: keep the proven co-located difference; OT enters ADDITIVELY.
                d_e = a1 - a0
                if self.transport_cost_only:
                    tau_e, H_e = self.sinkhorn.forward_cost(a1, a0)
                    tau_list.append(tau_e)
                    H_list.append(H_e)
                else:  # transport_marginal (B): per-expert gated residual into d_e
                    m_s = self.sinkhorn.forward_marginals(a0, a1)  # (B,N,2)
                    beta = torch.nn.functional.softplus(self.transport_beta[e_idx])
                    d_e = d_e + beta * self.evidence_proj(m_s)
            elif (self.bitemporal or self.transport) and T >= 2:
                # Correspondence-aware change: match each t1 cell to its t0
                # content, then blend with the naive co-located difference.
                if self.transport:
                    matched_t0, tau_e, unm_e = self.sinkhorn(a1, a0)
                    tau_list.append(tau_e)
                    unmatched_list.append(unm_e)
                else:
                    matched_t0 = self._temporal_match(a0, a1)  # (B, G*G, d)
                d_corr = a1 - matched_t0
                d_naive = a1 - a0
                mix = torch.sigmoid(self.temporal_mix)
                d_e = mix * d_corr + (1.0 - mix) * d_naive
            else:
                d_e = a1 - a0  # (B, G*G, d)
            diffs.append(d_e)
            units.append(d_e / (d_e.norm(dim=-1, keepdim=True) + eps))
        D = torch.stack(diffs, dim=1)  # (B, E, G*G, d)
        U = torch.stack(units, dim=1)  # (B, E, G*G, d)
        mean_u = U.mean(dim=1)  # (B, G*G, d)
        agreement = mean_u.norm(dim=-1, keepdim=True)  # (B, G*G, 1) in [0,1]
        self.last_agreement = agreement.detach()

        # Optional per-expert gate from pooled change summaries.
        gate = None
        if self.cfg.cross_expert_gate:
            summ = D.mean(dim=2)  # (B, E, d)
            summ = self.summary_in_norm(summ)
            attn_out, _ = self.cross_attn(summ, summ, summ, need_weights=False)
            summ = self.summary_out_norm(summ + attn_out)
            gate = torch.sigmoid(self.gate_mlp(summ))  # (B, E, 1)

        # Fused signed change.  PoE: precision-weighted (difference-precision)
        # posterior mean over experts; else the legacy gated sum.
        if self.poe_fuse and self.poe_dense:
            LamD = []
            for e, key in enumerate(self.branch_keys):
                l0 = self._precision(e, aligned[key][:, 0])
                l1 = self._precision(e, aligned[key][:, -1])
                if self.poe_diff_corollary:
                    ld = 1.0 / (1.0 / (l0 + 1e-6) + 1.0 / (l1 + 1e-6))  # Var(x1-x0)=S0+S1
                else:
                    ld = l1  # ablation: single-timestep precision (no corollary)
                if gate is not None:
                    ld = ld * gate[:, e].unsqueeze(1)
                LamD.append(ld)
            LamD_star = sum(LamD) + 1e-6
            fused_change = sum(LamD[e] * D[:, e] for e in range(self.n_experts)) / LamD_star
        elif gate is not None:
            fused_change = (D * gate.unsqueeze(2)).sum(dim=1)  # (B, G*G, d)
        else:
            fused_change = D.sum(dim=1)
        if self.cfg.spatial_agreement:
            # Zero-mean (over space) bidirectional modulation: amplify
            # corroborated locations, attenuate disagreed ones.
            centered = agreement - agreement.mean(dim=1, keepdim=True)
            fused_change = fused_change * (1.0 + self.agree_scale * centered)

        # (v12) head-aware transport: dense spatial-conv residual into change_grid
        # (only on change_seg configs; zero-init last conv => baseline at init), and
        # ALWAYS stash a pooled 8-stat summary for the classifier head (which reads
        # it through its own zero-init gate, never perturbing fused_change).
        if self.transport_pool and pool_ap:
            ap = torch.stack(pool_ap, dim=1).mean(dim=1)  # (B,N,1)
            dis = torch.stack(pool_dis, dim=1).mean(dim=1)
            tau = torch.stack(pool_tau, dim=1).mean(dim=1)
            Hh = torch.stack(pool_H, dim=1).mean(dim=1)

            def _zc(x):
                return (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-5)

            if self.transport_dense_on:
                Sg = torch.cat([_zc(ap), _zc(dis), _zc(tau), _zc(Hh)], dim=-1)  # (B,N,4)
                Sg = Sg.transpose(1, 2).reshape(B, 4, self.G, self.G)
                resid = self.td(Sg).reshape(B, self.d_model, self.n_grid).transpose(1, 2)
                fused_change = fused_change + self.td_alpha * resid
            summary = torch.cat(
                [
                    ap.mean(dim=1),
                    dis.mean(dim=1),
                    (ap > 0).to(ap.dtype).mean(dim=1),
                    (dis > 0).to(dis.dtype).mean(dim=1),
                    tau.mean(dim=1),
                    Hh.mean(dim=1),
                    tau.std(dim=1),
                    (agreement * ap).sum(dim=1) / (agreement.sum(dim=1) + 1e-8),
                ],
                dim=-1,
            )  # (B,8)
            self.last_transport_summary = summary

        # Transport-residual modulation: amplify created/destroyed locations
        # (high unmatched mass = real change) and attenuate transported/persisted
        # ones (low tau = pseudo-change).  Zero-mean over space so it re-weights
        # rather than globally rescales; cost_scale init 0 => no-op at start.
        if self.transport_cost_only and tau_list:
            # (A) additive tau + entropy residual (z-normalised over space).
            tau = torch.stack(tau_list, dim=1).mean(dim=1)  # (B, G*G, 1)
            H = torch.stack(H_list, dim=1).mean(dim=1)  # (B, G*G, 1)

            def _z(x):
                return (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-5)

            s = torch.cat([_z(tau), _z(H)], dim=-1)  # (B, G*G, 2)
            fused_change = fused_change + self.cost_alpha * self.cost_lift(s)
            self.last_transport = {"tau": tau.detach(), "H": H.detach()}
        elif (
            self.transport
            and not self.transport_marginal
            and getattr(self.cfg, "transport_cost_channel", True)
            and tau_list
        ):
            # legacy v10b multiplicative fold (matched path)
            tau = torch.stack(tau_list, dim=1).mean(dim=1)  # (B, G*G, 1)
            unm = torch.stack(unmatched_list, dim=1).mean(dim=1)  # (B, G*G, 1)
            u = unm - unm.mean(dim=1, keepdim=True)
            tcen = tau - tau.mean(dim=1, keepdim=True)
            fused_change = fused_change * (1.0 + self.cost_scale * (u - tcen))
            self.last_transport = {"tau": tau.detach(), "unmatched": unm.detach()}

        # Expose the shared-grid change field (B, G*G, d) for the dense
        # change-segmentation head (kept attached so grads reach the fusion).
        self.last_change_grid = fused_change

        tok_parts: list[Tensor] = []
        mask_parts: list[Tensor] = []

        # (a) per-timestep, expert-fused aligned spatial tokens.
        for t in range(T):
            seg_idx = _SEG_T0 if t == 0 else _SEG_T1
            if self.poe_fuse and self.poe_cls:
                # PoE posterior mean over experts (reaches the mean-pool classifier).
                A = [
                    aligned[key][:, t] + self.expert_emb[e]
                    for e, key in enumerate(self.branch_keys)
                ]
                Lam = [
                    self._precision(e, aligned[key][:, t]) for e, key in enumerate(self.branch_keys)
                ]
                if gate is not None:
                    Lam = [Lam[e] * gate[:, e].unsqueeze(1) for e in range(self.n_experts)]
                Lstar = sum(Lam) + 1e-6
                fused_t = sum(Lam[e] * A[e] for e in range(self.n_experts)) / Lstar
                if t == T - 1:
                    # diagnostic: per-expert channel-summed precision share (B,N,E)
                    self.last_poe_weights = torch.stack(
                        [lm.mean(dim=-1) for lm in Lam], dim=-1
                    ).detach()
            elif gate is not None:
                fused_t = sum(
                    (aligned[key][:, t] + self.expert_emb[e]) * gate[:, e].unsqueeze(1)
                    for e, key in enumerate(self.branch_keys)
                )
            else:
                fused_t = sum(
                    aligned[key][:, t] + self.expert_emb[e]
                    for e, key in enumerate(self.branch_keys)
                )
            fused_t = fused_t + self._seg(seg_idx, fused_t)
            tok_parts.append(fused_t)
            valid = image_mask[:, t].unsqueeze(1).expand(B, self.n_grid)
            mask_parts.append(valid)

        both_valid = image_mask[:, 0] & image_mask[:, -1]  # (B,)

        # (b) agreement-weighted change-map tokens.
        if self.cfg.keep_change_tokens:
            change_tok = fused_change + self._seg(_SEG_DIFF, fused_change)
            tok_parts.append(change_tok)
            mask_parts.append(both_valid.unsqueeze(1).expand(B, self.n_grid))

        # (c) global agreement-summary tokens (one per expert, agreement-pooled).
        if self.cfg.agreement_tokens:
            w = agreement / (agreement.sum(dim=1, keepdim=True) + eps)  # (B,G*G,1)
            for e in range(self.n_experts):
                pooled = (D[:, e] * w).sum(dim=1)  # (B, d) agreement-weighted
                summary = (pooled + self.expert_emb[e]).unsqueeze(1)
                summary = summary + self._seg(_SEG_AGREE, summary)
                tok_parts.append(summary)
                mask_parts.append(both_valid.unsqueeze(1))

        tokens = torch.cat(tok_parts, dim=1).contiguous()  # (B, L, d)
        token_mask = torch.cat(mask_parts, dim=1).contiguous()  # (B, L)
        self.last_layout = {
            "n_grid": self.n_grid,
            "n_time": T,
            "has_change": bool(self.cfg.keep_change_tokens),
            "n_tail": self.n_experts if self.cfg.agreement_tokens else 0,
        }
        return tokens, token_mask


__all__ = ["CrossExpertChangeFusion", "SpatialChangeAgreementFusion"]
