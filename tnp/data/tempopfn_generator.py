import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch

# TNP batch type
from tnp.data.synthetic import SyntheticBatch  # adjust import if your path differs
from tnp.data.base import DataGenerator        # adjust import if your path differs

# TempoPFN loader
from TempoPFN.src.data.datasets import CyclicalBatchDataset  # from TempoPFN repo


def _normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
    w = {k: float(v) for k, v in w.items() if float(v) > 0}
    s = sum(w.values())
    if s <= 0:
        raise ValueError("All mixture weights are <= 0. Provide at least one positive weight.")
    return {k: v / s for k, v in w.items()}

def _normalize_seq_mean_std(y: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Z-score normalize ONE sequence: (y - mean) / std
    Accepts y with shape [L], [L,1], or [1,L,1].
    """
    orig_shape = y.shape
    y1 = y.reshape(-1).to(dtype=torch.float32)  # [L]

    mean = y1.mean()
    std = y1.std(unbiased=False).clamp_min(eps)
    y1n = (y1 - mean) / std
    #y1n = torch.clamp(y1n, min=-5.0, max=5.0)

    return y1n.reshape(orig_shape)


def _normalize_seq_minmax_to_unit(y: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Min-max normalize ONE sequence to [-1, 1]:
        2*(y - min)/(max-min) - 1
    Accepts y with shape [L], [L,1], or [1,L,1].
    """
    orig_shape = y.shape
    y1 = y.reshape(-1).to(dtype=torch.float32)  # [L]

    y_min = y1.min()
    y_max = y1.max()
    denom = (y_max - y_min).clamp_min(eps)
    y1n = 2.0 * (y1 - y_min) / denom - 1.0

    return y1n.reshape(orig_shape)


def _values_to_tensor(sample: dict) -> torch.Tensor:
    """
    Convert TempoPFN sample['values'] to torch Tensor of shape [1, L, 1] (univariate).
    TempoPFN can store values as either:
      - old: [v1, v2, ...]
      - new: [[v1, v2, ...]] for univariate
    """
    values = sample["values"]
    num_channels = int(sample.get("num_channels", 1))
    if num_channels != 1:
        raise ValueError(f"Expected univariate (num_channels=1), got {num_channels}")

    if isinstance(values, list) and len(values) > 0 and isinstance(values[0], list):
        # new format: [[...]]
        arr = torch.tensor(values[0], dtype=torch.float32)
    else:
        # old format: [...]
        arr = torch.tensor(values, dtype=torch.float32)

    # [1, L, 1]
    return arr.view(1, -1, 1)


def _make_time_inputs(
    L: int,
    dim: int = 1,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Create x in [0, 1] for each series, shape [1, L, dim].
    For dim>1, we just repeat the 1D time input across dims.
    """
    t = (torch.arange(L, dtype=torch.float32, device=device) / float(L)).view(1, L, 1)
    if dim == 1:
        return t
    return t.repeat(1, 1, dim)


def _split_prefix_context(
    x_full: torch.Tensor,  # [B, L, dim]
    y_full: torch.Tensor,  # [B, L, 1]
    min_nc: int,
    max_nc: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample context length nc and split as:
      context = prefix length nc
      target  = remainder length L-nc
    """
    B, L, _ = x_full.shape
    # ensure at least 1 target point
    max_nc_eff = min(int(max_nc), L - 1)
    min_nc_eff = min(int(min_nc), max_nc_eff)
    if max_nc_eff < 1:
        raise ValueError(f"Sequence length L={L} too short to form targets.")

    nc = int(torch.randint(low=min_nc_eff, high=max_nc_eff + 1, size=()).item())

    xc = x_full[:, :nc, :]
    yc = y_full[:, :nc, :]
    xt = x_full[:, nc:, :]
    yt = y_full[:, nc:, :]
    return xc, yc, xt, yt


class TempoPFNGenerator(DataGenerator):
    """
    IterableDataset that:
      - samples which TempoPFN generator to draw from using mixture weights
      - uses TempoPFN CyclicalBatchDataset to fetch real saved samples
      - converts them into TNP SyntheticBatch
      - context = prefix, targets = remainder
    """

    def __init__(
        self,
        *,
        tempo_root: str,
        mixture_weights: Dict[str, float],
        length: int,
        dim: int,
        min_nc: int,
        max_nc: int,
        samples_per_epoch: int,
        batch_size: int,
        deterministic: bool = False,
        deterministic_seed: int = 0,
        global_seed: int = 0,
        reject_nans: bool = True,
        max_resample_tries: int = 200,
        **kwargs,
    ):
        super().__init__(
            samples_per_epoch=samples_per_epoch,
            batch_size=batch_size,
            deterministic=deterministic,
            deterministic_seed=deterministic_seed,
            **kwargs,
        )

        self.tempo_root = tempo_root
        self.length = int(length)
        self.dim = int(dim)
        self.min_nc = int(min_nc)
        self.max_nc = int(max_nc)
        self.reject_nans = bool(reject_nans)
        self.max_resample_tries = int(max_resample_tries)

        # RNG
        self.rng = np.random.default_rng(int(global_seed))
        random.seed(int(global_seed))
        torch.manual_seed(int(global_seed))

        # Normalize weights
        self.mixture_weights = _normalize_weights(mixture_weights)
        self._gen_names = list(self.mixture_weights.keys())
        self._gen_probs = np.array([self.mixture_weights[k] for k in self._gen_names], dtype=float)

        # Build datasets
        self.datasets: Dict[str, CyclicalBatchDataset] = {}
        for gen_name in self._gen_names:
            batches_dir = f"{self.tempo_root}/{gen_name}"
            self.datasets[gen_name] = CyclicalBatchDataset(
                batches_dir=batches_dir,
                generator_type=gen_name,
                device=None,
                prefetch_next=True,
                prefetch_threshold=32,
            )

    def _choose_generator(self) -> str:
        return str(self.rng.choice(self._gen_names, p=self._gen_probs))

    def _get_one_univariate_series(self) -> Tuple[torch.Tensor, str]:
        """
        Returns y_full of shape [1, L, 1] with robust clipping and normalization.
        """
        for _ in range(self.max_resample_tries):
            gen_name = self._choose_generator()
            sample = self.datasets[gen_name].get_sample()
            y = _values_to_tensor(sample)  # [1, L_raw, 1]

            # 1. Length Check & Cutting
            L_raw = y.shape[1]
            if L_raw < self.length:
                continue
            if L_raw > self.length:
                start = int(self.rng.integers(0, L_raw - self.length + 1))
                y = y[:, start : start + self.length, :]

            # 2. Initial NaN check
            if torch.isnan(y).any():
                continue

            # 3. CLIPPING (Crucial for FP16/BFP16 stability)
            # We clip to a range that won't overflow when squared (for variance math)
            y = torch.clamp(y, min=-2000.0, max=2000.0)

            # 4. Robust Normalization
            # Using a larger epsilon (1e-5) to prevent division by near-zero std
            if int(self.rng.integers(0, 2)) == 0:
                y = _normalize_seq_mean_std(y, eps=1e-5)
            else:
                y = _normalize_seq_minmax_to_unit(y, eps=1e-5)

            # 5. FINAL SAFETY CHECK
            # If normalization resulted in Inf or NaN (e.g. constant sequence), reject it
            if not torch.isfinite(y).all():
                continue

            return y, gen_name

        raise RuntimeError(f"Could not sample a valid series after {self.max_resample_tries} tries.")


    def generate_batch(self) -> SyntheticBatch:
        # Build batch by sampling batch_size independent series
        ys = []
        gen_names = []
        for _ in range(self.batch_size):
            y, gen_name = self._get_one_univariate_series()
            ys.append(y)
            gen_names.append(gen_name)
        y_full = torch.cat(ys, dim=0)  # [B, L, 1]

        # Make time inputs
        x_one = _make_time_inputs(L=self.length, dim=self.dim, device=y_full.device)  # [1, L, dim]
        x_full = x_one.repeat(self.batch_size, 1, 1)  # [B, L, dim]

        # Split
        xc, yc, xt, yt = _split_prefix_context(x_full, y_full, self.min_nc, self.max_nc)

        return SyntheticBatch(
            x=x_full,
            y=y_full,
            xc=xc,
            yc=yc,
            xt=xt,
            yt=yt,
            gt_pred=None,   # saved TempoPFN data has no analytic GT predictor
            generator_name=gen_names
        )
