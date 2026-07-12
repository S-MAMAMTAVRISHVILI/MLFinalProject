from __future__ import annotations

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

__all__ = ["DLinear", "NBeats", "PatchTST", "train_torch", "train_dlinear",
           "SeqForecaster", "DLinearForecaster"]


class _MovingAvg(nn.Module):

    def __init__(self, kernel: int):
        super().__init__()
        self.kernel = kernel
        self.pool = nn.AvgPool1d(kernel, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left = (self.kernel - 1) // 2
        right = self.kernel - 1 - left
        pad = torch.cat([x[:, :1].repeat(1, left), x, x[:, -1:].repeat(1, right)], dim=1)
        return self.pool(pad.unsqueeze(1)).squeeze(1)


class DLinear(nn.Module):

    def __init__(self, lookback: int, horizon: int, kernel: int = 25):
        super().__init__()
        self.lookback, self.horizon, self.kernel = lookback, horizon, kernel
        self.decomp = _MovingAvg(kernel)
        self.linear_trend = nn.Linear(lookback, horizon)
        self.linear_seasonal = nn.Linear(lookback, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trend = self.decomp(x)
        seasonal = x - trend
        return self.linear_trend(trend) + self.linear_seasonal(seasonal)

    @property
    def arch(self) -> dict:
        return {"lookback": self.lookback, "horizon": self.horizon, "kernel": self.kernel}


class _NBeatsBlock(nn.Module):

    def __init__(self, lookback: int, horizon: int, width: int, n_layers: int):
        super().__init__()
        layers = [nn.Linear(lookback, width), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(width, width), nn.ReLU()]
        self.fc = nn.Sequential(*layers)
        self.backcast = nn.Linear(width, lookback)
        self.forecast = nn.Linear(width, horizon)

    def forward(self, x):
        h = self.fc(x)
        return self.backcast(h), self.forecast(h)


class NBeats(nn.Module):

    def __init__(self, lookback: int, horizon: int, width: int = 256,
                 n_blocks: int = 3, n_layers: int = 4):
        super().__init__()
        self.lookback, self.horizon = lookback, horizon
        self.width, self.n_blocks, self.n_layers = width, n_blocks, n_layers
        self.blocks = nn.ModuleList([
            _NBeatsBlock(lookback, horizon, width, n_layers) for _ in range(n_blocks)])

    def forward(self, x):
        residual = x
        forecast = x.new_zeros(x.size(0), self.horizon)
        for block in self.blocks:
            back, fore = block(residual)
            residual = residual - back
            forecast = forecast + fore
        return forecast

    @property
    def arch(self) -> dict:
        return {"lookback": self.lookback, "horizon": self.horizon, "width": self.width,
                "n_blocks": self.n_blocks, "n_layers": self.n_layers}


class PatchTST(nn.Module):
    """Channel-independent PatchTST (Nie et al., 2023) on univariate windows.

    The lookback is cut into overlapping patches, each patch is linearly embedded,
    a Transformer encoder mixes the patch sequence, and a flatten head maps to the
    horizon. Same [B, lookback] -> [B, horizon] contract as DLinear/NBeats, so it
    trains with the same `train_torch` loop and WalmartPanel windows.
    """

    def __init__(self, lookback: int, horizon: int, patch_len: int = 8, stride: int = 4,
                 d_model: int = 64, n_heads: int = 4, n_layers: int = 2,
                 d_ff: int = 128, dropout: float = 0.1):
        super().__init__()
        if lookback < patch_len:
            raise ValueError(f"lookback {lookback} shorter than patch_len {patch_len}")
        self.lookback, self.horizon = lookback, horizon
        self.patch_len, self.stride = patch_len, stride
        self.d_model, self.n_heads, self.n_layers = d_model, n_heads, n_layers
        self.d_ff, self.dropout = d_ff, dropout
        n_patches = (lookback - patch_len) // stride + 1
        self.n_patches = n_patches
        self.embed = nn.Linear(patch_len, d_model)
        self.pos = nn.Parameter(torch.zeros(1, n_patches, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_ff,
                                           dropout=dropout, batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.head = nn.Linear(n_patches * d_model, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = x.unfold(1, self.patch_len, self.stride)      # [B, n_patches, patch_len]
        z = self.embed(patches) + self.pos
        z = self.encoder(z)
        return self.head(z.flatten(1))

    @property
    def arch(self) -> dict:
        return {"lookback": self.lookback, "horizon": self.horizon,
                "patch_len": self.patch_len, "stride": self.stride,
                "d_model": self.d_model, "n_heads": self.n_heads,
                "n_layers": self.n_layers, "d_ff": self.d_ff, "dropout": self.dropout}


def train_torch(net: nn.Module, panel, *, epochs: int = 40, lr: float = 1e-3,
                batch_size: int = 512, weight_decay: float = 1e-4, device: str = "cpu",
                seed: int = 0, weighted_loss: bool = True, grad_clip: float = 1.0,
                verbose: bool = True) -> nn.Module:

    torch.manual_seed(seed)
    w = panel.make_windows()
    X = torch.tensor(w.past, dtype=torch.float32)
    Y = torch.tensor(w.future, dtype=torch.float32)
    W = (torch.tensor(panel.horizon_holiday_weights(w.origin_t), dtype=torch.float32)
         if weighted_loss else torch.ones_like(Y))

    net = net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    n = len(X)
    gen = torch.Generator().manual_seed(seed)

    net.train()
    for epoch in range(epochs):
        perm = torch.randperm(n, generator=gen)
        total = 0.0
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            xb = X[idx].to(device); yb = Y[idx].to(device); wb = W[idx].to(device)
            opt.zero_grad()
            loss = (wb * (net(xb) - yb).abs()).sum() / wb.sum()
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
            opt.step()
            total += loss.item() * len(idx)
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:3d}  weighted L1 (norm) = {total / n:.4f}")
    net.eval()
    return net


def train_dlinear(panel, *, kernel: int = 25, **kw) -> DLinear:
    net = DLinear(panel.lookback, panel.horizon, kernel)
    return train_torch(net, panel, **kw)


def _forecast_horizon(net: DLinear, panel, device: str = "cpu") -> np.ndarray:

    past, usable = panel.forecast_inputs()
    x = np.nan_to_num(past, nan=0.0)
    net = net.to(device).eval()
    with torch.no_grad():
        out = net(torch.tensor(x, dtype=torch.float32, device=device)).cpu().numpy()

    if getattr(panel, "target_mode", "level") == "residual":
        base = panel.forecast_baseline()
        horizon = base + panel.std_[:, None] * out
        horizon[~usable | ~np.isfinite(base).all(axis=1)] = np.nan
    else:
        horizon = panel.denormalise(np.arange(panel.n_series_), out)
        horizon[~usable] = np.nan
    return horizon


class SeqForecaster:
    """Registry-safe wrapper for any panel sequence net (DLinear, NBeats, PatchTST).

    Stores the architecture name + kwargs + weights as plain numpy, so the pickled
    object reloads on a fresh runtime without a live torch module, exactly like
    DLinearForecaster — but reconstructs whichever architecture it was given.
    """

    _ARCHS = {"DLinear": DLinear, "NBeats": NBeats, "PatchTST": PatchTST}

    def __init__(self, panel, net: nn.Module):
        self.panel = panel
        self.arch_name = type(net).__name__
        if self.arch_name not in self._ARCHS:
            raise ValueError(f"unsupported architecture {self.arch_name}")
        self.arch = net.arch
        self.state = {k: v.cpu().numpy() for k, v in net.state_dict().items()}

    def _net(self) -> nn.Module:
        net = self._ARCHS[self.arch_name](**self.arch)
        net.load_state_dict({k: torch.tensor(v) for k, v in self.state.items()})
        return net.eval()

    def predict(self, context, model_input: pd.DataFrame) -> np.ndarray:
        horizon = _forecast_horizon(self._net(), self.panel, device="cpu")
        return self.panel.predict_frame(horizon, model_input)

    def __call__(self, model_input: pd.DataFrame) -> np.ndarray:
        return self.predict(None, model_input)


class DLinearForecaster(SeqForecaster):

    def __init__(self, panel, net: DLinear):
        super().__init__(panel, net)
