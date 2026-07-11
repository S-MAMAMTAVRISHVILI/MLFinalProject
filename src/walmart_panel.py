from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from walmart_prep import (
    HORIZON, ORIGIN, calendar_features, seasonal_naive, week_index,
)

__all__ = ["WalmartPanel", "PanelWindows", "PrecomputedForecaster"]


@dataclass
class PanelWindows:
    past: np.ndarray
    future: np.ndarray
    series_idx: np.ndarray
    origin_t: np.ndarray

    def __len__(self) -> int:
        return len(self.past)


class WalmartPanel:

    def __init__(self, train: pd.DataFrame, features: pd.DataFrame, stores: pd.DataFrame,
                 lookback: int = 52, horizon: int = HORIZON, target_mode: str = "level"):
        self.train_full = train
        self.features = features
        self.stores = stores
        self.lookback = int(lookback)
        self.horizon = int(horizon)
        if target_mode not in ("level", "residual"):
            raise ValueError("target_mode must be 'level' or 'residual'")
        self.target_mode = target_mode
        self.season = 52


    def fit(self, cutoff: pd.Timestamp | str | None = None) -> "WalmartPanel":
        df = self.train_full
        self.cutoff_ = pd.Timestamp(df.Date.max() if cutoff is None else cutoff)
        df = df[df.Date <= self.cutoff_]

        n_weeks = int(week_index([self.cutoff_])[0]) + 1
        if self.lookback + self.horizon > n_weeks:
            raise ValueError(
                f"lookback {self.lookback} + horizon {self.horizon} = "
                f"{self.lookback + self.horizon} exceeds the {n_weeks} weeks available up to "
                f"{self.cutoff_.date()}; no training window would fit.")

        pairs = (df[["Store", "Dept"]].drop_duplicates()
                   .sort_values(["Store", "Dept"]).to_records(index=False))
        self.keys_ = [(int(s), int(d)) for s, d in pairs]
        self.key_pos_ = {k: i for i, k in enumerate(self.keys_)}
        self.n_series_ = len(self.keys_)
        self.n_weeks_ = n_weeks

        t = week_index(df.Date)
        pi = df.set_index(["Store", "Dept"]).index.map(self.key_pos_).to_numpy()
        values = np.full((self.n_series_, n_weeks), np.nan)
        values[pi, t] = df.Weekly_Sales.to_numpy()

        self.mask_ = ~np.isnan(values)
        seen = self.mask_
        first = seen.argmax(axis=1)
        last = n_weeks - 1 - seen[:, ::-1].argmax(axis=1)
        has_any = seen.any(axis=1)
        cols = np.arange(n_weeks)[None, :]
        inside = (cols >= first[:, None]) & (cols <= last[:, None]) & has_any[:, None]
        values = np.where(inside & np.isnan(values), 0.0, values)
        self.values_ = values
        self.first_t_, self.last_t_, self.has_any_ = first, last, has_any

        self.mean_ = np.zeros(self.n_series_)
        self.std_ = np.ones(self.n_series_)
        for i in range(self.n_series_):
            obs = values[i, self.mask_[i]]
            if obs.size >= 2 and np.nanstd(obs) > 1e-6:
                self.mean_[i] = float(np.nanmean(obs))
                self.std_[i] = float(np.nanstd(obs))
            elif obs.size >= 1:
                self.mean_[i] = float(np.nanmean(obs))

        span = pd.to_datetime(ORIGIN) + pd.to_timedelta(
            np.arange(n_weeks + self.horizon) * 7, unit="D")
        self.calendar_ = calendar_features(pd.Series(span)).reset_index(drop=True)
        self.calendar_dates_ = span

        hol_dates = set(self.features.loc[self.features.IsHoliday, "Date"])
        self.holiday_week_ = np.array([d in hol_dates for d in span], dtype=bool)
        return self

    def horizon_holiday_weights(self, origin_t: np.ndarray) -> np.ndarray:
        H = self.horizon
        steps = origin_t[:, None] + 1 + np.arange(H)[None, :]
        return np.where(self.holiday_week_[steps], 5.0, 1.0).astype(np.float32)

    def normalise(self, series_idx: np.ndarray, values: np.ndarray) -> np.ndarray:
        return (values - self.mean_[series_idx][:, None]) / self.std_[series_idx][:, None]

    def denormalise(self, series_idx: np.ndarray, values: np.ndarray) -> np.ndarray:
        return values * self.std_[series_idx][:, None] + self.mean_[series_idx][:, None]

    def make_windows(self, stride: int = 1, require_full_observed: bool = True) -> PanelWindows:
        L, H, S = self.lookback, self.horizon, self.season
        residual = self.target_mode == "residual"
        past, target, sidx, origin = [], [], [], []
        for i in range(self.n_series_):
            if not self.has_any_[i]:
                continue
            lo, hi = self.first_t_[i], self.last_t_[i]
            for start in range(lo, hi - L - H + 2, stride):
                h0 = start + L
                if require_full_observed and not self.mask_[i, start:h0 + H].all():
                    continue
                past_raw = self.values_[i, start:h0]
                future_raw = self.values_[i, h0:h0 + H]
                if residual:
                    b0 = h0 - S
                    if b0 < lo or not self.mask_[i, b0:b0 + H].all():
                        continue
                    tgt = (future_raw - self.values_[i, b0:b0 + H]) / self.std_[i]
                else:
                    tgt = (future_raw - self.mean_[i]) / self.std_[i]
                past.append((past_raw - self.mean_[i]) / self.std_[i])
                target.append(tgt)
                sidx.append(i)
                origin.append(h0 - 1)
        if not past:
            raise RuntimeError("no training windows produced; lookback likely too long")
        return PanelWindows(np.asarray(past, dtype=np.float32),
                            np.asarray(target, dtype=np.float32),
                            np.asarray(sidx), np.asarray(origin))

    def forecast_baseline(self) -> np.ndarray:
        H, S = self.horizon, self.season
        end = self.n_weeks_ - 1
        b0 = end + 1 - S
        out = np.full((self.n_series_, H), np.nan, dtype=np.float32)
        if b0 < 0:
            return out
        for i in range(self.n_series_):
            if self.has_any_[i] and self.mask_[i, b0:b0 + H].all():
                out[i] = self.values_[i, b0:b0 + H]
        return out

    def forecast_inputs(self):
        L = self.lookback
        past = np.full((self.n_series_, L), np.nan, dtype=np.float32)
        usable = np.zeros(self.n_series_, dtype=bool)
        end = self.n_weeks_ - 1
        for i in range(self.n_series_):
            if not self.has_any_[i] or self.last_t_[i] < end:
                continue
            span = slice(end - L + 1, end + 1)
            if span.start < 0 or not self.mask_[i, span].all():
                continue
            past[i] = self.values_[i, span]
            usable[i] = True
        idx = np.arange(self.n_series_)
        past[usable] = self.normalise(idx[usable], past[usable])
        return past, usable

    def forecast_dates(self) -> pd.DatetimeIndex:
        start = self.cutoff_ + pd.Timedelta(weeks=1)
        return pd.to_datetime([start + pd.Timedelta(weeks=k) for k in range(self.horizon)])

    def series_history(self, i: int) -> np.ndarray:
        return self.values_[i, self.first_t_[i]:self.last_t_[i] + 1].astype(float)

    def predict_frame(self, horizon_pred: np.ndarray, target: pd.DataFrame) -> np.ndarray:
        dates = self.forecast_dates()
        date_pos = {d: j for j, d in enumerate(dates)}
        out = np.full(len(target), np.nan)
        st = target.Store.to_numpy(); dp = target.Dept.to_numpy()
        tdt = pd.to_datetime(target.Date).to_numpy()
        for r in range(len(target)):
            i = self.key_pos_.get((int(st[r]), int(dp[r])))
            j = date_pos.get(pd.Timestamp(tdt[r]))
            if i is not None and j is not None:
                out[r] = horizon_pred[i, j]
        miss = ~np.isfinite(out)
        if miss.any():
            fb = seasonal_naive(self.train_full[self.train_full.Date <= self.cutoff_],
                                target)
            out[miss] = fb[miss]
        return out


class PrecomputedForecaster:

    def __init__(self, panel: "WalmartPanel", horizon_pred: np.ndarray):
        self.panel = panel
        self.horizon_pred = np.asarray(horizon_pred, dtype=float)

    def predict(self, context, model_input: pd.DataFrame) -> np.ndarray:
        return self.panel.predict_frame(self.horizon_pred, model_input)

    def __call__(self, model_input: pd.DataFrame) -> np.ndarray:
        return self.predict(None, model_input)
