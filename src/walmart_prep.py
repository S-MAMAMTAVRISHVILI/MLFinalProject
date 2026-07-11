"""Shared preprocessing for Walmart Recruiting - Store Sales Forecasting.

Every model notebook imports from here so that all architectures see exactly the
same features, the same validation folds and the same metric.

The central object is `WalmartFeatureBuilder`, a scikit-learn transformer that is
fitted on the raw training frame and can then transform the *raw, unprocessed*
test frame. That makes it dropped straight into a `sklearn.pipeline.Pipeline`,
which is what the assignment requires the registered model to be.

Design constraints that drove this module (see docs/preprocessing_plan.md):

* The test set is a 39-week block that starts the week after training ends.
  Therefore any lag shorter than 39 weeks is unavailable for at least one test
  week. All lag features here use lag >= 39. Nothing is recursive.
* Weekly sales correlate far more strongly with lag-52 (0.983) than with lag-1
  (0.948), so the year-ago window is the backbone of the feature set.
* Weeks end on a Friday but Christmas does not, so the flagged "Christmas week"
  contains 0 / 1 / 3 pre-Christmas shopping days in 2010 / 2011 / 2012. A plain
  lag-52 therefore reads the 2012 holiday week (weight 5) off a trough. The
  `xmas_aligned_lag` feature re-reads the year-ago series at the same distance
  from Christmas, interpolating between the two surrounding weeks.
"""

from __future__ import annotations

import contextlib
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone


@contextlib.contextmanager
def _quiet_nan():
    """Pairs with no history give all-NaN slices; NaN is the answer we want."""
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        yield

__all__ = [
    "ORIGIN",
    "HORIZON",
    "MIN_SAFE_LAG",
    "MARKDOWN_COLS",
    "load_raw",
    "wmae",
    "wmae_weights",
    "week_index",
    "xmas_k",
    "fit_daily_xmas_profile",
    "xmas_week_profile",
    "calendar_features",
    "FOLDS",
    "iter_folds",
    "WalmartFeatureBuilder",
    "SeasonalResidualRegressor",
    "BASE_COLUMNS",
    "seasonal_naive",
    "score_fold",
    "december_shape_report",
    "make_submission",
]

# --------------------------------------------------------------------------
# Constants grounded in the data
# --------------------------------------------------------------------------

ORIGIN = pd.Timestamp("2010-02-05")  # first week-ending date in train.csv
HORIZON = 39                         # test spans 39 weeks after train ends
MIN_SAFE_LAG = HORIZON               # shorter lags are unknown for late test weeks
MARKDOWN_ERA = pd.Timestamp("2011-11-11")  # first week markdowns were recorded

MARKDOWN_COLS = [f"MarkDown{i}" for i in range(1, 6)]

# Week-ending (Friday) dates the competition flags as IsHoliday.
# Event dates are the real calendar dates, used for signed distance features.
SUPERBOWL = pd.to_datetime(["2010-02-07", "2011-02-06", "2012-02-05", "2013-02-03"])
LABOR_DAY = pd.to_datetime(["2010-09-06", "2011-09-05", "2012-09-03", "2013-09-02"])
THANKSGIVING = pd.to_datetime(["2010-11-25", "2011-11-24", "2012-11-22", "2013-11-28"])
CHRISTMAS = pd.to_datetime(["2009-12-25", "2010-12-25", "2011-12-25", "2012-12-25", "2013-12-25"])
EASTER = pd.to_datetime(["2010-04-04", "2011-04-24", "2012-04-08", "2013-03-31"])

# Daily-profile deconvolution: grid of days relative to Dec 25.
XMAS_K_LO, XMAS_K_HI = -60, 20
#: Outside this window the profile is pinned to 1.0. The lower edge stops short of
#: Thanksgiving (which falls at k = -33..-27) so that a Christmas-anchored profile
#: never absorbs a holiday whose distance to Christmas moves from year to year.
XMAS_ACTIVE = (-26, 12)
#: Ordinary weeks used to normalise each season; early November, before Thanksgiving.
XMAS_BASE_K = (-54, -40)


# --------------------------------------------------------------------------
# IO + metric
# --------------------------------------------------------------------------

def load_raw(data_dir: str = "."):
    """Load the four competition CSVs with parsed dates."""
    train = pd.read_csv(f"{data_dir}/train.csv", parse_dates=["Date"])
    test = pd.read_csv(f"{data_dir}/test.csv", parse_dates=["Date"])
    features = pd.read_csv(f"{data_dir}/features.csv", parse_dates=["Date"])
    stores = pd.read_csv(f"{data_dir}/stores.csv")
    return train, test, features, stores


def wmae_weights(is_holiday) -> np.ndarray:
    """Competition sample weights: 5 on holiday weeks, 1 otherwise."""
    return np.where(np.asarray(is_holiday, dtype=bool), 5.0, 1.0)


def wmae(y_true, y_pred, is_holiday) -> float:
    """Weighted mean absolute error, the competition metric."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    w = wmae_weights(is_holiday)
    return float(np.sum(w * np.abs(y_true - y_pred)) / np.sum(w))


def week_index(dates) -> np.ndarray:
    """Integer weeks since ORIGIN. train -> 0..142, test -> 143..181."""
    return ((pd.to_datetime(pd.Series(dates)).values - ORIGIN.to_datetime64())
            / np.timedelta64(7, "D")).astype(int)


# --------------------------------------------------------------------------
# Validation folds
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Fold:
    name: str
    cut: pd.Timestamp          # last training week-ending date (inclusive)
    note: str = ""

    @property
    def val_start(self) -> pd.Timestamp:
        return self.cut + pd.Timedelta(weeks=1)

    @property
    def val_end(self) -> pd.Timestamp:
        return self.cut + pd.Timedelta(weeks=HORIZON)


#: `mirror` occupies the same calendar slot as the test window one year earlier.
#: It is the only 39-week fold whose holiday mix (Thanksgiving + Christmas +
#: Super Bowl) and holiday weight share (29.7% vs the test set's 29.6%) match the
#: real test set. Select models on it. `recent` is a second opinion on the most
#: recent data; `early` mainly guards against overfitting the mirror fold.
FOLDS = [
    Fold("mirror", pd.Timestamp("2011-10-28"), "same calendar slot as test; Thx+Xmas+SB"),
    Fold("recent", pd.Timestamp("2012-01-27"), "most recent 39 weeks; SB+LaborDay only"),
    Fold("early", pd.Timestamp("2011-07-29"), "LaborDay+Thx+Xmas+SB"),
]


def iter_folds(train: pd.DataFrame, folds=None):
    """Yield (fold, train_part, val_part) with a strict time cut."""
    for fold in folds or FOLDS:
        tr = train[train.Date <= fold.cut]
        va = train[(train.Date >= fold.val_start) & (train.Date <= fold.val_end)]
        yield fold, tr, va


# --------------------------------------------------------------------------
# Calendar features
# --------------------------------------------------------------------------

def _signed_days_to_nearest(dates: pd.Series, events: pd.DatetimeIndex, clip: int) -> np.ndarray:
    """Signed days from each date to the nearest event, clipped to +-clip."""
    d = dates.values.astype("datetime64[D]").astype(int)
    e = events.values.astype("datetime64[D]").astype(int)
    diff = d[:, None] - e[None, :]
    nearest = diff[np.arange(len(d)), np.abs(diff).argmin(axis=1)]
    return np.clip(nearest, -clip, clip).astype(np.int16)


def _pre_christmas_days(dates: pd.Series) -> np.ndarray:
    """How many of Dec 22/23/24 fall inside the 7-day window ending on `date`.

    For the flagged Christmas week this is 0 in 2010, 1 in 2011 and 3 in 2012 --
    the single biggest source of year-over-year mismatch in this dataset, and it
    lands on the weight-5 week.
    """
    end = dates.values.astype("datetime64[D]").astype(int)
    start = end - 6
    out = np.zeros(len(end), dtype=np.int8)
    seen = np.unique(dates.dt.year.values)
    # a week ending in early January still looks back at the prior December
    years = sorted(set(seen.tolist()) | set((seen - 1).tolist()))
    for y in years:
        for day in (22, 23, 24):
            c = np.datetime64(f"{y}-12-{day:02d}", "D").astype(int)
            out += ((start <= c) & (c <= end)).astype(np.int8)
    return out


def xmas_k(dates: pd.Series) -> np.ndarray:
    """Signed days from Dec 25 of the season each date belongs to.

    July onward looks ahead to this December; January-June looks back at the last.
    """
    dates = pd.to_datetime(dates)
    ref = np.where(dates.dt.month.values >= 7, dates.dt.year.values, dates.dt.year.values - 1)
    xm = pd.to_datetime([f"{y}-12-25" for y in ref])
    return (dates.values - xm.values).astype("timedelta64[D]").astype(int)


def _week_row(k_end: int, klo: int, khi: int):
    """Row of a design matrix: a week's ratio is the mean of g over its seven days."""
    n = khi - klo + 1
    a = np.zeros(n)
    for j in range(7):
        k = k_end - j
        if not (klo <= k <= khi):
            return None
        a[k - klo] = 1.0 / 7.0
    return a


def fit_daily_xmas_profile(weekly_totals: pd.Series,
                           lam_smooth: float = 1.5, lam_tail: float = 0.6) -> np.ndarray:
    """Recover a daily sales profile g(k) around Christmas from weekly aggregates.

    Weeks end on Friday; Christmas does not. Each season therefore samples the same
    daily shape through a differently-phased 7-day window, and pooling seasons makes
    the daily profile identifiable:  ratio(week) = mean of g over the week's 7 days.

    Smoothness is imposed *within* the pre- and post-Christmas segments but never
    across Dec 25 -- Dec 24 is one of the year's biggest retail days and Dec 25 is
    nearly dead, so a penalty spanning the boundary would erase the very structure
    we are trying to recover. g >= 0 is enforced.

    Weeks containing Thanksgiving are excluded: Thanksgiving sits 27-33 days before
    Christmas depending on the year, so it would otherwise smear into the profile.

    Returns g over `range(XMAS_K_LO, XMAS_K_HI + 1)`; all-ones if there is not
    enough December history to identify it (e.g. a fold that ends before December).
    """
    from scipy.optimize import lsq_linear

    klo, khi = XMAS_K_LO, XMAS_K_HI
    kgrid = np.arange(klo, khi + 1)
    nk = len(kgrid)
    ones = np.ones(nk)

    dates = pd.DatetimeIndex(weekly_totals.index)
    ks = xmas_k(pd.Series(dates))
    ref_year = np.where(dates.month >= 7, dates.year, dates.year - 1)
    thanks = set(THANKSGIVING)

    A, r = [], []
    for ry in np.unique(ref_year):
        m = ref_year == ry
        kk, vv, dd = ks[m], weekly_totals.values[m], dates[m]
        base_m = (kk >= XMAS_BASE_K[0]) & (kk <= XMAS_BASE_K[1])
        dec_m = (kk >= -24) & (kk <= 7)
        if base_m.sum() < 2 or dec_m.sum() < 2:
            continue                      # this season cannot anchor a profile
        base = vv[base_m].mean()
        for k_end, v, d in zip(kk, vv, dd):
            # skip weeks that contain Thanksgiving
            if any(d - pd.Timedelta(days=6) <= th <= d for th in thanks):
                continue
            row = _week_row(int(k_end), klo, khi)
            if row is None:
                continue
            A.append(row); r.append(v / base)

    if len(A) < 6:
        return ones

    D2 = []
    for k in kgrid[:-2]:
        if k <= -3 or k >= 1:             # never span Dec 25
            row = np.zeros(nk)
            row[k - klo], row[k + 1 - klo], row[k + 2 - klo] = 1.0, -2.0, 1.0
            D2.append(row)
    D2 = np.array(D2)
    tail = np.array([1.0 if (k <= -40 or k >= 13) else 0.0 for k in kgrid])

    design = np.vstack([np.array(A), np.sqrt(lam_smooth) * D2,
                        np.sqrt(lam_tail) * np.diag(tail), np.sqrt(1e-3) * np.eye(nk)])
    rhs = np.concatenate([np.array(r), np.zeros(len(D2)),
                          np.sqrt(lam_tail) * tail, np.sqrt(1e-3) * ones])
    try:
        g = lsq_linear(design, rhs, bounds=(0, np.inf)).x
    except Exception:
        return ones
    return g if np.isfinite(g).all() else ones


def xmas_week_profile(dates: pd.Series, g: np.ndarray) -> np.ndarray:
    """Expected level of each week = mean of g over its seven days; 1.0 far from Christmas."""
    klo, khi = XMAS_K_LO, XMAS_K_HI
    ks = xmas_k(dates)
    out = np.ones(len(ks))
    lo, hi = XMAS_ACTIVE
    for i, k_end in enumerate(ks):
        if not (lo <= k_end <= hi):
            continue
        row = _week_row(int(k_end), klo, khi)
        if row is not None:
            out[i] = row @ g
    return out


def calendar_features(dates: pd.Series) -> pd.DataFrame:
    """Deterministic calendar features. Known for every future week."""
    dates = pd.to_datetime(dates)
    iso = dates.dt.isocalendar()
    woy = iso.week.astype(int).values

    out = pd.DataFrame(index=dates.index)
    out["t"] = week_index(dates)
    out["year"] = dates.dt.year.values.astype(np.int16)
    out["month"] = dates.dt.month.values.astype(np.int8)
    out["woy"] = woy.astype(np.int8)
    out["woy_sin"] = np.sin(2 * np.pi * woy / 52.0)
    out["woy_cos"] = np.cos(2 * np.pi * woy / 52.0)

    out["days_to_xmas"] = _signed_days_to_nearest(dates, CHRISTMAS, 40)
    out["days_to_thanksgiving"] = _signed_days_to_nearest(dates, THANKSGIVING, 40)
    out["days_to_superbowl"] = _signed_days_to_nearest(dates, SUPERBOWL, 40)
    out["days_to_laborday"] = _signed_days_to_nearest(dates, LABOR_DAY, 40)
    out["days_to_easter"] = _signed_days_to_nearest(dates, EASTER, 40)

    out["pre_xmas_days"] = _pre_christmas_days(dates)
    # A week "contains" an event when the signed distance sits in (-7, 0].
    for name, col in [
        ("is_xmas_week", "days_to_xmas"),
        ("is_thanksgiving_week", "days_to_thanksgiving"),
        ("is_superbowl_week", "days_to_superbowl"),
        ("is_laborday_week", "days_to_laborday"),
        ("is_easter_week", "days_to_easter"),
    ]:
        out[name] = ((out[col] > -7) & (out[col] <= 0)).astype(np.int8)
    # the week *before* Easter is the real shopping peak, not Easter week itself
    out["is_pre_easter_week"] = ((out["days_to_easter"] > -14) & (out["days_to_easter"] <= -7)).astype(np.int8)
    out["markdown_era"] = (dates.values >= MARKDOWN_ERA.to_datetime64()).astype(np.int8)
    return out


# --------------------------------------------------------------------------
# The feature builder
# --------------------------------------------------------------------------

@dataclass
class _Panel:
    """Dense [n_pairs, n_weeks] matrix of Weekly_Sales, indexed by week number."""
    pair_pos: dict = field(default_factory=dict)
    values: np.ndarray = None
    t_max: int = 0

    def lag(self, pair_idx: np.ndarray, t: np.ndarray, lag: int) -> np.ndarray:
        src = t - lag
        out = np.full(len(t), np.nan)
        ok = (src >= 0) & (src <= self.t_max) & (pair_idx >= 0)
        out[ok] = self.values[pair_idx[ok], src[ok]]
        return out

    def at(self, pair_idx: np.ndarray, t: np.ndarray) -> np.ndarray:
        out = np.full(len(t), np.nan)
        ok = (t >= 0) & (t <= self.t_max) & (pair_idx >= 0)
        out[ok] = self.values[pair_idx[ok], t[ok]]
        return out


class WalmartFeatureBuilder(BaseEstimator, TransformerMixin):
    """Turn raw (Store, Dept, Date, IsHoliday) rows into a model-ready matrix.

    Parameters
    ----------
    features, stores
        The raw `features.csv` / `stores.csv` frames. They are reference data
        covering the full 2010-2013 span, including every test week, so they are
        supplied up front rather than learned.
    lags
        Sales lags in weeks. Must all be >= 39, otherwise the value is unknown
        for the late test weeks and the model would silently train on a signal it
        cannot have at inference time.
    xmas_align_days
        (lo, hi) signed day-distance to Christmas within which the aligned lag is
        computed; outside it the column simply repeats lag-52. The default window
        is the one that helped on the 2010->2011 replay: alignment beat lag-52 at
        offsets -16..+5 days and lost at -23 and +12.
    fill_internal_gaps
        Weeks missing inside a series' observed span are set to 0. Justified: 94.5%
        of the sales adjacent to such gaps are below $500, i.e. the department was
        effectively not selling.
    use_raw_cpi
        CPI and Unemployment drift out of the training range on 54% / 36% of test
        weeks, which tree models cannot extrapolate. By default only the deviation
        from each store's training mean is exposed.
    profile_features
        Emit the deconvolved-Christmas-profile columns. Off by default, and that is
        a measured decision rather than a cautious one. Identifying the Dec 25
        collapse needs at least two seasons of December: with one season the solver
        returns g(Dec 25) = 0.96 instead of 0.56, and the resulting rescaling made
        every fold worse (mirror 1822.5 -> 1932.5). The only fold whose training window
        holds two Decembers -- `recent` -- has no December left in its validation
        window, so the regime the final model runs in cannot be validated at all.
        The profile is still fitted and exposed as `xmas_profile_` for diagnostics;
        see docs/preprocessing_plan.md.
    """

    def __init__(
        self,
        features: pd.DataFrame,
        stores: pd.DataFrame,
        lags=(39, 45, 52, 53, 104),
        xmas_align_days=(-16, 7),
        fill_internal_gaps: bool = True,
        use_raw_cpi: bool = False,
        profile_features: bool = False,
    ):
        self.features = features
        self.stores = stores
        self.lags = tuple(lags)
        self.xmas_align_days = tuple(xmas_align_days)
        self.fill_internal_gaps = fill_internal_gaps
        self.use_raw_cpi = use_raw_cpi
        self.profile_features = profile_features

    # -- fit ---------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y=None):
        bad = [l for l in self.lags if l < MIN_SAFE_LAG]
        if bad:
            raise ValueError(
                f"lags {bad} are shorter than the {HORIZON}-week forecast horizon; "
                "they are unknown for the later test weeks."
            )

        df = X[["Store", "Dept", "Date"]].copy()
        df["Weekly_Sales"] = np.asarray(y, dtype=float) if y is not None else X["Weekly_Sales"].values
        df["t"] = week_index(df.Date)

        pairs = df[["Store", "Dept"]].drop_duplicates().sort_values(["Store", "Dept"])
        self.pair_pos_ = {(s, d): i for i, (s, d) in enumerate(pairs.itertuples(index=False))}
        self.t_max_ = int(df.t.max())

        M = np.full((len(self.pair_pos_), self.t_max_ + 1), np.nan)
        pi = df.set_index(["Store", "Dept"]).index.map(self.pair_pos_).to_numpy()
        M[pi, df.t.values] = df.Weekly_Sales.values

        if self.fill_internal_gaps:
            seen = ~np.isnan(M)
            first = seen.argmax(axis=1)
            last = M.shape[1] - 1 - seen[:, ::-1].argmax(axis=1)
            cols = np.arange(M.shape[1])[None, :]
            inside = (cols >= first[:, None]) & (cols <= last[:, None])
            M[inside & np.isnan(M)] = 0.0

        self.panel_ = _Panel(self.pair_pos_, M, self.t_max_)

        # store- and dept-level aggregates share strength with sparse pairs
        store_of = np.array([s for s, _ in self.pair_pos_])
        dept_of = np.array([d for _, d in self.pair_pos_])
        self.store_panel_ = self._group_panel(M, store_of, np.nansum)
        self.dept_panel_ = self._group_panel(M, dept_of, np.nanmean)

        # per-store CPI / Unemployment baselines, from the training window only
        f = self.features[self.features.Date <= df.Date.max()]
        self.cpi_base_ = f.groupby("Store").CPI.mean()
        self.unemp_base_ = f.groupby("Store").Unemployment.mean()

        # Daily Christmas profile, deconvolved from THIS fold's weeks only, so that
        # a validation fold never sees a December it is about to be scored on.
        # Restricted to series present in every week, otherwise composition changes
        # in the panel would masquerade as seasonality.
        weeks = df.t.nunique()
        cnt = df.groupby(["Store", "Dept"]).size()
        complete = set(cnt[cnt == weeks].index)
        if complete:
            dense = df[[p in complete for p in zip(df.Store, df.Dept)]]
            self.xmas_profile_ = fit_daily_xmas_profile(dense.groupby("Date").Weekly_Sales.sum())
        else:
            self.xmas_profile_ = np.ones(XMAS_K_HI - XMAS_K_LO + 1)
        # how many December seasons went into it -- below 2 the profile is unreliable
        self.n_xmas_seasons_ = int(df.Date.dt.month.isin([12]).groupby(df.Date.dt.year).any().sum())

        self.exog_ = self._build_exog()
        self.static_ = self.stores.set_index("Store")
        self.type_map_ = {t: i for i, t in enumerate(sorted(self.static_.Type.unique()))}

        self.global_median_ = float(np.nanmedian(df.Weekly_Sales.values))

        # Drop degenerate columns, and drop them consistently at transform time.
        # This matters: on the `mirror` fold the training window ends 2011-10-28,
        # before markdowns were ever recorded, so every MarkDown column is a
        # constant zero there; and lag_104 is all-NaN whenever the fold has fewer
        # than 104 weeks of history. Histogram learners cannot bin such columns.
        probe = self._build_frame(X)
        n_distinct = probe.nunique(dropna=True)
        self.dropped_columns_ = sorted(n_distinct[n_distinct < 2].index)
        self.feature_names_ = [c for c in probe.columns if c not in self.dropped_columns_]

        lost = [c for c in BASE_COLUMNS if c in self.dropped_columns_]
        if lost:
            warnings.warn(
                f"seasonal baseline columns {lost} carry no information after fitting on "
                f"{df.t.nunique()} weeks. A lag of {min(self.lags)} weeks needs at least that "
                "many weeks of history; short training windows leave the lag columns empty.",
                RuntimeWarning, stacklevel=2)
        return self

    @staticmethod
    def _group_panel(M, keys, agg):
        uniq = np.unique(keys)
        pos = {k: i for i, k in enumerate(uniq)}
        out = np.full((len(uniq), M.shape[1]), np.nan)
        with _quiet_nan():
            for k in uniq:
                rows = M[keys == k]
                out[pos[k]] = agg(rows, axis=0) if rows.size else np.nan
        return _Panel(pos, out, M.shape[1] - 1)

    def _build_exog(self) -> pd.DataFrame:
        """features.csv, cleaned. Indexed by (Store, t)."""
        f = self.features.copy()
        f = f.sort_values(["Store", "Date"])
        # CPI / Unemployment are absent for the final 13 weeks of the test window
        # (a reporting lag, not a data problem). Carry the last value forward.
        f[["CPI", "Unemployment"]] = f.groupby("Store")[["CPI", "Unemployment"]].ffill()

        # Before 2011-11-11 markdowns were simply not recorded; after that date a
        # blank means "no markdown ran". Zero is the right fill for both, and the
        # markdown_era flag lets a model tell the two regimes apart.
        f[MARKDOWN_COLS] = f[MARKDOWN_COLS].fillna(0.0)
        f["md_total"] = f[MARKDOWN_COLS].sum(axis=1)
        f["md_any"] = (f[MARKDOWN_COLS].abs().sum(axis=1) > 0).astype(np.int8)
        for c in MARKDOWN_COLS + ["md_total"]:
            f[c] = np.sign(f[c]) * np.log1p(np.abs(f[c]))  # heavy right tail

        f["CPI_dev"] = f.CPI - f.Store.map(self.cpi_base_)
        f["Unemployment_dev"] = f.Unemployment - f.Store.map(self.unemp_base_)
        f["t"] = week_index(f.Date)
        keep = (["Store", "t", "Temperature", "Fuel_Price", "CPI_dev", "Unemployment_dev",
                 "md_total", "md_any"] + MARKDOWN_COLS)
        if self.use_raw_cpi:
            keep += ["CPI", "Unemployment"]
        return f[keep].set_index(["Store", "t"])

    # -- transform ---------------------------------------------------------
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._build_frame(X)[self.feature_names_]

    def _build_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X[["Store", "Dept", "Date", "IsHoliday"]].copy().reset_index(drop=True)
        out = calendar_features(df.Date)
        t = out["t"].values
        pi = df.set_index(["Store", "Dept"]).index.map(
            lambda k: self.pair_pos_.get(k, -1)).to_numpy()

        out["Store"] = df.Store.values
        out["Dept"] = df.Dept.values
        out["IsHoliday"] = df.IsHoliday.values.astype(np.int8)
        out["Size"] = df.Store.map(self.static_.Size).values
        out["Type"] = df.Store.map(self.static_.Type).map(self.type_map_).values

        # ---- sales lags (all >= 39 weeks, so no recursion is needed) ----
        for L in self.lags:
            out[f"lag_{L}"] = self.panel_.lag(pi, t, L)

        # ---- windows around the year-ago week ----
        with _quiet_nan():
            near = np.vstack([self.panel_.lag(pi, t, L) for L in (50, 51, 52, 53, 54)])
            out["roll_mean_lag52_w5"] = np.nanmean(near, axis=0)
            out["roll_std_lag52_w5"] = np.nanstd(near, axis=0)
            out["roll_median_lag52_w5"] = np.nanmedian(near, axis=0)

            recent = np.vstack([self.panel_.lag(pi, t, L) for L in range(39, 52)])
            out["roll_mean_lag39_51"] = np.nanmean(recent, axis=0)
            out["roll_std_lag39_51"] = np.nanstd(recent, axis=0)

            level = np.vstack([self.panel_.lag(pi, t, L) for L in range(39, 91, 4)])
            out["level_mean_lag39_90"] = np.nanmean(level, axis=0)

            prior = np.vstack([self.panel_.lag(pi, t, L) for L in range(91, 104)])
            prior_mean = np.nanmean(prior, axis=0)

            # year-over-year growth of the pair, computed entirely from safe lags
            out["yoy_trend"] = np.clip(out["roll_mean_lag39_51"] / prior_mean, 0.2, 5.0)

        # ---- Christmas-aligned lag ----
        out["xmas_aligned_lag"] = self._xmas_aligned_lag(pi, df.Date, out["lag_52"].values)

        # ---- daily-profile correction (opt-in; see the class docstring) ----
        if self.profile_features:
            prof = xmas_week_profile(df.Date, self.xmas_profile_)
            prof_src = xmas_week_profile(df.Date - pd.Timedelta(days=364), self.xmas_profile_)
            out["xmas_profile"] = prof
            out["xmas_profile_ratio"] = np.where(prof_src > 1e-6, prof / prof_src, 1.0)
            out["base_profile_adj"] = out["lag_52"].values * out["xmas_profile_ratio"].values

        # ---- shared-strength aggregates ----
        si = df.Store.map(lambda s: self.store_panel_.pair_pos.get(s, -1)).to_numpy()
        di = df.Dept.map(lambda d: self.dept_panel_.pair_pos.get(d, -1)).to_numpy()
        out["store_lag_52"] = self.store_panel_.lag(si, t, 52)
        out["dept_lag_52"] = self.dept_panel_.lag(di, t, 52)
        with _quiet_nan():
            out["pair_share_of_store"] = out["lag_52"] / out["store_lag_52"]

        # ---- exogenous, all known for the full test horizon ----
        idx = pd.MultiIndex.from_arrays([df.Store.values, t])
        exog = self.exog_.reindex(idx)
        for c in exog.columns:
            out[c] = exog[c].values

        out["Size_per_dept_sale"] = out["Size"] / (out["lag_52"].abs() + 1.0)

        return out.astype({c: "float32" for c in out.columns if out[c].dtype == "float64"})

    def _xmas_aligned_lag(self, pi: np.ndarray, dates: pd.Series, fallback: np.ndarray) -> np.ndarray:
        """Read last year's series at the same *distance from Christmas*.

        A plain lag-52 lands on the same ISO week, but Christmas moves across the
        Friday-ending week boundary. For a target week ending T in year y, the
        matching point one year earlier is `Christmas[y-1] + (T - Christmas[y])`,
        which generally falls between two week-endings; we interpolate linearly.
        """
        dates = pd.to_datetime(dates).reset_index(drop=True)
        yr = np.where(dates.dt.month.values == 12, dates.dt.year.values, dates.dt.year.values - 1)
        xmas_this = pd.to_datetime([f"{y}-12-25" for y in yr])
        xmas_prev = pd.to_datetime([f"{y - 1}-12-25" for y in yr])
        offset = (dates.values - xmas_this.values).astype("timedelta64[D]").astype(int)

        lo_d, hi_d = self.xmas_align_days
        near = (offset >= lo_d) & (offset <= hi_d)
        out = fallback.copy()
        if not near.any():
            return out

        src_point = xmas_prev[near].values.astype("datetime64[D]") \
            + offset[near].astype("timedelta64[D]")
        # the Friday on or before src_point, expressed as a week number
        src = pd.to_datetime(src_point)
        back = (src.dayofweek.values - 4) % 7
        lo_date = src - pd.to_timedelta(back, unit="D")
        alpha = back / 7.0
        t_lo = ((lo_date.values - ORIGIN.to_datetime64()) / np.timedelta64(7, "D")).astype(int)

        v_lo = self.panel_.at(pi[near], t_lo)
        v_hi = self.panel_.at(pi[near], t_lo + 1)
        # alpha weights the *earlier* week because src_point sits after lo_date
        blended = v_lo * (1.0 - alpha) + v_hi * alpha
        blended = np.where(np.isnan(blended), np.where(np.isnan(v_lo), v_hi, v_lo), blended)
        out[near] = np.where(np.isnan(blended), fallback[near], blended)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_, dtype=object)


# --------------------------------------------------------------------------
# Residual learner
# --------------------------------------------------------------------------

#: Coalesced in order to form the seasonal baseline. The first non-null wins.
BASE_COLUMNS = ("xmas_aligned_lag", "lag_52", "roll_mean_lag52_w5",
                "roll_mean_lag39_51", "level_mean_lag39_90")


class SeasonalResidualRegressor(BaseEstimator, RegressorMixin):
    """Fit `estimator` on (y - seasonal baseline) instead of on y itself.

    Regressing the raw sales level makes a single global model spend its capacity
    re-learning that department 92 of store 20 is large -- information the lag-52
    column already carries exactly. It also cannot reproduce holiday spikes, which
    live outside the range any split can extrapolate to.

    Measured on the two 39-week folds (HistGradientBoosting, absolute_error):

        setup                          mirror   recent
        seasonal naive (lag-52)        2037.8   1807.2
        aligned baseline, no model     1911.5   1799.0
        level target                   2210.8   1735.0
        residual target                1822.5   1656.3

    The level model is *worse than doing nothing* on the fold that contains
    Thanksgiving and Christmas. Residualising fixes that: Thanksgiving-week MAE
    drops from 5057 to 2196.
    """

    def __init__(self, estimator, base_columns=BASE_COLUMNS):
        self.estimator = estimator
        self.base_columns = base_columns

    def baseline(self, X: pd.DataFrame) -> np.ndarray:
        cols = [c for c in self.base_columns if c in X.columns]
        if not cols:
            raise ValueError(
                f"none of the baseline columns {self.base_columns} are present in X. "
                "Either the feature builder dropped them all as empty -- which happens when "
                "the training window is shorter than the shortest lag (39 weeks) -- or X was "
                "not produced by WalmartFeatureBuilder.")
        b = X[cols[0]].to_numpy(dtype=float).copy()
        for c in cols[1:]:
            m = ~np.isfinite(b)
            if not m.any():
                break
            b[m] = X[c].to_numpy(dtype=float)[m]
        b[~np.isfinite(b)] = 0.0   # cold-start pairs: nothing to lean on
        return b

    def fit(self, X, y, sample_weight=None):
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, np.asarray(y, dtype=float) - self.baseline(X),
                            sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.baseline(X) + self.estimator_.predict(X)


# --------------------------------------------------------------------------
# Scoring helpers shared by every model notebook
# --------------------------------------------------------------------------

HOLIDAY_LABELS = {
    "2011-11-25": "thanksgiving", "2010-11-26": "thanksgiving",
    "2011-12-30": "christmas", "2010-12-31": "christmas",
    "2012-02-10": "superbowl", "2011-02-11": "superbowl", "2010-02-12": "superbowl",
    "2011-09-09": "laborday", "2012-09-07": "laborday", "2010-09-10": "laborday",
}


def score_fold(y_true, y_pred, val: pd.DataFrame) -> dict:
    """WMAE plus the breakdowns that a single WMAE number hides.

    29.6% of the test set's weight sits on three weeks, so a change that improves
    the overall figure while degrading a holiday week is a bad trade. Log all of it.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    hol = val.IsHoliday.to_numpy(dtype=bool)
    out = {
        "wmae": wmae(y_true, y_pred, hol),
        "mae": float(np.abs(y_true - y_pred).mean()),
        "mae_holiday": float(np.abs(y_true[hol] - y_pred[hol]).mean()) if hol.any() else float("nan"),
        "mae_nonholiday": float(np.abs(y_true[~hol] - y_pred[~hol]).mean()),
        "bias": float((y_pred - y_true).mean()),
    }
    for d, label in HOLIDAY_LABELS.items():
        m = (val.Date == pd.Timestamp(d)).to_numpy()
        if m.any():
            out[f"mae_{label}"] = float(np.abs(y_true[m] - y_pred[m]).mean())
    return out


def december_shape_report(train: pd.DataFrame, test: pd.DataFrame,
                          y_pred: np.ndarray, g: np.ndarray):
    """Compare a model's predicted December 2012 shape against the daily profile.

    Every December test week has a (days_to_christmas, pre_xmas_days) geometry that
    never occurs in training, so NO validation fold can catch a model that gets
    December wrong. This is the only available check. Both scales are normalised on
    the same early-November weeks so the ratios are comparable.

    Returns (frame, verdict). `verdict["passed"]` is False when the predicted
    December peak lands on the wrong week, or when the weight-5 Christmas week is
    off by more than the profile's own leave-one-season-out uncertainty (~6%).
    """
    cnt = train.groupby(["Store", "Dept"]).size()
    complete = set(cnt[cnt == train.Date.nunique()].index)
    mask = np.array([p in complete for p in zip(test.Store, test.Dept)])

    totals = pd.Series(np.asarray(y_pred)[mask]).groupby(test.Date.values[mask]).sum()
    k = xmas_k(pd.Series(totals.index))
    base_m = (k >= XMAS_BASE_K[0]) & (k <= XMAS_BASE_K[1])
    if base_m.sum() == 0:
        raise ValueError("no early-November weeks in the prediction window to normalise on")
    base = totals[base_m].mean()

    lo, hi = XMAS_ACTIVE
    weeks = [d for d in totals.index if lo <= xmas_k(pd.Series([d]))[0] <= hi]
    implied = xmas_week_profile(pd.Series(weeks), g)
    predicted = (totals[weeks] / base).to_numpy()

    frame = pd.DataFrame({
        "week": [d.date() for d in weeks],
        "k_end": [int(xmas_k(pd.Series([d]))[0]) for d in weeks],
        "predicted": predicted,
        "implied": implied,
        "gap_pct": 100 * (predicted / implied - 1),
    })

    xmas = pd.Timestamp("2012-12-28")
    i = weeks.index(xmas) if xmas in weeks else int(np.argmin(np.abs(frame.k_end - 3)))
    peak_pred, peak_impl = weeks[int(np.argmax(predicted))], weeks[int(np.argmax(implied))]
    gap = float(frame.gap_pct.iloc[i])

    problems = []
    if peak_pred != peak_impl:
        problems.append(f"December peak at {peak_pred.date()}, should be {peak_impl.date()}")
    if abs(gap) > 10:
        problems.append(f"Christmas week off by {gap:+.1f}%, beyond the profile's uncertainty")

    verdict = {
        "passed": not problems,
        "problems": problems,
        "xmas_gap_pct": gap,
        "peak_predicted": str(peak_pred.date()),
        "peak_implied": str(peak_impl.date()),
    }
    return frame, verdict


# --------------------------------------------------------------------------
# Reference baseline
# --------------------------------------------------------------------------

def seasonal_naive(train_part: pd.DataFrame, target: pd.DataFrame, lag_weeks: int = 52) -> np.ndarray:
    """Predict each row with the same (Store, Dept) value `lag_weeks` earlier.

    Falls back to the pair median, then the global median. This is the number every
    model must beat: WMAE 2037.8 on `mirror`, 1807.2 on `recent`, 2018.6 on `early`.
    """
    key = train_part.set_index(["Store", "Dept", "Date"]).Weekly_Sales
    src = target.Date - pd.Timedelta(weeks=lag_weeks)
    p = key.reindex(pd.MultiIndex.from_arrays([target.Store, target.Dept, src])).to_numpy(dtype=float)

    med = train_part.groupby(["Store", "Dept"]).Weekly_Sales.median()
    fb = target.set_index(["Store", "Dept"]).index.map(med).to_numpy(dtype=float)
    fb = np.where(np.isfinite(fb), fb, float(train_part.Weekly_Sales.median()))
    return np.where(np.isfinite(p), p, fb)


def make_submission(test: pd.DataFrame, y_pred: np.ndarray, path=None) -> pd.DataFrame:
    """Build the `Store_Dept_Date` submission frame in sampleSubmission row order."""
    sub = pd.DataFrame({
        "Id": (test.Store.astype(str) + "_" + test.Dept.astype(str) + "_"
               + test.Date.dt.strftime("%Y-%m-%d")),
        "Weekly_Sales": np.asarray(y_pred, dtype=float),
    })
    if path is not None:
        import pathlib
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        sub.to_csv(p, index=False)
    return sub
