"""Matched difference-in-differences and event-study estimates for one campaign panel."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def pre_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-household behaviour in the pre-period only (never uses post-period data)."""
    pre = panel[panel.rel_week < 0]
    half = pre.rel_week.min() // 2
    g = pre.groupby("household_key")
    feats = pd.DataFrame({
        "log_spend": np.log1p(g.spend.mean()),
        "log_baskets": np.log1p(g.baskets.mean()),
        "active_share": g.spend.apply(lambda s: (s > 0).mean()),
        "log_spend_early": np.log1p(pre[pre.rel_week < half].groupby("household_key").spend.mean()),
        "log_spend_late": np.log1p(pre[pre.rel_week >= half].groupby("household_key").spend.mean()),
        "treated": g.treated.first(),
    })
    return feats


def history_features(
    transactions: pd.DataFrame, panel: pd.DataFrame, start_day: int, weeks: int = 26
) -> pd.DataFrame:
    """Long-run behaviour in the `weeks` before the panel's pre-period.

    Matching only on the short pre-period picks controls who had an unusually good few weeks;
    they regress back down after the start date and fake a positive effect. Long-run spend is
    what the retailer actually targeted on, so matching on it removes that bias.
    """
    pre_weeks = -int(panel.rel_week.min())
    hi = start_day - 7 * pre_weeks - 1
    lo = hi - 7 * weeks + 1
    h = transactions[
        (transactions.day >= lo)
        & (transactions.day <= hi)
        & transactions.household_key.isin(panel.household_key.unique())
    ]
    g = h.groupby("household_key")
    out = pd.DataFrame({
        "log_hist_spend": np.log1p(g.sales_value.sum() / weeks),
        "log_hist_baskets": np.log1p(g.basket_id.nunique() / weeks),
    })
    return out.reindex(panel.household_key.unique(), fill_value=0.0)


def match(feats: pd.DataFrame, caliper: float = 0.05) -> pd.Series:
    """1:1 nearest-neighbour propensity matching with replacement.

    Returns a weight per household: treated households inside the caliper get 1, each control
    gets the number of times it was used. Households with weight 0 drop out of the estimate.
    """
    x = feats.drop(columns="treated")
    y = feats.treated
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(x, y)
    ps = pd.Series(model.predict_proba(x)[:, 1], index=feats.index)
    logit = np.log(ps / (1 - ps))

    t_idx, c_idx = y[y == 1].index, y[y == 0].index
    nn = NearestNeighbors(n_neighbors=1).fit(logit[c_idx].to_numpy().reshape(-1, 1))
    dist, pos = nn.kneighbors(logit[t_idx].to_numpy().reshape(-1, 1))
    ok = dist[:, 0] <= caliper * logit.std()

    w = pd.Series(0.0, index=feats.index)
    w[t_idx[ok]] = 1.0
    used = pd.Series(c_idx[pos[ok, 0]]).value_counts()
    w[used.index] += used.to_numpy()
    w.attrs["propensity"] = ps
    return w


@dataclass
class DiDResult:
    att: float  # $/household/week
    se: float
    pre_mean_treated: float
    n_treated: int
    n_controls: int
    event: pd.DataFrame  # rel_week, coef, se

    @property
    def pct(self) -> float:
        return 100 * self.att / self.pre_mean_treated


def _twfe(panel: pd.DataFrame, regressors: pd.DataFrame, weights: pd.Series):
    """Household + week fixed effects via within-household demeaning, SEs clustered by household."""
    y = panel.spend - panel.groupby("household_key").spend.transform("mean")
    weeks = pd.get_dummies(panel.rel_week, prefix="w", drop_first=True, dtype=float)
    x = pd.concat([regressors, weeks], axis=1)
    x = x - x.groupby(panel.household_key).transform("mean")
    fit = sm.WLS(y, x, weights=weights).fit(
        cov_type="cluster", cov_kwds={"groups": panel.household_key.to_numpy()}
    )
    return fit


def estimate(panel: pd.DataFrame, weights: pd.Series | None = None) -> DiDResult:
    """Pooled DiD (treated x post) and event study (treated x each week, week -1 as reference)."""
    if weights is None:
        weights = pd.Series(1.0, index=panel.household_key.unique())
    keep = weights[weights > 0].index
    p = panel[panel.household_key.isin(keep)].reset_index(drop=True)
    w = p.household_key.map(weights)

    did = _twfe(p, pd.DataFrame({"did": p.treated * p.post}), w)

    weeks = [k for k in sorted(p.rel_week.unique()) if k != -1]
    ev_x = pd.DataFrame({f"ev_{k}": p.treated * (p.rel_week == k) for k in weeks}).astype(float)
    ev = _twfe(p, ev_x, w)
    event = pd.DataFrame({
        "rel_week": weeks,
        "coef": [ev.params[f"ev_{k}"] for k in weeks],
        "se": [ev.bse[f"ev_{k}"] for k in weeks],
    })
    event = pd.concat([event, pd.DataFrame({"rel_week": [-1], "coef": [0.0], "se": [0.0]})])

    pre_t = p[(p.treated == 1) & (p.post == 0)].spend.mean()
    return DiDResult(
        att=did.params["did"],
        se=did.bse["did"],
        pre_mean_treated=pre_t,
        n_treated=int(p[p.treated == 1].household_key.nunique()),
        n_controls=int(p[p.treated == 0].household_key.nunique()),
        event=event.sort_values("rel_week").reset_index(drop=True),
    )
