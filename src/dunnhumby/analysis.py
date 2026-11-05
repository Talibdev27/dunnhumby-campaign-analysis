"""Campaign-impact estimates: naive, short-window matched, long-history matched, and placebos.

Run with:  uv run python -m dunnhumby.analysis
"""

import warnings

import pandas as pd

from dunnhumby.data import REPORTS_DIR, load
from dunnhumby.did import estimate, history_features, match, pre_features
from dunnhumby.panel import campaign_panel

TYPE_A = [8, 13, 18]
PLACEBO_SHIFT_DAYS = 70


def _shift(campaigns: pd.DataFrame, campaign: int, days: int) -> pd.DataFrame:
    """Fake campaign that ends the day before the real one starts and lasts `days`."""
    c = campaigns.copy()
    s = c.campaign == campaign
    c.loc[s, "end_day"] = c.loc[s, "start_day"] - 1
    c.loc[s, "start_day"] = c.loc[s, "start_day"] - days
    return c


def run_campaign(t, campaigns, ch, campaign: int) -> list[dict]:
    rows = []
    placebo = _shift(campaigns, campaign, PLACEBO_SHIFT_DAYS)
    for label, camps in [("real", campaigns), ("placebo", placebo)]:
        p = campaign_panel(t, camps, ch, campaign)
        start = int(camps.set_index("campaign").loc[campaign].start_day)
        short = pre_features(p)
        long_ = short.join(history_features(t, p, start))
        weightings = [("naive", None), ("match_8wk", match(short)), ("match_history", match(long_))]
        for method, w in weightings:
            r = estimate(p, w)
            rows.append({
                "campaign": campaign, "period": label, "method": method,
                "att_usd_wk": r.att, "se": r.se,
                "pct": r.pct, "pct_se": 100 * r.se / r.pre_mean_treated,
                "n_treated": r.n_treated, "n_controls": r.n_controls,
            })
    return rows


def main() -> pd.DataFrame:
    warnings.filterwarnings("ignore")
    t, c, ch = load("transactions"), load("campaigns"), load("campaign_households")
    res = pd.DataFrame([row for k in TYPE_A for row in run_campaign(t, c, ch, k)])
    REPORTS_DIR.mkdir(exist_ok=True)
    res.to_csv(REPORTS_DIR / "did_results.csv", index=False)
    print(res.round(2).to_string(index=False))
    return res


if __name__ == "__main__":
    main()
