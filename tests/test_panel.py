import numpy as np
import pandas as pd

from dunnhumby.did import estimate
from dunnhumby.panel import campaign_panel


def _toy(effect: float = 0.0):
    """5 treated + 5 control households buying $10 every day from day 1 to 200.

    Treated households spend `effect` extra per day once campaign 1 starts (day 100).
    Household 99 got campaign 2, which overlaps the window, so it must not be a control.
    """
    rows = []
    for hh in list(range(10)) + [99]:
        for day in range(1, 201):
            if hh == 7 and day % 7 == 0:
                continue  # leave gaps so zero-filling is exercised
            value = 10.0 + (effect if hh < 5 and 100 <= day <= 141 else 0.0)
            rows.append((hh, hh * 1000 + day, day, value))
    t = pd.DataFrame(rows, columns=["household_key", "basket_id", "day", "sales_value"])
    campaigns = pd.DataFrame({"campaign": [1, 2], "start_day": [100, 90], "end_day": [141, 120]})
    ch = pd.DataFrame({"campaign": [1] * 5 + [2], "household_key": [0, 1, 2, 3, 4, 99]})
    return t, campaigns, ch


def test_panel_is_balanced_and_excludes_contaminated_controls():
    t, c, ch = _toy()
    p = campaign_panel(t, c, ch, campaign=1, pre_weeks=4)
    assert 99 not in set(p.household_key)
    assert p.groupby("household_key").size().nunique() == 1
    assert set(p.rel_week) == set(range(-4, 6))
    assert p.treated.sum() == 5 * 10


def test_did_recovers_known_effect():
    t, c, ch = _toy(effect=2.0)
    p = campaign_panel(t, c, ch, campaign=1, pre_weeks=4)
    r = estimate(p)
    # +$2/day for 42 campaign days, spread over 6 post weeks = $14/week
    assert np.isclose(r.att, 14.0, atol=1e-6)
