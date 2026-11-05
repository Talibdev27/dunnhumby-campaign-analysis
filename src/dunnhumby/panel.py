"""Build household x week panels aligned on a campaign's start day."""

import numpy as np
import pandas as pd

PRE_WEEKS = 8


def campaign_panel(
    transactions: pd.DataFrame,
    campaigns: pd.DataFrame,
    campaign_households: pd.DataFrame,
    campaign: int,
    pre_weeks: int = PRE_WEEKS,
) -> pd.DataFrame:
    """Balanced panel of weekly spend around one campaign.

    rel_week is 7-day bins counted from the campaign's start_day (0 = first campaign week).
    treated = household was sent this campaign. Controls exclude households that were sent
    any *other* campaign overlapping the analysis window, so they are untouched by marketing.
    Only households already shopping before the pre-period are kept, so nobody enters mid-window.
    Weeks with no purchases are filled with zero spend.
    """
    row = campaigns.set_index("campaign").loc[campaign]
    start, end = int(row.start_day), int(row.end_day)
    post_weeks = int(np.ceil((end - start + 1) / 7))
    win_lo, win_hi = start - 7 * pre_weeks, start + 7 * post_weeks - 1

    ch = campaign_households
    treated = set(ch.loc[ch.campaign == campaign, "household_key"])
    overlapping = campaigns[
        (campaigns.campaign != campaign)
        & (campaigns.start_day <= win_hi)
        & (campaigns.end_day >= win_lo)
    ].campaign
    contaminated = set(
        campaign_households.loc[campaign_households.campaign.isin(overlapping), "household_key"]
    )

    first_day = transactions.groupby("household_key").day.min()
    eligible = set(first_day[first_day < win_lo].index)
    controls = eligible - treated - contaminated
    households = sorted((treated & eligible) | controls)

    win = transactions[(transactions.day >= win_lo) & (transactions.day <= win_hi)]
    win = win[win.household_key.isin(households)]
    rel_week = (win.day - start) // 7
    spend = (
        win.assign(rel_week=rel_week)
        .groupby(["household_key", "rel_week"])
        .agg(spend=("sales_value", "sum"), baskets=("basket_id", "nunique"))
    )

    grid = pd.MultiIndex.from_product(
        [households, range(-pre_weeks, post_weeks)], names=["household_key", "rel_week"]
    )
    panel = spend.reindex(grid, fill_value=0).reset_index()
    panel["treated"] = panel.household_key.isin(treated).astype(int)
    panel["post"] = (panel.rel_week >= 0).astype(int)
    panel["campaign"] = campaign
    return panel
