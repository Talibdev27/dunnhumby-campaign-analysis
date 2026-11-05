# Did the campaigns work?

Measuring whether a grocery retailer's direct-marketing campaigns actually raised household spend,
using the [dunnhumby *Complete Journey*](https://www.kaggle.com/datasets/frtgnn/dunnhumby-the-complete-journey)
dataset: 2,500 households, 2 years, 2.6M line items, 30 campaigns.

**Status:** work in progress. The spend-lift analysis for the three large TypeA campaigns is done.

## The problem: the retailer picked its best customers

Households that received a campaign spent a median **$3,444** over two years; households that never
did spent **$745**. Targeting wasn't random, so "targeted vs. not" mostly measures *who was chosen*,
not what the campaign did. This is a causal question, not a correlation one.

## Approach

For each campaign:

1. Build a household × week panel of spend: 8 weeks before the start date, plus every campaign week.
2. **Controls** are households not sent this campaign *or any overlapping campaign*, and already
   shopping before the window opens.
3. Match each treated household to a control with a similar propensity to be targeted, then run
   difference-in-differences with household and week fixed effects (SEs clustered by household).
4. **Placebo:** rerun the identical pipeline on a fake campaign 10 weeks *earlier*, when nothing
   was running. A sound method should find ≈ 0 there.

## Results (TypeA campaigns, ~1,100 households each)

Lift in weekly spend, as % of treated households' pre-campaign spend (± 1 SE):

| Campaign | Naive DiD | Matched on 8 weeks | Placebo, 8 weeks | **Matched on 26-week history** | Placebo, history |
|---|---|---|---|---|---|
| 8  | −3.0 ± 1.8 | +11.8 ± 4.8 | +12.8 ± 4.2 | **−3.9 ± 5.8** | +5.3 ± 4.0 |
| 13 | −2.4 ± 1.7 | +12.9 ± 4.7 | +8.5 ± 5.2  | **+5.4 ± 7.5** | +1.0 ± 7.2 |
| 18 | +3.3 ± 1.7 | +15.7 ± 4.8 | +13.4 ± 5.7 | **+3.4 ± 7.0** | −5.1 ± 9.2 |

**What this says:**

- The obvious fix (match on recent spend) produces a convincing-looking **+12–16% lift** that
  *also appears when no campaign is running*. It's regression to the mean: matching on 8 weeks picks
  controls who had an unusually good stretch and then drift back down.
- Matching on long-run history (what the retailer actually targeted on) removes that artifact.
  Placebos land near zero, and the real estimates are **small and not distinguishable from zero**.
- With these sample sizes, any single campaign's lift is pinned down only to roughly ±12–15%
  (95% CI). We can rule out large effects, not small ones.

## Reproduce

```bash
uv sync
uv run python -m dunnhumby.analysis   # writes reports/did_results.csv
uv run pytest
```

The Kaggle CSVs are expected in the project root. First load caches them as parquet in `data/processed/`.

## Next

- Pool campaigns into one stacked estimate to tighten the interval.
- Event-study chart (weekly effects with CIs).
- Coupon redemption: who redeems, and is redemption incremental or just subsidising purchases
  that would have happened anyway?
