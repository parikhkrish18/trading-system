# Benchmark results: does this model beat doing nothing?

Measured 2026-08-22 on feature set `v4`, 503 S&P 500 names, 10 expanding
walk-forward folds with the purge gap, shared fold boundaries across
horizons, test windows spanning 2024-10 to 2026-07.

**Answer: no. There is no demonstrated edge over buy-and-hold, in either
target mode, at either horizon.**

## What was wrong before

The harness reported cost-adjusted return per trade with nothing to compare
it against. Those numbers were positive, so the system looked like it
worked. It did not: it was measuring market drift and reporting it as skill.
94.6% of its trades were longs in a rising market.

| Horizon | Model, net of costs | Buy-and-hold, same windows | Excess |
|--------:|--------------------:|---------------------------:|-------:|
| 5d      | +0.12%              | +0.29%                     | −0.17% |
| 10d     | +0.31%              | +0.55%                     | −0.24% |
| 20d     | +0.71%              | +1.13%                     | −0.42% |
| 40d     | +1.97%              | +2.26%                     | −0.29% |

Reproduced independently from the cached predictions and from MLflow before
any code was changed. The benchmark is the equal-weight mean forward return
of every candidate row in the same test window — gross of costs, while the
model pays its round trip, which makes it the harder comparison to win.

## The four runs

`benchmark` is identical within a horizon by construction. `excess` =
`model_net` − `benchmark`. All p-values are one-sample t-tests of the
per-fold values against zero.

| Horizon | Mode | Benchmark | Model net | Excess | Folds + | p | % long |
|--------:|:-----|----------:|----------:|-------:|--------:|--:|-------:|
| 20d | absolute | +1.15% | +0.81% | **−0.34%** | 0/10 | 0.14 | 96.6% |
| 20d | relative | +1.15% | −0.03% | **−1.18%** | 4/10 | 0.14 | 34.6% |
| 5d  | absolute | +0.28% | +0.09% | **−0.19%** | 2/10 | 0.21 | 93.2% |
| 5d  | relative | +0.28% | +0.01% | **−0.27%** | 4/10 | 0.27 | 34.9% |

Paired fold-by-fold, relative minus absolute on excess: −0.84% at 20d
(p=0.18), −0.08% at 5d (p=0.62). Relative is not better than absolute on
this metric, and the difference is not distinguishable from noise either.

### What the relative target did change

It fixed the diagnosis even though it did not produce an edge. The absolute
model took 96.6% of its trades long because a rising market makes almost
every absolute forecast positive. The relative model splits ~35/65
long/short, which is what a genuine cross-sectional ranking looks like. Its
directional accuracy against its own objective is 51.9% — the same coin flip
as before — while its accuracy at calling absolute direction drops to 47.9%,
exactly as expected once the drift term is removed.

So the model was never learning stock selection. Removing the drift did not
reveal a hidden signal underneath; it revealed that there was nothing
underneath.

## Long-only, and why it is not a result

Production runs `ALLOW_SHORTS=false` (set on separate evidence, before this
experiment ran), so the operationally relevant slice is the long side alone.

| Horizon | Mode | Long-only excess | Folds + | p |
|--------:|:-----|-----------------:|--------:|--:|
| 5d  | relative | **+0.19%** | 9/10 | 0.021 |
| 20d | relative | **+0.72%** | 6/10 | 0.129 |
| 5d  | absolute | −0.07% | 3/10 | 0.145 |
| 20d | absolute | −0.12% | 1/10 | 0.067 |

The 5d relative cell is the only positive, nominally significant number
anywhere in this work. It does not survive contact with three checks:

1. **Multiplicity.** Eight configurations were examined. If every null were
   true, the chance of at least one p<0.05 is 34%. Bonferroni threshold is
   p<0.0063. Under Benjamini-Hochberg at 5%, **nothing survives** — not one
   of the eight.

2. **Cost.** Every figure uses the spread-only 2bp round trip with no market
   impact modelled. The +19bp edge at 5d becomes +11bp (p=0.14) at a 10bp
   round trip, +1bp (p=0.89) at 20bp, and negative at 30bp. An edge that
   disappears at realistic transaction costs is not an edge. The 20d cell is
   less cost-sensitive but never reaches significance at any cost level.

3. **It is not the book that would be traded.** The harness's "long book" is
   every row with a positive prediction — about 176 of 503 names. Production
   holds the top 10 by conviction. These numbers do not measure the
   portfolio that would actually be held.

## Statistical caveats

- **Overlapping windows.** Consecutive rows share almost their entire
  forward-return window, so row-level observations are nowhere near
  independent and row-level p-values would be badly over-optimistic. Every
  test here is paired at the fold level instead. Even those are somewhat
  optimistic: adjacent folds are adjacent in time and share market regimes,
  and 10 folds is a small sample.
- **Survivorship bias.** The universe is today's S&P 500 membership.
  Companies that dropped out of the index — usually after performing badly —
  are absent from the entire history. This inflates the benchmark and any
  long-heavy strategy alike.
- **No tuning was performed.** These are the first and only runs of each
  configuration. Nothing was adjusted after seeing a result.

## Recommendation

Do not put capital behind this model on the strength of these numbers. The
honest summary is that the system has no measured stock-selection skill: its
apparent profitability was the market's, and correcting the prediction
target to remove the market did not uncover an edge beneath it.

The infrastructure is now sound enough to detect an edge if one appears —
that is what changed. Any future feature work should be judged on
`excess_return` from the first run, never on raw return.
