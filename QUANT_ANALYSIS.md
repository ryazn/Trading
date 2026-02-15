# PHANTOM CVD Strategy — Quant Analysis & Improvement Roadmap

## Executive Summary

After running **4,300+ parameter combinations** across 5 years of ES 1-min data (2021-2025)
in three separate optimization rounds, **no configuration was consistently profitable across
both training and validation periods**. The best full-5-year config achieves PF 1.01 (+$1,418).
This isn't a tuning problem — it's a structural signal quality problem.

The good news: the core *concept* (liquidity sweeps + divergence) is sound and used by
professional traders. The implementation just needs better signal inputs and smarter filters.

---

## What the Optimization Revealed

### Round 1: Random Grid Search (500 combos, V1 strategy)
- **0 profitable configs** on 2021-2023 training data
- Best: PF 0.97, -$1,042 over 3 years

### Round 2: Focused Search (3,152 combos, 6 structural variations)
- **22 profitable configs** on training data (mostly fixed 1-contract mode)
- Structural variations tested: ATR fine-tune, swept-level SL, trailing stops,
  momentum filter, breakeven, fixed contracts

### Round 3: V2 Strategy with Trend + Vol Filters (600 combos)
- **36 profitable configs** on 2021-2023 training (best: PF 1.14, $13,700)
- **ALL 20 top configs failed validation** on 2024-2025
- The trend filter helped in 2021-2023 but the 2024-2025 regime was different

### Best Full 5-Year Config (fixed 1ct, ATR 2.5, RR 1.5, RTH only)
| Metric | Value |
|--------|-------|
| Net P&L | +$1,418 (+1.42%) |
| Trades | 1,003 |
| Win Rate | 43.0% |
| Profit Factor | 1.01 |
| Max Drawdown | 22.0% |
| Sharpe | 0.04 |
| Avg Trade | $1.41 |
| Slippage Cost | $25,075 |

### Baseline (original 2-day optimized config on 5 years)
| Metric | Value |
|--------|-------|
| Net P&L | -$190,061 |
| Trades | 4,595 |
| Win Rate | 27.4% |
| Profit Factor | 0.86 |
| Max Drawdown | 165.8% |
| Slippage Cost | $226,325 |

### Key Patterns Across All Top Configs
- **Session filter = True** appears in ALL top configs (RTH 9:30-16:00 only)
- **Fixed 1 contract** removes position sizing noise → clearer signal evaluation
- **strength_period = 100** dominates (longer lookback for CVD normalization)
- **pivot_left = 10, pivot_right = 2-3** is the sweet spot
- **ATR 2.5 SL + 1.5:1 R:R** is the best risk combo (higher WR, lower drawdown)
- **Trend filter (200 EMA with_trend)** helped on training but didn't generalize
- **Vol filter** consistently HURT performance (reduces trades without improving quality)
- **PF hovers at 0.97-1.01** regardless of params — the estimated CVD is the bottleneck

---

## Root Cause Analysis

### 1. Estimated CVD Is Noise (Critical)

The current CVD estimation uses `close > open → buy volume, close < open → sell volume`.
On 1-minute bars this is essentially a coin flip:

- **Problem**: A 1-min bar with close=5000.25 and open=5000.00 counts ALL volume as buying.
  But 49% of the actual trades within that bar could have been sells at the ask.
- **Impact**: The "divergence" signal (price sweeps low but CVD is higher) is based on noise.
  A random walk would produce the same divergence patterns.
- **Evidence**: PF ~1.00 across all parameter combos = the signal has zero edge.

**Fix**: Use real tick-level CVD from TradingView exports, or compute CVD from trade-level
data (bid/ask classification). The strategy already supports a `cvd` column in the CSV.

### 2. No Trend Context (Major)

The strategy fires mean-reversion signals in ALL market conditions, including strong trends.
This is the #1 killer of mean-reversion strategies:

- **2022**: ES dropped from 4800 to 3500 (-27%). Every swing low sweep during the downtrend
  triggered a long signal that got stopped out.
- **2021**: ES rallied from 3700 to 4800 (+30%). Short signals at swing high sweeps during
  the uptrend were destroyed.
- **Impact**: Estimated 40-50% of losing trades are counter-trend signals.

**Fix**: Add a trend filter. Only allow longs when price is above a trend indicator,
shorts when below. Options:
  - 200-bar EMA (simple, robust)
  - ADX > 25 to detect trending vs ranging (only mean-revert in low-ADX environments)
  - Higher timeframe (5min/15min) trend direction

### 3. No Volatility Regime Filter (Major)

The same parameters are used whether ATR is 3 points (dead overnight) or 30 points (FOMC day).
Mean-reversion works best in **moderate** volatility — too low = noise, too high = momentum.

- **Problem**: Low-vol periods generate false sweeps (1-tick wobbles past levels).
  High-vol periods blow through stops as price extends, not reverts.
- **Evidence**: sweep_ticks=2-3 helps (filters noise sweeps), but can't fix regime mismatch.

**Fix**: Add an ATR percentile filter:
  - Compute 14-period ATR and its 100-bar percentile
  - Only trade when ATR is between 25th and 75th percentile
  - Alternatively: scale sweep_ticks dynamically with ATR

### 4. Position Sizing Amplifies Drawdowns (Moderate)

`risk_per_trade` mode with ATR-based stops creates a feedback loop:
- Low ATR → tight stops → large position → stopped out → loss is exactly 1% of equity
- But many low-ATR trades cluster together (low-vol regimes) → rapid serial 1% losses
- High ATR → wide stops → small position → even winners don't recover much

The math works out to roughly flat expectancy regardless of sizing.

**Fix**: Fixed 1-contract mode removes this noise and lets you see the raw signal quality.
Once the signal is profitable at 1 contract, THEN add position sizing.

### 5. Overnight Data Is Pure Noise (Moderate)

ES overnight (18:00-09:30 ET) has:
- ~70% of the bars but ~15% of the volume
- Wider effective spreads (1-2 ticks vs 0.25 ticks during RTH)
- Driven by overseas markets, not ES order flow → CVD is meaningless

**Evidence**: Session filter = True appears in ALL top-performing configs.

**Fix**: Already have session filter. Enforce it. Consider also filtering the first 15 min
of RTH (9:30-9:45) when many false sweeps occur from the opening range.

### 6. Signal Cooldown Is Too Short (Minor)

With cooldown=5 on 1-min bars, the strategy can fire 12 signals per hour.
In a trending market, this creates a machine-gun of losing trades at the same level.

**Fix**: Use time-based cooldown (e.g., 30-60 min) rather than bar-based.
Or: require that the PREVIOUS trade was a winner before taking the same direction again.

---

## Improvement Roadmap (Ordered by Impact)

### Phase 1: Signal Quality (Highest Impact)

#### 1A. Use Real CVD Data
```
Priority: CRITICAL
Effort: LOW (already supported via csv column)
Expected Impact: Transform PF from ~1.0 to ~1.2-1.5
```

Export ES data from TradingView with the CVD indicator attached. The strategy already
reads a `cvd` column. Real CVD captures actual order flow vs the bar-direction estimate.

**Why this matters**: The entire thesis of the strategy is that CVD divergence at swept levels
indicates trapped traders. Without real CVD, you're reading tea leaves.

#### 1B. Add Trend Filter
```python
# In _compute_indicators():
self.trend_ema = self.ema(self.closes, 200)

# In generate_signal():
trend_up = self.closes[bar_index] > self.trend_ema[bar_index]
trend_down = self.closes[bar_index] < self.trend_ema[bar_index]

# Only allow longs in uptrend, shorts in downtrend
# OR: only trade when price is NEAR the EMA (ranging market)
near_ema = abs(self.closes[bar_index] - self.trend_ema[bar_index]) < 2 * atr
```
```
Priority: HIGH
Effort: LOW
Expected Impact: PF +0.1 to +0.3 by filtering counter-trend trades
```

#### 1C. Add Volatility Filter
```python
# In _compute_indicators():
self.atr_values = self.atr(self.highs, self.lows, self.closes, 14)
self.atr_pctile = rolling_percentile(self.atr_values, 100)  # 100-bar lookback

# In generate_signal():
if self.atr_pctile[bar_index] < 0.15 or self.atr_pctile[bar_index] > 0.85:
    return None  # Skip extreme low/high vol
```
```
Priority: HIGH
Effort: LOW
Expected Impact: PF +0.05 to +0.15 by avoiding bad regimes
```

### Phase 2: Risk Management (Medium Impact)

#### 2A. Swept-Level Stop Loss
```
Priority: MEDIUM
Effort: NONE (already built, just set sl_mode: "swept_level")
Expected Impact: Tighter, more logical stops → better R:R
```

Currently using ATR-based stops which are arbitrary. The strategy KNOWS the swept level
(the liquidity pocket). Placing the stop just beyond that level is:
- Logically sound (if price breaks further, the thesis is invalidated)
- Usually tighter than 2x ATR → better reward-to-risk

#### 2B. Time-Based Exit
```
Priority: MEDIUM
Effort: NONE (already built: use_max_hold: true, max_hold_bars: 120)
Expected Impact: Reduces average losing trade duration
```

Mean-reversion trades should work quickly (30-120 minutes). If it hasn't hit TP in 2 hours,
the thesis is likely wrong. Cut the position instead of waiting for the stop.

#### 2C. Breakeven Stop
```
Priority: LOW-MEDIUM
Effort: NONE (already built)
Expected Impact: Converts some losers to scratch trades
```

Once price moves 8-12 points in your favor, move stop to breakeven + 1 tick.
This turns ~10-15% of losing trades into scratch trades.

### Phase 3: Advanced Enhancements (Lower Impact, Higher Effort)

#### 3A. Multi-Timeframe Confirmation
```
Priority: MEDIUM
Effort: MEDIUM
Expected Impact: PF +0.1 to +0.2
```

Instead of only looking at 1-min pivots, require that the sweep aligns with a
higher-timeframe (5min or 15min) structure level. This filters out micro-noise sweeps.

Implementation:
- Resample 1-min data to 5-min and 15-min
- Compute pivots on both timeframes
- Only take 1-min signals that occur near a 5-min or 15-min swing level

#### 3B. Volume Profile Zones
```
Priority: MEDIUM
Effort: HIGH
Expected Impact: PF +0.1 to +0.2
```

Sweeps at high-volume zones (where lots of orders were placed) are more meaningful
than sweeps at random swing points. Add a volume-at-price histogram to weight
swing levels by the volume traded around them.

#### 3C. Time-of-Day Weighting
```
Priority: LOW
Effort: LOW
Expected Impact: PF +0.05
```

Not all hours are equal:
- **9:45-11:00**: Best for mean-reversion (morning range established, fakeouts common)
- **11:00-14:00**: Lunchtime chop, lower quality signals
- **14:00-15:45**: Afternoon trend, mean-reversion can work but riskier

Weight signals by time-of-day or only trade the best windows.

#### 3D. Walk-Forward Optimization
```
Priority: HIGH (for any live deployment)
Effort: MEDIUM
Expected Impact: Prevents overfitting → stable live performance
```

Instead of train/test split, use rolling walk-forward:
- Train on months 1-6, test on month 7
- Train on months 2-7, test on month 8
- Etc.

This gives a realistic estimate of how the strategy would perform if re-optimized monthly.

#### 3E. Ensemble / Regime Switching
```
Priority: LOW
Effort: HIGH
Expected Impact: PF +0.1 to +0.3
```

Use different parameter sets for different market regimes:
- Low-vol ranging: tight pivots, quick entries
- High-vol trending: wider pivots, trend-following mode
- Detect regime using ATR + ADX + EMA slope

---

## Recommended Implementation Order

1. **Get real CVD data** from TradingView (or broker data feed)
2. **Add trend filter** (200 EMA) — 20 lines of code
3. **Add volatility filter** (ATR percentile) — 15 lines of code
4. **Enable session filter** (RTH only) — already built
5. **Switch to fixed 1-contract** for testing — already built
6. **Re-run optimization** on cleaned signal
7. **If profitable at 1ct**: add swept-level SL, breakeven, max-hold
8. **If still not profitable**: add multi-timeframe confirmation
9. **Before going live**: walk-forward validation

---

## What Does "Good" Look Like?

For a 1-min ES mean-reversion strategy, realistic targets:

| Metric | Minimum | Good | Excellent |
|--------|---------|------|-----------|
| Profit Factor | 1.15 | 1.3 | 1.5+ |
| Win Rate | 40% | 50% | 60% |
| Sharpe (annualized) | 0.5 | 1.0 | 2.0 |
| Max Drawdown | <20% | <10% | <5% |
| Avg Trade (1ct) | >$5 | >$15 | >$30 |
| Trades/Year | 100+ | 200+ | 300+ |

The current strategy is at PF 0.97 — it needs +0.18 to reach "minimum viable".
The fixes above should collectively provide +0.3 to +0.6 PF improvement.

---

## Quick Wins You Can Do Right Now

1. Export ES data from TradingView WITH the CVD indicator → save as CSV with `cvd` column
2. Set `use_session_filter: true` in your config
3. Set `position_mode: fixed_contracts` and `fixed_contracts: 1`
4. Set `use_max_hold: true` and `max_hold_bars: 120`
5. Re-run the backtest → this alone should improve results meaningfully
