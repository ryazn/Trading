"""
PHANTOM CVD Divergence Strategy V2 — Enhanced with quant filters.

Improvements over V1:
1. Trend filter (200 EMA) — only mean-revert in ranging/with-trend conditions
2. Volatility filter (ATR percentile) — skip extreme low/high vol regimes
3. Dynamic sweep threshold — scales with ATR instead of fixed ticks
"""

import numpy as np
from typing import Optional
from strategies.base_strategy import BaseStrategy


class PhantomCVDStrategyV2(BaseStrategy):
    """
    PHANTOM CVD V2 — Liquidity sweep + CVD divergence with trend & vol filters.
    """

    DEFAULT_PARAMS = {
        # CVD
        "cvd_smooth": 1,
        "cvd_anchor_bars": 390,
        # Swing Detection
        "pivot_left": 10,
        "pivot_right": 2,
        "max_level_age": 100,
        "max_levels": 30,
        # Sweep Detection
        "sweep_ticks": 2,
        "tick_size": 0.25,
        # Strength Filter
        "min_strength": 0.3,
        "strength_period": 100,
        # CVD Momentum
        "use_momentum_filter": False,
        "momentum_bars": 5,
        "momentum_tolerance": 0.02,
        # Signal Control
        "signal_cooldown": 5,
        # Direction
        "allow_longs": True,
        "allow_shorts": True,
        # --- V2: Trend Filter ---
        "use_trend_filter": True,
        "trend_ema_period": 200,
        "trend_mode": "with_trend",  # "with_trend", "ranging", "near_ema"
        "trend_atr_band": 3.0,  # For "near_ema": only trade within N*ATR of EMA
        # --- V2: Volatility Filter ---
        "use_vol_filter": True,
        "vol_atr_period": 14,
        "vol_lookback": 200,  # Percentile lookback
        "vol_min_pctile": 0.15,  # Skip below this ATR percentile
        "vol_max_pctile": 0.85,  # Skip above this ATR percentile
        # --- V2: Dynamic Sweep Threshold ---
        "use_dynamic_sweep": False,
        "dynamic_sweep_atr_frac": 0.1,  # Sweep threshold = fraction of ATR
    }

    def __init__(self, params: Optional[dict] = None):
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__(merged)

        # Pre-computed arrays
        self.cvd: Optional[np.ndarray] = None
        self.cvd_highest: Optional[np.ndarray] = None
        self.cvd_lowest: Optional[np.ndarray] = None
        self.pivot_highs: Optional[np.ndarray] = None
        self.pivot_lows: Optional[np.ndarray] = None

        # V2: Filter arrays
        self.trend_ema_values: Optional[np.ndarray] = None
        self.atr_values: Optional[np.ndarray] = None
        self.atr_percentile: Optional[np.ndarray] = None

        # Runtime state
        self.swing_hi_price: list[float] = []
        self.swing_hi_cvd: list[float] = []
        self.swing_hi_bar: list[int] = []
        self.swing_lo_price: list[float] = []
        self.swing_lo_cvd: list[float] = []
        self.swing_lo_bar: list[int] = []

        self.last_bull_bar: int = -9999
        self.last_bear_bar: int = -9999
        self._pivots_processed_to: int = 0

    def _compute_indicators(self):
        """Pre-compute all indicators."""
        p = self.params
        n = len(self.closes)

        # --- CVD (same as V1) ---
        if self.data is not None and "cvd" in self.data.columns:
            cvd_raw = self.data["cvd"].values.astype(float)
            for i in range(len(cvd_raw)):
                if np.isnan(cvd_raw[i]):
                    cvd_raw[i] = cvd_raw[i - 1] if i > 0 else 0.0
            self._using_real_cvd = True
        else:
            delta = np.zeros(n)
            for i in range(n):
                if self.closes[i] > self.opens[i]:
                    delta[i] = self.volumes[i]
                elif self.closes[i] < self.opens[i]:
                    delta[i] = -self.volumes[i]
                else:
                    if i > 0 and self.closes[i] > self.closes[i - 1]:
                        delta[i] = self.volumes[i]
                    elif i > 0 and self.closes[i] < self.closes[i - 1]:
                        delta[i] = -self.volumes[i]
                    else:
                        delta[i] = 0

            timestamps = self.data.index if self.data is not None else None
            cvd_raw = np.zeros(n)
            cumulative = 0.0
            for i in range(n):
                if timestamps is not None and i > 0:
                    if timestamps[i].date() != timestamps[i - 1].date():
                        cumulative = 0.0
                elif timestamps is None:
                    anchor = p["cvd_anchor_bars"]
                    if anchor > 0 and i % anchor == 0:
                        cumulative = 0.0
                cumulative += delta[i]
                cvd_raw[i] = cumulative
            self._using_real_cvd = False

        if p["cvd_smooth"] > 1:
            self.cvd = self.ema(cvd_raw, p["cvd_smooth"])
            for i in range(len(self.cvd)):
                if np.isnan(self.cvd[i]):
                    self.cvd[i] = cvd_raw[i]
                else:
                    break
        else:
            self.cvd = cvd_raw

        self.cvd_highest = self.highest(self.cvd, p["strength_period"])
        self.cvd_lowest = self.lowest(self.cvd, p["strength_period"])

        # --- Pivot points ---
        self.pivot_highs = self.pivot_high(self.highs, p["pivot_left"], p["pivot_right"])
        self.pivot_lows = self.pivot_low(self.lows, p["pivot_left"], p["pivot_right"])

        # --- V2: Trend EMA ---
        if p["use_trend_filter"]:
            self.trend_ema_values = self.ema(self.closes, p["trend_ema_period"])

        # --- V2: ATR + Percentile ---
        atr_period = p["vol_atr_period"]
        self.atr_values = self.atr(self.highs, self.lows, self.closes, atr_period)

        if p["use_vol_filter"]:
            vol_lookback = p["vol_lookback"]
            self.atr_percentile = np.full(n, np.nan)
            for i in range(vol_lookback, n):
                if np.isnan(self.atr_values[i]):
                    continue
                window = self.atr_values[i - vol_lookback:i + 1]
                valid = window[~np.isnan(window)]
                if len(valid) > 10:
                    self.atr_percentile[i] = np.sum(valid < self.atr_values[i]) / len(valid)

        # Reset state
        self.swing_hi_price = []
        self.swing_hi_cvd = []
        self.swing_hi_bar = []
        self.swing_lo_price = []
        self.swing_lo_cvd = []
        self.swing_lo_bar = []
        self.last_bull_bar = -9999
        self.last_bear_bar = -9999
        self._pivots_processed_to = 0

    def _update_swing_levels(self, bar_index: int):
        """Update swing level arrays with any new pivots up to bar_index."""
        p = self.params
        pivot_right = p["pivot_right"]

        for i in range(self._pivots_processed_to, bar_index + 1):
            if not np.isnan(self.pivot_highs[i]):
                pivot_bar = i - pivot_right
                if pivot_bar >= 0:
                    self.swing_hi_price.append(self.pivot_highs[i])
                    self.swing_hi_cvd.append(self.cvd[pivot_bar])
                    self.swing_hi_bar.append(pivot_bar)

            if not np.isnan(self.pivot_lows[i]):
                pivot_bar = i - pivot_right
                if pivot_bar >= 0:
                    self.swing_lo_price.append(self.pivot_lows[i])
                    self.swing_lo_cvd.append(self.cvd[pivot_bar])
                    self.swing_lo_bar.append(pivot_bar)

        self._pivots_processed_to = bar_index + 1

        max_age = p["max_level_age"]
        max_levels = p["max_levels"]

        i = len(self.swing_hi_price) - 1
        while i >= 0:
            if (bar_index - self.swing_hi_bar[i]) > max_age:
                self.swing_hi_price.pop(i)
                self.swing_hi_cvd.pop(i)
                self.swing_hi_bar.pop(i)
            i -= 1
        while len(self.swing_hi_price) > max_levels:
            self.swing_hi_price.pop(0)
            self.swing_hi_cvd.pop(0)
            self.swing_hi_bar.pop(0)

        i = len(self.swing_lo_price) - 1
        while i >= 0:
            if (bar_index - self.swing_lo_bar[i]) > max_age:
                self.swing_lo_price.pop(i)
                self.swing_lo_cvd.pop(i)
                self.swing_lo_bar.pop(i)
            i -= 1
        while len(self.swing_lo_price) > max_levels:
            self.swing_lo_price.pop(0)
            self.swing_lo_cvd.pop(0)
            self.swing_lo_bar.pop(0)

    def _check_trend_filter(self, bar_index: int) -> tuple[bool, bool]:
        """
        Check trend filter. Returns (allow_long, allow_short).

        Modes:
        - "with_trend": longs only above EMA, shorts only below
        - "ranging": only trade when price is near EMA (within N*ATR band)
        - "near_ema": combine both — trade with trend but only near EMA
        """
        p = self.params
        if not p["use_trend_filter"] or self.trend_ema_values is None:
            return True, True

        ema_val = self.trend_ema_values[bar_index]
        if np.isnan(ema_val):
            return True, True

        close = self.closes[bar_index]
        atr_val = self.atr_values[bar_index] if self.atr_values is not None else 10.0
        if np.isnan(atr_val):
            atr_val = 10.0

        mode = p["trend_mode"]
        band = p["trend_atr_band"] * atr_val

        if mode == "with_trend":
            allow_long = close > ema_val
            allow_short = close < ema_val
            return allow_long, allow_short

        elif mode == "ranging":
            in_range = abs(close - ema_val) < band
            return in_range, in_range

        elif mode == "near_ema":
            in_range = abs(close - ema_val) < band
            allow_long = close > ema_val and in_range
            allow_short = close < ema_val and in_range
            return allow_long, allow_short

        return True, True

    def _check_vol_filter(self, bar_index: int) -> bool:
        """Check volatility filter. Returns True if OK to trade."""
        p = self.params
        if not p["use_vol_filter"] or self.atr_percentile is None:
            return True

        pctile = self.atr_percentile[bar_index]
        if np.isnan(pctile):
            return True

        return p["vol_min_pctile"] <= pctile <= p["vol_max_pctile"]

    def generate_signal(self, bar_index: int) -> Optional[dict]:
        """Check for liquidity sweep + CVD divergence with V2 filters."""
        p = self.params

        if bar_index < max(p["pivot_left"] + p["pivot_right"] + 1, p.get("trend_ema_period", 200) + 1):
            return None

        # V2: Volatility filter
        if not self._check_vol_filter(bar_index):
            # Still update swing levels even when filtered
            self._update_swing_levels(bar_index)
            return None

        # V2: Trend filter
        trend_allow_long, trend_allow_short = self._check_trend_filter(bar_index)

        # Update swing levels
        self._update_swing_levels(bar_index)

        current_high = self.highs[bar_index]
        current_low = self.lows[bar_index]
        current_cvd = self.cvd[bar_index]

        # Sweep threshold (static or dynamic)
        if p["use_dynamic_sweep"] and self.atr_values is not None:
            atr_val = self.atr_values[bar_index]
            if not np.isnan(atr_val):
                sweep_threshold = atr_val * p["dynamic_sweep_atr_frac"]
            else:
                sweep_threshold = p["sweep_ticks"] * p["tick_size"]
        else:
            sweep_threshold = p["sweep_ticks"] * p["tick_size"]

        # CVD range for strength
        cvd_hi = self.cvd_highest[bar_index] if not np.isnan(self.cvd_highest[bar_index]) else current_cvd + 1
        cvd_lo = self.cvd_lowest[bar_index] if not np.isnan(self.cvd_lowest[bar_index]) else current_cvd - 1
        cvd_range = max(cvd_hi - cvd_lo, 1.0)

        # CVD momentum
        mom_bars = p["momentum_bars"]
        cvd_momentum = (current_cvd - self.cvd[bar_index - mom_bars]) if bar_index >= mom_bars else 0.0
        momentum_tol = cvd_range * p["momentum_tolerance"]

        # --- BEARISH ---
        bear_signal = None
        if p["allow_shorts"] and trend_allow_short and len(self.swing_hi_price) > 0:
            consumed = []
            best_strength = 0.0
            found = False
            f_swept_price = None

            momentum_ok = (cvd_momentum <= momentum_tol) if p["use_momentum_filter"] else True

            for i in range(len(self.swing_hi_price)):
                sh_price = self.swing_hi_price[i]
                sh_cvd = self.swing_hi_cvd[i]

                if current_high > sh_price + sweep_threshold:
                    consumed.append(i)

                    if current_cvd < sh_cvd and momentum_ok:
                        div_magnitude = abs(current_cvd - sh_cvd)
                        this_strength = min(div_magnitude / cvd_range, 1.0)

                        if this_strength > best_strength:
                            best_strength = this_strength
                            found = True
                            f_swept_price = sh_price

            for i in sorted(consumed, reverse=True):
                self.swing_hi_price.pop(i)
                self.swing_hi_cvd.pop(i)
                self.swing_hi_bar.pop(i)

            if found and best_strength >= p["min_strength"] and (bar_index - self.last_bear_bar) >= p["signal_cooldown"]:
                self.last_bear_bar = bar_index
                bear_signal = {
                    "direction": -1,
                    "strength": best_strength,
                    "swept_level": f_swept_price,
                }

        # --- BULLISH ---
        bull_signal = None
        if p["allow_longs"] and trend_allow_long and len(self.swing_lo_price) > 0:
            consumed = []
            best_strength = 0.0
            found = False
            f_swept_price = None

            momentum_ok = (cvd_momentum >= -momentum_tol) if p["use_momentum_filter"] else True

            for i in range(len(self.swing_lo_price)):
                sl_price = self.swing_lo_price[i]
                sl_cvd = self.swing_lo_cvd[i]

                if current_low < sl_price - sweep_threshold:
                    consumed.append(i)

                    if current_cvd > sl_cvd and momentum_ok:
                        div_magnitude = abs(current_cvd - sl_cvd)
                        this_strength = min(div_magnitude / cvd_range, 1.0)

                        if this_strength > best_strength:
                            best_strength = this_strength
                            found = True
                            f_swept_price = sl_price

            for i in sorted(consumed, reverse=True):
                self.swing_lo_price.pop(i)
                self.swing_lo_cvd.pop(i)
                self.swing_lo_bar.pop(i)

            if found and best_strength >= p["min_strength"] and (bar_index - self.last_bull_bar) >= p["signal_cooldown"]:
                self.last_bull_bar = bar_index
                bull_signal = {
                    "direction": 1,
                    "strength": best_strength,
                    "swept_level": f_swept_price,
                }

        if bear_signal and bull_signal:
            return bear_signal if bear_signal["strength"] >= bull_signal["strength"] else bull_signal

        return bear_signal or bull_signal
