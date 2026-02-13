"""
PHANTOM CVD Divergence Strategy - Python implementation.

Port of the PineScript PHANTOM CHART indicator logic:
1. Track swing highs/lows
2. Detect liquidity sweeps past those levels
3. Check for CVD divergence at the sweep
4. Generate entry signals on confirmed divergences
"""

import numpy as np
from typing import Optional
from strategies.base_strategy import BaseStrategy


class PhantomCVDStrategy(BaseStrategy):
    """
    PHANTOM CVD Divergence Strategy.

    Detects liquidity sweeps at prior swing levels combined with
    CVD (Cumulative Volume Delta) divergence for mean-reversion entries.

    Signal Logic:
    - LONG: Price sweeps below a prior swing low + CVD is higher than at the swing
    - SHORT: Price sweeps above a prior swing high + CVD is lower than at the swing
    """

    DEFAULT_PARAMS = {
        # CVD
        "cvd_smooth": 1,
        "cvd_anchor_bars": 390,  # 1 day of 1-min bars
        # Swing Detection
        "pivot_left": 10,
        "pivot_right": 1,
        "max_level_age": 200,
        "max_levels": 30,
        # Sweep Detection
        "sweep_ticks": 1,
        "tick_size": 0.25,
        # Strength Filter
        "min_strength": 0.5,
        "strength_period": 50,
        # CVD Momentum
        "use_momentum_filter": False,
        "momentum_bars": 5,
        "momentum_tolerance": 0.02,
        # Signal Control
        "signal_cooldown": 5,
        # Direction
        "allow_longs": True,
        "allow_shorts": True,
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

        # Runtime state for swing tracking
        self.swing_hi_price: list[float] = []
        self.swing_hi_cvd: list[float] = []
        self.swing_hi_bar: list[int] = []
        self.swing_lo_price: list[float] = []
        self.swing_lo_cvd: list[float] = []
        self.swing_lo_bar: list[int] = []

        # Signal cooldown
        self.last_bull_bar: int = -9999
        self.last_bear_bar: int = -9999

        # Track which bars we've already processed for pivots
        self._pivots_processed_to: int = 0

    def _compute_indicators(self):
        """Pre-compute CVD and pivot points on full dataset."""
        p = self.params
        n = len(self.closes)

        # --- Compute CVD ---
        # Use real CVD from TradingView data if available
        if self.data is not None and "cvd" in self.data.columns:
            cvd_raw = self.data["cvd"].values.astype(float)
            # Fill any NaN at the end
            for i in range(len(cvd_raw)):
                if np.isnan(cvd_raw[i]):
                    cvd_raw[i] = cvd_raw[i - 1] if i > 0 else 0.0
            self._using_real_cvd = True
        else:
            # Estimate CVD from bar direction (close vs open)
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

            anchor = p["cvd_anchor_bars"]
            cvd_raw = np.zeros(n)
            cumulative = 0.0
            for i in range(n):
                if anchor > 0 and i % anchor == 0:
                    cumulative = 0.0
                cumulative += delta[i]
                cvd_raw[i] = cumulative
            self._using_real_cvd = False

        # Smooth with EMA
        if p["cvd_smooth"] > 1:
            self.cvd = self.ema(cvd_raw, p["cvd_smooth"])
            for i in range(len(self.cvd)):
                if np.isnan(self.cvd[i]):
                    self.cvd[i] = cvd_raw[i]
                else:
                    break
        else:
            self.cvd = cvd_raw

        # CVD range for strength normalization
        self.cvd_highest = self.highest(self.cvd, p["strength_period"])
        self.cvd_lowest = self.lowest(self.cvd, p["strength_period"])

        # --- Pre-compute pivot points ---
        self.pivot_highs = self.pivot_high(self.highs, p["pivot_left"], p["pivot_right"])
        self.pivot_lows = self.pivot_low(self.lows, p["pivot_left"], p["pivot_right"])

        # Reset swing tracking state
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

        # Process any new pivot detections
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

        # Prune old levels
        max_age = p["max_level_age"]
        max_levels = p["max_levels"]

        # Remove old swing highs
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

        # Remove old swing lows
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

    def generate_signal(self, bar_index: int) -> Optional[dict]:
        """
        Check for liquidity sweep + CVD divergence at the current bar.
        """
        p = self.params

        if bar_index < p["pivot_left"] + p["pivot_right"] + 1:
            return None

        # Update swing levels
        self._update_swing_levels(bar_index)

        current_high = self.highs[bar_index]
        current_low = self.lows[bar_index]
        current_cvd = self.cvd[bar_index]

        sweep_threshold = p["sweep_ticks"] * p["tick_size"]

        # CVD range for strength
        cvd_hi = self.cvd_highest[bar_index] if not np.isnan(self.cvd_highest[bar_index]) else current_cvd + 1
        cvd_lo = self.cvd_lowest[bar_index] if not np.isnan(self.cvd_lowest[bar_index]) else current_cvd - 1
        cvd_range = max(cvd_hi - cvd_lo, 1.0)

        # CVD momentum
        mom_bars = p["momentum_bars"]
        cvd_momentum = (current_cvd - self.cvd[bar_index - mom_bars]) if bar_index >= mom_bars else 0.0
        momentum_tol = cvd_range * p["momentum_tolerance"]

        # --- Check BEARISH: sweep above swing high + CVD divergence ---
        bear_signal = None
        if p["allow_shorts"] and len(self.swing_hi_price) > 0:
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

            # Remove consumed levels (reverse order)
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

        # --- Check BULLISH: sweep below swing low + CVD divergence ---
        bull_signal = None
        if p["allow_longs"] and len(self.swing_lo_price) > 0:
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

            # Remove consumed levels (reverse order)
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

        # If both signals fire on same bar, pick the stronger one
        if bear_signal and bull_signal:
            if bear_signal["strength"] >= bull_signal["strength"]:
                return bear_signal
            else:
                return bull_signal

        return bear_signal or bull_signal
