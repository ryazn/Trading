"""
Base strategy class for the backtesting engine.
All strategies should inherit from this.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional


class BaseStrategy(ABC):
    """
    Abstract base strategy.

    Subclasses must implement:
    - initialize(data): Pre-compute indicators on the full dataset
    - generate_signal(bar_index): Return signal dict or None for each bar
    """

    def __init__(self, params: Optional[dict] = None):
        self.params = params or {}
        self.data: Optional[pd.DataFrame] = None
        self.opens: Optional[np.ndarray] = None
        self.highs: Optional[np.ndarray] = None
        self.lows: Optional[np.ndarray] = None
        self.closes: Optional[np.ndarray] = None
        self.volumes: Optional[np.ndarray] = None

    def initialize(self, data: pd.DataFrame):
        """
        Called once before the backtest loop.
        Store references and pre-compute all indicators.
        """
        self.data = data
        self.opens = data["open"].values
        self.highs = data["high"].values
        self.lows = data["low"].values
        self.closes = data["close"].values
        self.volumes = data["volume"].values
        self._compute_indicators()

    @abstractmethod
    def _compute_indicators(self):
        """Pre-compute all strategy indicators on the full dataset."""
        pass

    @abstractmethod
    def generate_signal(self, bar_index: int) -> Optional[dict]:
        """
        Generate a trading signal for the given bar.

        Returns:
            None for no signal, or a dict with:
            {
                "direction": 1 (long) or -1 (short),
                "strength": float 0-1 (signal strength),
                "swept_level": float or None (price level that was swept),
            }
        """
        pass

    # --- Utility methods for subclasses ---

    @staticmethod
    def ema(values: np.ndarray, period: int) -> np.ndarray:
        """Exponential moving average."""
        result = np.full_like(values, np.nan, dtype=float)
        if len(values) < period:
            return result

        multiplier = 2.0 / (period + 1)
        result[period - 1] = np.mean(values[:period])

        for i in range(period, len(values)):
            result[i] = (values[i] - result[i - 1]) * multiplier + result[i - 1]

        return result

    @staticmethod
    def sma(values: np.ndarray, period: int) -> np.ndarray:
        """Simple moving average."""
        result = np.full_like(values, np.nan, dtype=float)
        for i in range(period - 1, len(values)):
            result[i] = np.mean(values[i - period + 1:i + 1])
        return result

    @staticmethod
    def highest(values: np.ndarray, period: int) -> np.ndarray:
        """Rolling highest value."""
        result = np.full_like(values, np.nan, dtype=float)
        for i in range(period - 1, len(values)):
            result[i] = np.max(values[i - period + 1:i + 1])
        return result

    @staticmethod
    def lowest(values: np.ndarray, period: int) -> np.ndarray:
        """Rolling lowest value."""
        result = np.full_like(values, np.nan, dtype=float)
        for i in range(period - 1, len(values)):
            result[i] = np.min(values[i - period + 1:i + 1])
        return result

    @staticmethod
    def atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> np.ndarray:
        """Average True Range."""
        n = len(highs)
        tr = np.zeros(n)
        tr[0] = highs[0] - lows[0]
        for i in range(1, n):
            tr[i] = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
        result = np.full(n, np.nan)
        for i in range(period - 1, n):
            result[i] = np.mean(tr[i - period + 1:i + 1])
        return result

    @staticmethod
    def pivot_high(highs: np.ndarray, left: int, right: int) -> np.ndarray:
        """
        Detect pivot highs.
        Returns array where non-NaN values are pivot high prices.
        Pivots are placed at the pivot bar (offset by -right from detection).
        """
        n = len(highs)
        result = np.full(n, np.nan)

        for i in range(left + right, n):
            pivot_bar = i - right
            pivot_val = highs[pivot_bar]

            is_pivot = True
            for j in range(pivot_bar - left, pivot_bar):
                if highs[j] >= pivot_val:
                    is_pivot = False
                    break
            if is_pivot:
                for j in range(pivot_bar + 1, pivot_bar + right + 1):
                    if j < n and highs[j] >= pivot_val:
                        is_pivot = False
                        break

            if is_pivot:
                result[i] = pivot_val

        return result

    @staticmethod
    def pivot_low(lows: np.ndarray, left: int, right: int) -> np.ndarray:
        """
        Detect pivot lows.
        Returns array where non-NaN values are pivot low prices.
        """
        n = len(lows)
        result = np.full(n, np.nan)

        for i in range(left + right, n):
            pivot_bar = i - right
            pivot_val = lows[pivot_bar]

            is_pivot = True
            for j in range(pivot_bar - left, pivot_bar):
                if lows[j] <= pivot_val:
                    is_pivot = False
                    break
            if is_pivot:
                for j in range(pivot_bar + 1, pivot_bar + right + 1):
                    if j < n and lows[j] <= pivot_val:
                        is_pivot = False
                        break

            if is_pivot:
                result[i] = pivot_val

        return result
