"""
Data loader for backtesting engine.
Supports CSV files, Databento DBN files, and yfinance downloads for 1-min bar data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional


class DataLoader:
    """Load and prepare OHLCV bar data for backtesting."""

    REQUIRED_COLUMNS = ["open", "high", "low", "close"]

    def __init__(self):
        self._data: Optional[pd.DataFrame] = None

    @property
    def data(self) -> pd.DataFrame:
        if self._data is None:
            raise ValueError("No data loaded. Call load_csv() or download() first.")
        return self._data

    def load_csv(
        self,
        filepath: str,
        datetime_col: str = "datetime",
        datetime_format: Optional[str] = None,
        separator: str = ",",
    ) -> pd.DataFrame:
        """
        Load OHLCV data from a CSV file.

        Supports:
        - Standard format: datetime, open, high, low, close, volume
        - TradingView export: unix timestamp 'time', open, high, low, close, + extras (CVD, signals)

        Columns are case-insensitive (will be lowercased).
        Extra columns (cvd, bullish entry, bearish entry) are preserved.
        """
        df = pd.read_csv(filepath, sep=separator)
        df.columns = [c.strip().lower() for c in df.columns]

        # Find datetime column
        dt_col = None
        for candidate in [datetime_col.lower(), "datetime", "date", "time", "timestamp"]:
            if candidate in df.columns:
                dt_col = candidate
                break

        if dt_col is None:
            raise ValueError(
                f"No datetime column found. Available: {list(df.columns)}"
            )

        # Detect unix timestamps (large integers) vs datetime strings
        sample_val = df[dt_col].iloc[0]
        if isinstance(sample_val, (int, float, np.integer, np.floating)) and sample_val > 1e9:
            df[dt_col] = pd.to_datetime(df[dt_col], unit="s", utc=True)
            # Convert to US/Eastern for ES futures
            df[dt_col] = df[dt_col].dt.tz_convert("US/Eastern").dt.tz_localize(None)
        elif datetime_format:
            df[dt_col] = pd.to_datetime(df[dt_col], format=datetime_format)
        else:
            df[dt_col] = pd.to_datetime(df[dt_col])

        df = df.set_index(dt_col)
        df.index.name = "datetime"

        # Validate required columns (volume is optional for TV exports with CVD)
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # If no volume column, create a synthetic one (needed by engine)
        if "volume" not in df.columns:
            df["volume"] = 1000  # placeholder

        # Ensure numeric types for core columns
        for col in self.REQUIRED_COLUMNS + ["volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Preserve extra columns (cvd, signals) as numeric
        for col in df.columns:
            if col not in self.REQUIRED_COLUMNS + ["volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=self.REQUIRED_COLUMNS)
        df = df.sort_index()

        self._data = df
        return df

    def load_directory(
        self,
        dirpath: str,
        pattern: str = "*.csv",
        min_price: float = 0.0,
    ) -> pd.DataFrame:
        """
        Load and concatenate multiple CSV files from a directory.

        Useful for yearly Databento CSV splits (ES_2010.csv, ES_2011.csv, ...).
        Deduplicates by keeping the highest-volume row per timestamp
        (front-month contract for futures with overlapping contract months).

        Args:
            min_price: Filter out bars with close below this price.
                       Use to remove spread/micro contract data.
                       For ES futures, 1000.0 is a safe floor.
        """
        import glob as _glob

        files = sorted(_glob.glob(str(Path(dirpath) / pattern)))
        if not files:
            raise ValueError(f"No files matching '{pattern}' in {dirpath}")

        dfs = []
        for f in files:
            loader = DataLoader()
            dfs.append(loader.load_csv(f))

        df = pd.concat(dfs)
        df = df.sort_index()

        # Filter out non-front-month data (spreads, micros) by price floor
        if min_price > 0:
            before = len(df)
            df = df[df["close"] >= min_price]
            removed = before - len(df)
            if removed > 0:
                print(f"  Price filter (>= {min_price}): removed {removed:,} bars")

        # Deduplicate: keep highest-volume row per timestamp
        # (front-month futures contract has more volume than back-month)
        if df.index.duplicated().any():
            before = len(df)
            df = df.sort_values("volume", ascending=False).groupby(level=0).first()
            df = df.sort_index()
            after = len(df)
            print(f"  Deduplicated: {before:,} → {after:,} bars "
                  f"(removed {before - after:,} duplicate timestamps)")

        self._data = df
        return df

    def load_databento(self, filepath: str) -> pd.DataFrame:
        """
        Load OHLCV data from a Databento file (.csv.zst or .dbn.zst).

        Supports:
        - CSV compressed with zstandard (.csv.zst) — uses pandas directly
        - DBN compressed with zstandard (.dbn.zst) — uses databento library

        Databento CSV columns: ts_event, open, high, low, close, volume, ...
        Timestamps are converted to US/Eastern for ES futures.
        """
        filepath_lower = filepath.lower()

        if filepath_lower.endswith(".csv.zst") or filepath_lower.endswith(".csv"):
            df = self._load_databento_csv(filepath)
        elif filepath_lower.endswith(".dbn.zst") or filepath_lower.endswith(".dbn"):
            df = self._load_databento_dbn(filepath)
        else:
            # Try CSV first, fall back to DBN
            try:
                df = self._load_databento_csv(filepath)
            except Exception:
                df = self._load_databento_dbn(filepath)

        self._data = df
        return df

    def _load_databento_csv(self, filepath: str) -> pd.DataFrame:
        """Load Databento CSV (.csv.zst) using pandas."""
        compression = "zstd" if filepath.lower().endswith(".zst") else "infer"
        df = pd.read_csv(filepath, compression=compression)
        df.columns = [c.strip().lower() for c in df.columns]

        # Find timestamp column
        ts_col = None
        for candidate in ["ts_event", "ts_recv", "timestamp", "datetime", "time"]:
            if candidate in df.columns:
                ts_col = candidate
                break

        if ts_col is None:
            raise ValueError(f"No timestamp column found. Available: {list(df.columns)}")

        # Databento ts_event is nanosecond unix timestamp
        sample_val = df[ts_col].iloc[0]
        if isinstance(sample_val, (int, float, np.integer, np.floating)) and sample_val > 1e15:
            # Nanosecond timestamps
            df[ts_col] = pd.to_datetime(df[ts_col], unit="ns", utc=True)
            df[ts_col] = df[ts_col].dt.tz_convert("US/Eastern").dt.tz_localize(None)
        elif isinstance(sample_val, (int, float, np.integer, np.floating)) and sample_val > 1e9:
            # Second timestamps
            df[ts_col] = pd.to_datetime(df[ts_col], unit="s", utc=True)
            df[ts_col] = df[ts_col].dt.tz_convert("US/Eastern").dt.tz_localize(None)
        else:
            df[ts_col] = pd.to_datetime(df[ts_col])

        df = df.set_index(ts_col)
        df.index.name = "datetime"

        # Drop Databento metadata columns
        drop_cols = [
            "rtype", "publisher_id", "instrument_id", "ts_recv",
            "flags", "sequence", "symbol", "ts_event",
        ]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

        # Validate OHLCV columns exist
        price_cols = ["open", "high", "low", "close"]
        missing = [c for c in price_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Ensure numeric types
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        if "volume" not in df.columns:
            df["volume"] = 1000

        df = df.dropna(subset=price_cols)
        df = df.sort_index()
        return df

    def _load_databento_dbn(self, filepath: str) -> pd.DataFrame:
        """Load Databento DBN (.dbn.zst) using databento library."""
        try:
            import databento as db
        except ImportError:
            raise ImportError("databento is required: pip install databento")

        store = db.DBNStore.from_file(filepath)
        df = store.to_df()

        # Convert fixed-point integer prices to float if needed
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            if col in df.columns and df[col].dtype in [np.int64, np.int32]:
                df[col] = df[col] / 1_000_000_000

        # Handle datetime index
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = df.index.tz_convert("US/Eastern").tz_localize(None)
        elif df.index.dtype == np.int64 or df.index.dtype == np.uint64:
            df.index = pd.to_datetime(df.index, unit="ns", utc=True)
            df.index = df.index.tz_convert("US/Eastern").tz_localize(None)

        df.index.name = "datetime"

        drop_cols = [
            "rtype", "publisher_id", "instrument_id", "ts_recv",
            "flags", "sequence", "symbol",
        ]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        if "volume" not in df.columns:
            df["volume"] = 1000

        df = df.dropna(subset=price_cols)
        df = df.sort_index()
        return df

    def download(
        self,
        symbol: str,
        period: str = "5d",
        interval: str = "1m",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Download data via yfinance.

        For 1-min data, yfinance limits to ~7 days of history.
        For longer history, use 2m, 5m, 15m, etc.
        """
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance is required: pip install yfinance")

        ticker = yf.Ticker(symbol)

        if start and end:
            df = ticker.history(start=start, end=end, interval=interval)
        else:
            df = ticker.history(period=period, interval=interval)

        if df.empty:
            raise ValueError(f"No data returned for {symbol}")

        df.columns = [c.strip().lower() for c in df.columns]

        # yfinance may return extra columns (dividends, stock splits)
        keep_cols = [c for c in self.REQUIRED_COLUMNS if c in df.columns]
        df = df[keep_cols]
        df.index.name = "datetime"

        self._data = df
        return df

    def resample(self, timeframe: str) -> pd.DataFrame:
        """
        Resample loaded data to a different timeframe.

        Args:
            timeframe: pandas offset alias, e.g. '5min', '15min', '1h', '1D'
        """
        df = self.data
        resampled = df.resample(timeframe).agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        )
        resampled = resampled.dropna()
        return resampled

    def generate_sample_data(
        self,
        bars: int = 5000,
        start_price: float = 5000.0,
        volatility: float = 0.001,
        start_date: str = "2024-01-02 09:30:00",
        freq: str = "1min",
    ) -> pd.DataFrame:
        """
        Generate synthetic OHLCV data for testing.
        Creates realistic-looking price action with trends, mean reversion, and volume patterns.
        """
        np.random.seed(42)
        dates = pd.date_range(start=start_date, periods=bars, freq=freq)

        prices = np.zeros(bars)
        prices[0] = start_price

        # Generate price with trending + mean-reverting components
        trend = np.cumsum(np.random.randn(bars) * volatility * start_price * 0.1)
        noise = np.random.randn(bars) * volatility * start_price

        for i in range(1, bars):
            prices[i] = prices[i - 1] + trend[i] * 0.01 + noise[i]
            # Mean revert slightly
            prices[i] += (start_price - prices[i]) * 0.0001

        opens = prices.copy()
        closes = prices + np.random.randn(bars) * volatility * start_price * 0.5

        highs = np.maximum(opens, closes) + np.abs(
            np.random.randn(bars) * volatility * start_price * 0.3
        )
        lows = np.minimum(opens, closes) - np.abs(
            np.random.randn(bars) * volatility * start_price * 0.3
        )

        # Volume with intraday pattern (higher at open/close)
        base_volume = 1000
        volume = np.random.exponential(base_volume, bars)
        # Add intraday U-shape if using minute data
        if "min" in freq.lower() or "t" in freq.lower():
            bar_in_session = np.arange(bars) % 390  # ~6.5 hours of 1-min bars
            session_factor = 1 + 0.5 * np.exp(-((bar_in_session - 0) ** 2) / 2000) + \
                            0.3 * np.exp(-((bar_in_session - 389) ** 2) / 2000)
            volume *= session_factor

        df = pd.DataFrame(
            {
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": volume.astype(int),
            },
            index=dates,
        )
        df.index.name = "datetime"

        self._data = df
        return df

    def get_stats(self) -> dict:
        """Return summary statistics for loaded data."""
        df = self.data
        return {
            "symbol": "N/A",
            "bars": len(df),
            "start": str(df.index[0]),
            "end": str(df.index[-1]),
            "timeframe": pd.infer_freq(df.index) or "irregular",
            "price_range": f"{df['low'].min():.2f} - {df['high'].max():.2f}",
            "avg_volume": f"{df['volume'].mean():.0f}",
            "total_volume": f"{df['volume'].sum():.0f}",
        }
