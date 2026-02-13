"""
Risk management module for the backtesting engine.
Handles stop loss, take profit, trailing stops, break-even, position sizing,
session filters, drawdown limits, and time-based exits.
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class StopLossMode(Enum):
    FIXED_POINTS = "fixed_points"
    PERCENTAGE = "percentage"
    ATR = "atr"
    SWEPT_LEVEL = "swept_level"


class TakeProfitMode(Enum):
    FIXED_POINTS = "fixed_points"
    PERCENTAGE = "percentage"
    ATR = "atr"
    RISK_REWARD = "risk_reward"


class TrailingMode(Enum):
    FIXED_POINTS = "fixed_points"
    PERCENTAGE = "percentage"
    ATR = "atr"


class PositionSizeMode(Enum):
    PERCENT_EQUITY = "percent_equity"
    FIXED_CONTRACTS = "fixed_contracts"
    RISK_PER_TRADE = "risk_per_trade"


@dataclass
class RiskConfig:
    """Complete risk management configuration."""

    # Position Sizing
    position_mode: PositionSizeMode = PositionSizeMode.RISK_PER_TRADE
    equity_pct: float = 100.0
    fixed_contracts: int = 1
    risk_pct: float = 1.0

    # Stop Loss
    use_stop_loss: bool = True
    sl_mode: StopLossMode = StopLossMode.ATR
    sl_fixed_points: float = 20.0
    sl_percentage: float = 1.0
    sl_atr_mult: float = 2.0
    sl_atr_period: int = 14
    sl_swept_offset_ticks: float = 2.0

    # Take Profit
    use_take_profit: bool = True
    tp_mode: TakeProfitMode = TakeProfitMode.RISK_REWARD
    tp_fixed_points: float = 40.0
    tp_percentage: float = 2.0
    tp_atr_mult: float = 3.0
    tp_rr_ratio: float = 2.0

    # Trailing Stop
    use_trailing: bool = False
    trail_mode: TrailingMode = TrailingMode.ATR
    trail_fixed_points: float = 15.0
    trail_percentage: float = 1.0
    trail_atr_mult: float = 1.5

    # Break Even
    use_breakeven: bool = False
    be_trigger_points: float = 15.0
    be_offset: float = 1.0

    # Session Filter
    use_session_filter: bool = False
    session_start_hour: int = 9
    session_start_minute: int = 30
    session_end_hour: int = 16
    session_end_minute: int = 0
    close_at_session_end: bool = False

    # Risk Limits
    use_max_drawdown: bool = False
    max_drawdown_pct: float = 5.0
    use_daily_loss_limit: bool = False
    daily_loss_pct: float = 2.0

    # Time Exit
    use_max_hold: bool = False
    max_hold_bars: int = 50

    # Commission
    commission_per_contract: float = 1.24
    slippage_ticks: int = 1
    tick_size: float = 0.25
    point_value: float = 50.0  # ES futures: $50/point


@dataclass
class Position:
    """Tracks an open position."""

    direction: int  # 1 = long, -1 = short
    entry_price: float = 0.0
    quantity: int = 1
    entry_bar: int = 0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None
    trailing_high: Optional[float] = None  # Highest price since entry (long)
    trailing_low: Optional[float] = None  # Lowest price since entry (short)
    breakeven_triggered: bool = False
    swept_level: Optional[float] = None


@dataclass
class Trade:
    """Completed trade record."""

    direction: int
    entry_price: float
    exit_price: float
    quantity: int
    entry_bar: int
    exit_bar: int
    entry_time: str = ""
    exit_time: str = ""
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""
    commission: float = 0.0
    slippage: float = 0.0
    bars_held: int = 0


class RiskManager:
    """
    Manages all risk management for positions.
    Calculates stops, targets, position sizes, and enforces risk limits.
    """

    def __init__(self, config: RiskConfig):
        self.config = config
        self.position: Optional[Position] = None
        self.trades: list[Trade] = []
        self.peak_equity: float = 0.0
        self.day_start_equity: float = 0.0
        self.current_day: Optional[str] = None
        self.trading_halted: bool = False

    def reset(self, initial_capital: float):
        """Reset state for a new backtest run."""
        self.position = None
        self.trades = []
        self.peak_equity = initial_capital
        self.day_start_equity = initial_capital
        self.current_day = None
        self.trading_halted = False

    def calc_atr(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> float:
        """Calculate ATR from recent bars."""
        period = self.config.sl_atr_period
        if len(highs) < period + 1:
            return (highs[-1] - lows[-1])  # fallback to last bar range

        tr_values = np.zeros(period)
        for i in range(period):
            idx = -(period - i)
            h = highs[idx]
            l = lows[idx]
            prev_c = closes[idx - 1]
            tr_values[i] = max(h - l, abs(h - prev_c), abs(l - prev_c))

        return np.mean(tr_values)

    def calc_stop_loss(
        self,
        direction: int,
        entry_price: float,
        atr: float,
        swept_level: Optional[float] = None,
    ) -> Optional[float]:
        """Calculate stop loss price."""
        if not self.config.use_stop_loss:
            return None

        cfg = self.config
        tick = cfg.tick_size

        if cfg.sl_mode == StopLossMode.FIXED_POINTS:
            dist = cfg.sl_fixed_points
        elif cfg.sl_mode == StopLossMode.PERCENTAGE:
            dist = entry_price * cfg.sl_percentage / 100
        elif cfg.sl_mode == StopLossMode.ATR:
            dist = atr * cfg.sl_atr_mult
        elif cfg.sl_mode == StopLossMode.SWEPT_LEVEL:
            if swept_level is not None:
                offset = cfg.sl_swept_offset_ticks * tick
                if direction == 1:  # long
                    return swept_level - offset
                else:  # short
                    return swept_level + offset
            else:
                dist = atr * 2.0  # fallback
        else:
            return None

        if direction == 1:
            return entry_price - dist
        else:
            return entry_price + dist

    def calc_take_profit(
        self,
        direction: int,
        entry_price: float,
        sl_price: Optional[float],
        atr: float,
    ) -> Optional[float]:
        """Calculate take profit price."""
        if not self.config.use_take_profit:
            return None

        cfg = self.config

        if cfg.tp_mode == TakeProfitMode.FIXED_POINTS:
            dist = cfg.tp_fixed_points
        elif cfg.tp_mode == TakeProfitMode.PERCENTAGE:
            dist = entry_price * cfg.tp_percentage / 100
        elif cfg.tp_mode == TakeProfitMode.ATR:
            dist = atr * cfg.tp_atr_mult
        elif cfg.tp_mode == TakeProfitMode.RISK_REWARD:
            if sl_price is not None:
                sl_dist = abs(entry_price - sl_price)
                dist = sl_dist * cfg.tp_rr_ratio
            else:
                dist = atr * 2 * cfg.tp_rr_ratio  # fallback
        else:
            return None

        if direction == 1:
            return entry_price + dist
        else:
            return entry_price - dist

    def calc_trailing_distance(self, current_price: float, atr: float) -> float:
        """Calculate trailing stop distance."""
        cfg = self.config
        if cfg.trail_mode == TrailingMode.FIXED_POINTS:
            return cfg.trail_fixed_points
        elif cfg.trail_mode == TrailingMode.PERCENTAGE:
            return current_price * cfg.trail_percentage / 100
        elif cfg.trail_mode == TrailingMode.ATR:
            return atr * cfg.trail_atr_mult
        return atr * 1.5  # fallback

    def calc_position_size(
        self,
        equity: float,
        entry_price: float,
        sl_price: Optional[float],
    ) -> int:
        """Calculate position size based on mode."""
        cfg = self.config

        if cfg.position_mode == PositionSizeMode.FIXED_CONTRACTS:
            return cfg.fixed_contracts

        if cfg.position_mode == PositionSizeMode.PERCENT_EQUITY:
            value = equity * (cfg.equity_pct / 100)
            qty = int(value / (entry_price * cfg.point_value)) if entry_price > 0 else 1
            return max(qty, 1)

        if cfg.position_mode == PositionSizeMode.RISK_PER_TRADE:
            if sl_price is not None and sl_price != entry_price:
                risk_amount = equity * (cfg.risk_pct / 100)
                sl_distance = abs(entry_price - sl_price)
                risk_per_contract = sl_distance * cfg.point_value
                if risk_per_contract > 0:
                    qty = int(risk_amount / risk_per_contract)
                    return max(qty, 1)
            # Fallback: use 2% of equity
            risk_amount = equity * 0.02
            return max(int(risk_amount / (entry_price * cfg.point_value)), 1)

        return 1

    def check_session(self, timestamp) -> bool:
        """Check if current time is within trading session."""
        if not self.config.use_session_filter:
            return True

        hour = timestamp.hour
        minute = timestamp.minute
        current_minutes = hour * 60 + minute
        start_minutes = self.config.session_start_hour * 60 + self.config.session_start_minute
        end_minutes = self.config.session_end_hour * 60 + self.config.session_end_minute

        return start_minutes <= current_minutes < end_minutes

    def check_session_close(self, timestamp) -> bool:
        """Check if session just ended (should close positions)."""
        if not self.config.use_session_filter or not self.config.close_at_session_end:
            return False

        hour = timestamp.hour
        minute = timestamp.minute
        current_minutes = hour * 60 + minute
        end_minutes = self.config.session_end_hour * 60 + self.config.session_end_minute

        return current_minutes >= end_minutes

    def check_drawdown(self, equity: float) -> bool:
        """Check if max drawdown limit is breached. Returns True if OK to trade."""
        if not self.config.use_max_drawdown:
            return True

        self.peak_equity = max(self.peak_equity, equity)
        if self.peak_equity > 0:
            dd_pct = (self.peak_equity - equity) / self.peak_equity * 100
            if dd_pct >= self.config.max_drawdown_pct:
                self.trading_halted = True
                return False
        return True

    def check_daily_loss(self, equity: float, timestamp) -> bool:
        """Check if daily loss limit is breached. Returns True if OK to trade."""
        if not self.config.use_daily_loss_limit:
            return True

        current_day = str(timestamp.date()) if hasattr(timestamp, 'date') else str(timestamp)[:10]
        if self.current_day != current_day:
            self.current_day = current_day
            self.day_start_equity = equity

        if self.day_start_equity > 0:
            daily_loss = (self.day_start_equity - equity) / self.day_start_equity * 100
            if daily_loss >= self.config.daily_loss_pct:
                return False
        return True

    def can_trade(self, equity: float, timestamp) -> bool:
        """Check all risk limits to determine if we can enter a new trade."""
        if self.trading_halted:
            return False
        if not self.check_session(timestamp):
            return False
        if not self.check_drawdown(equity):
            return False
        if not self.check_daily_loss(equity, timestamp):
            return False
        return True

    def open_position(
        self,
        direction: int,
        entry_price: float,
        bar_index: int,
        equity: float,
        atr: float,
        swept_level: Optional[float] = None,
        timestamp: str = "",
    ) -> Position:
        """Open a new position with calculated risk levels."""
        cfg = self.config
        slippage = cfg.slippage_ticks * cfg.tick_size

        # Apply slippage to entry
        if direction == 1:
            actual_entry = entry_price + slippage
        else:
            actual_entry = entry_price - slippage

        sl_price = self.calc_stop_loss(direction, actual_entry, atr, swept_level)
        tp_price = self.calc_take_profit(direction, actual_entry, sl_price, atr)
        qty = self.calc_position_size(equity, actual_entry, sl_price)

        self.position = Position(
            direction=direction,
            entry_price=actual_entry,
            quantity=qty,
            entry_bar=bar_index,
            stop_loss=sl_price,
            take_profit=tp_price,
            trailing_high=actual_entry if direction == 1 else None,
            trailing_low=actual_entry if direction == -1 else None,
            swept_level=swept_level,
        )
        return self.position

    def update_position(
        self,
        high: float,
        low: float,
        close: float,
        bar_index: int,
        atr: float,
        timestamp=None,
    ) -> Optional[Trade]:
        """
        Update position state and check all exit conditions.
        Returns a Trade if the position was closed, None otherwise.

        Checks in order of priority:
        1. Stop loss hit
        2. Take profit hit
        3. Trailing stop hit
        4. Break-even trigger
        5. Max hold time
        6. Session close
        """
        if self.position is None:
            return None

        pos = self.position
        cfg = self.config
        exit_price = None
        exit_reason = ""

        # --- Check Stop Loss ---
        if pos.stop_loss is not None:
            if pos.direction == 1 and low <= pos.stop_loss:
                exit_price = pos.stop_loss
                exit_reason = "Stop Loss"
            elif pos.direction == -1 and high >= pos.stop_loss:
                exit_price = pos.stop_loss
                exit_reason = "Stop Loss"

        # --- Check Take Profit ---
        if exit_price is None and pos.take_profit is not None:
            if pos.direction == 1 and high >= pos.take_profit:
                exit_price = pos.take_profit
                exit_reason = "Take Profit"
            elif pos.direction == -1 and low <= pos.take_profit:
                exit_price = pos.take_profit
                exit_reason = "Take Profit"

        # --- Update and Check Trailing Stop ---
        if exit_price is None and cfg.use_trailing:
            trail_dist = self.calc_trailing_distance(close, atr)

            if pos.direction == 1:
                pos.trailing_high = max(pos.trailing_high or high, high)
                new_trail = pos.trailing_high - trail_dist
                if pos.trailing_stop is None or new_trail > pos.trailing_stop:
                    pos.trailing_stop = new_trail
                if low <= pos.trailing_stop:
                    exit_price = pos.trailing_stop
                    exit_reason = "Trailing Stop"
            else:
                pos.trailing_low = min(pos.trailing_low or low, low)
                new_trail = pos.trailing_low + trail_dist
                if pos.trailing_stop is None or new_trail < pos.trailing_stop:
                    pos.trailing_stop = new_trail
                if high >= pos.trailing_stop:
                    exit_price = pos.trailing_stop
                    exit_reason = "Trailing Stop"

        # --- Check Break-Even ---
        if exit_price is None and cfg.use_breakeven and not pos.breakeven_triggered:
            if pos.direction == 1 and high >= pos.entry_price + cfg.be_trigger_points:
                pos.stop_loss = pos.entry_price + cfg.be_offset
                pos.breakeven_triggered = True
            elif pos.direction == -1 and low <= pos.entry_price - cfg.be_trigger_points:
                pos.stop_loss = pos.entry_price - cfg.be_offset
                pos.breakeven_triggered = True

        # --- Check Max Hold Time ---
        if exit_price is None and cfg.use_max_hold:
            bars_held = bar_index - pos.entry_bar
            if bars_held >= cfg.max_hold_bars:
                exit_price = close
                exit_reason = "Max Hold Time"

        # --- Check Session Close ---
        if exit_price is None and timestamp is not None and self.check_session_close(timestamp):
            exit_price = close
            exit_reason = "Session Close"

        # --- Execute Exit ---
        if exit_price is not None:
            return self._close_position(exit_price, bar_index, exit_reason, str(timestamp) if timestamp else "")

        return None

    def force_close(self, close_price: float, bar_index: int, reason: str = "Signal", timestamp: str = "") -> Optional[Trade]:
        """Force close the current position."""
        if self.position is None:
            return None
        return self._close_position(close_price, bar_index, reason, timestamp)

    def _close_position(self, exit_price: float, bar_index: int, reason: str, timestamp: str = "") -> Trade:
        """Internal: close position and create trade record."""
        pos = self.position
        cfg = self.config
        slippage = cfg.slippage_ticks * cfg.tick_size

        # Apply slippage to exit
        if pos.direction == 1:
            actual_exit = exit_price - slippage
        else:
            actual_exit = exit_price + slippage

        # Calculate P&L
        price_diff = (actual_exit - pos.entry_price) * pos.direction
        gross_pnl = price_diff * pos.quantity * cfg.point_value
        commission = cfg.commission_per_contract * pos.quantity * 2  # round trip
        net_pnl = gross_pnl - commission

        pnl_pct = (price_diff / pos.entry_price) * 100 if pos.entry_price != 0 else 0

        trade = Trade(
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=actual_exit,
            quantity=pos.quantity,
            entry_bar=pos.entry_bar,
            exit_bar=bar_index,
            exit_time=timestamp,
            pnl=net_pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason,
            commission=commission,
            slippage=slippage * 2 * pos.quantity * cfg.point_value,
            bars_held=bar_index - pos.entry_bar,
        )

        self.trades.append(trade)
        self.position = None
        return trade
