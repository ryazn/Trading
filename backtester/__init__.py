"""
Trading Backtester - A TradingView-inspired backtesting engine.

Supports 1-min bars, custom strategies, and comprehensive risk management.
"""

from backtester.engine import BacktestEngine
from backtester.data_loader import DataLoader
from backtester.risk_manager import RiskManager
from backtester.performance import PerformanceReport

__version__ = "1.0.0"
__all__ = ["BacktestEngine", "DataLoader", "RiskManager", "PerformanceReport"]
