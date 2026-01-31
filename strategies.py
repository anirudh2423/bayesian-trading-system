"""
Trading Strategies for Multi-Armed Bandit System
Each strategy generates signals from price data and returns rewards
"""

import numpy as np
import pandas as pd
from typing import Tuple
from abc import ABC, abstractmethod


class TradingStrategy(ABC):
    """Base class for trading strategies"""
    
    def __init__(self, name: str):
        self.name = name
        self.position = 0  # -1 short, 0 neutral, 1 long
        
    @abstractmethod
    def generate_signal(self, data: pd.DataFrame, idx: int) -> int:
        """Generate trading signal: -1, 0, or 1"""
        pass
    
    def calculate_reward(self, data: pd.DataFrame, idx: int) -> float:
        """
        Calculate reward based on signal and next period's return.
        Returns reward in [0, 1] range.
        """
        if idx >= len(data) - 1:
            return 0.5  # Neutral if no next period
        
        signal = self.generate_signal(data, idx)
        next_return = data['returns'].iloc[idx + 1]
        
        # Reward = alignment of signal with actual return
        # Scale to [0, 1] range
        raw_reward = signal * next_return
        
        # Normalize: typical daily returns are -3% to +3%
        # Map to [0, 1] with 0.5 being neutral
        reward = 0.5 + (raw_reward / 0.05)  # 5% = max expected daily move
        reward = np.clip(reward, 0, 1)
        
        self.position = signal
        return reward


class MomentumStrategy(TradingStrategy):
    """
    Momentum Strategy: Buy winners, sell losers.
    Uses rolling returns to identify momentum.
    Best in: Trending markets (Bull/Bear)
    """
    
    def __init__(self, lookback: int = 20):
        super().__init__("Momentum")
        self.lookback = lookback
        
    def generate_signal(self, data: pd.DataFrame, idx: int) -> int:
        if idx < self.lookback:
            return 0
        
        # Calculate momentum as cumulative return over lookback period
        momentum = data['Close'].iloc[idx] / data['Close'].iloc[idx - self.lookback] - 1
        
        if momentum > 0.02:  # 2% threshold
            return 1  # Long
        elif momentum < -0.02:
            return -1  # Short
        return 0


class MeanReversionStrategy(TradingStrategy):
    """
    Mean Reversion Strategy: Buy oversold, sell overbought.
    Uses Bollinger Bands to identify mean reversion opportunities.
    Best in: Ranging/sideways markets
    """
    
    def __init__(self, lookback: int = 20, num_std: float = 2.0):
        super().__init__("Mean Reversion")
        self.lookback = lookback
        self.num_std = num_std
        
    def generate_signal(self, data: pd.DataFrame, idx: int) -> int:
        if idx < self.lookback:
            return 0
        
        # Calculate Bollinger Bands
        window = data['Close'].iloc[idx - self.lookback:idx + 1]
        sma = window.mean()
        std = window.std()
        
        upper_band = sma + self.num_std * std
        lower_band = sma - self.num_std * std
        current_price = data['Close'].iloc[idx]
        
        if current_price < lower_band:
            return 1  # Long - oversold
        elif current_price > upper_band:
            return -1  # Short - overbought
        return 0


class TrendFollowingStrategy(TradingStrategy):
    """
    Trend Following Strategy: Follow the trend using moving average crossovers.
    Uses SMA crossover to identify trend direction.
    Best in: Strong trending markets
    """
    
    def __init__(self, fast_period: int = 10, slow_period: int = 30):
        super().__init__("Trend Following")
        self.fast_period = fast_period
        self.slow_period = slow_period
        
    def generate_signal(self, data: pd.DataFrame, idx: int) -> int:
        if idx < self.slow_period:
            return 0
        
        # Calculate fast and slow SMAs
        fast_sma = data['Close'].iloc[idx - self.fast_period:idx + 1].mean()
        slow_sma = data['Close'].iloc[idx - self.slow_period:idx + 1].mean()
        
        # Previous SMAs for crossover detection
        prev_fast_sma = data['Close'].iloc[idx - self.fast_period - 1:idx].mean()
        prev_slow_sma = data['Close'].iloc[idx - self.slow_period - 1:idx].mean()
        
        # Trend direction
        if fast_sma > slow_sma:
            return 1  # Uptrend
        elif fast_sma < slow_sma:
            return -1  # Downtrend
        return 0


def create_strategies() -> list:
    """Factory function to create all strategies"""
    return [
        MomentumStrategy(lookback=20),
        MeanReversionStrategy(lookback=20, num_std=2.0),
        TrendFollowingStrategy(fast_period=10, slow_period=30)
    ]


class Oracle:
    """
    Oracle: Perfect hindsight baseline.
    Always picks the strategy that would have performed best.
    Used for regret calculation.
    """
    
    def __init__(self, strategies: list):
        self.strategies = strategies
        
    def get_best_reward(self, data: pd.DataFrame, idx: int) -> Tuple[float, str]:
        """
        Get the best possible reward and which strategy achieves it.
        """
        best_reward = 0
        best_strategy = None
        
        for strategy in self.strategies:
            reward = strategy.calculate_reward(data, idx)
            if reward > best_reward:
                best_reward = reward
                best_strategy = strategy.name
        
        # Reset positions
        for strategy in self.strategies:
            strategy.position = 0
            
        return best_reward, best_strategy


if __name__ == "__main__":
    # Quick test with fake data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    prices = 100 * np.cumprod(1 + np.random.randn(100) * 0.02)
    
    data = pd.DataFrame({
        'Close': prices,
        'returns': np.concatenate([[0], np.diff(prices) / prices[:-1]])
    }, index=dates)
    
    strategies = create_strategies()
    
    print("Testing strategies on synthetic data:")
    for strategy in strategies:
        rewards = [strategy.calculate_reward(data, i) for i in range(50, 60)]
        print(f"  {strategy.name}: avg reward = {np.mean(rewards):.3f}")
