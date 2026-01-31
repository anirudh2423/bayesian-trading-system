"""
Market Data Fetcher and Regime Detector
Downloads SPY data and classifies market regimes
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Tuple, List
from enum import Enum


class MarketRegime(Enum):
    """Market regime classifications"""
    BULL = "BULL"
    BEAR = "BEAR"
    RANGING = "RANGING"
    VOLATILE = "VOLATILE"


class MarketDataFetcher:
    """Fetches and processes market data from Yahoo Finance"""
    
    def __init__(self, ticker: str = "SPY"):
        self.ticker = ticker
        self.data: pd.DataFrame = None
        
    def fetch(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch OHLCV data from Yahoo Finance.
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with OHLCV data and calculated features
        """
        print(f"Fetching {self.ticker} data from {start_date} to {end_date}...")
        
        self.data = yf.download(self.ticker, start=start_date, end=end_date, progress=False)
        
        if self.data.empty:
            raise ValueError(f"No data returned for {self.ticker}")
        
        # Flatten column names if MultiIndex (happens with yfinance)
        if isinstance(self.data.columns, pd.MultiIndex):
            self.data.columns = self.data.columns.get_level_values(0)
        
        # Calculate returns
        self.data['returns'] = self.data['Close'].pct_change()
        
        # Calculate additional features for regime detection
        self.data['sma_20'] = self.data['Close'].rolling(20).mean()
        self.data['sma_50'] = self.data['Close'].rolling(50).mean()
        self.data['volatility_20'] = self.data['returns'].rolling(20).std() * np.sqrt(252)
        self.data['momentum_20'] = self.data['Close'] / self.data['Close'].shift(20) - 1
        
        # Drop NaN rows
        self.data = self.data.dropna()
        
        print(f"Loaded {len(self.data)} days of data")
        return self.data


class RegimeDetector:
    """Detects market regimes from price data"""
    
    def __init__(self, 
                 trend_threshold: float = 0.05,
                 volatility_threshold: float = 0.25):
        self.trend_threshold = trend_threshold  # 5% for trend detection
        self.volatility_threshold = volatility_threshold  # 25% annualized vol threshold
        
    def detect_regime(self, data: pd.DataFrame, idx: int) -> MarketRegime:
        """
        Detect the current market regime at index idx.
        
        Classification logic:
        - VOLATILE: Annualized volatility > threshold
        - BULL: Positive momentum AND price > SMAs
        - BEAR: Negative momentum AND price < SMAs
        - RANGING: Everything else
        """
        if idx < 50:  # Need enough history
            return MarketRegime.RANGING
            
        row = data.iloc[idx]
        
        # Check volatility first
        volatility = row['volatility_20']
        if volatility > self.volatility_threshold:
            return MarketRegime.VOLATILE
        
        # Check trend
        momentum = row['momentum_20']
        price = row['Close']
        sma_20 = row['sma_20']
        sma_50 = row['sma_50']
        
        if momentum > self.trend_threshold and price > sma_20 and sma_20 > sma_50:
            return MarketRegime.BULL
        elif momentum < -self.trend_threshold and price < sma_20 and sma_20 < sma_50:
            return MarketRegime.BEAR
        else:
            return MarketRegime.RANGING
    
    def detect_all_regimes(self, data: pd.DataFrame) -> List[MarketRegime]:
        """Detect regime for entire dataset"""
        regimes = []
        for idx in range(len(data)):
            regime = self.detect_regime(data, idx)
            regimes.append(regime)
        return regimes
    
    def get_regime_stats(self, regimes: List[MarketRegime]) -> dict:
        """Get statistics about regime distribution"""
        total = len(regimes)
        counts = {}
        for regime in MarketRegime:
            count = sum(1 for r in regimes if r == regime)
            counts[regime.value] = {
                'count': count,
                'percentage': count / total * 100 if total > 0 else 0
            }
        return counts


def load_spy_data(years: int = 5) -> Tuple[pd.DataFrame, List[MarketRegime]]:
    """
    Convenience function to load SPY data and detect regimes.
    
    Args:
        years: Number of years of historical data
        
    Returns:
        (data DataFrame, list of regimes)
    """
    from datetime import datetime, timedelta
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=years * 365)).strftime('%Y-%m-%d')
    
    fetcher = MarketDataFetcher("SPY")
    data = fetcher.fetch(start_date, end_date)
    
    detector = RegimeDetector()
    regimes = detector.detect_all_regimes(data)
    
    # Add regime to dataframe
    data['regime'] = [r.value for r in regimes]
    
    return data, regimes


if __name__ == "__main__":
    # Test data loading
    data, regimes = load_spy_data(years=5)
    
    print(f"\nData shape: {data.shape}")
    print(f"\nColumns: {list(data.columns)}")
    print(f"\nDate range: {data.index[0]} to {data.index[-1]}")
    
    print(f"\nRegime distribution:")
    detector = RegimeDetector()
    stats = detector.get_regime_stats(regimes)
    for regime, info in stats.items():
        print(f"  {regime}: {info['count']} days ({info['percentage']:.1f}%)")
