"""
Regret Calculation Engine
Computes Bayesian regret with theoretical bounds for Thompson Sampling
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass, field


@dataclass
class RegretData:
    """Container for regret tracking data"""
    instantaneous: List[float] = field(default_factory=list)
    cumulative: List[float] = field(default_factory=list)
    oracle_rewards: List[float] = field(default_factory=list)
    actual_rewards: List[float] = field(default_factory=list)
    theoretical_bounds: List[float] = field(default_factory=list)


class RegretEngine:
    """
    Calculates and tracks regret for multi-armed bandit algorithms.
    
    Regret = sum of (oracle_reward - actual_reward) over all rounds
    
    Thompson Sampling theoretical bound: O(√(K * T * log(T)))
    where K = number of arms, T = number of rounds
    """
    
    def __init__(self, num_arms: int, bound_constant: float = 2.5):
        self.num_arms = num_arms
        self.bound_constant = bound_constant  # Empirically tuned constant
        self.data = RegretData()
        self.round = 0
        
    def record(self, oracle_reward: float, actual_reward: float) -> None:
        """
        Record a round's rewards and update regret.
        
        Args:
            oracle_reward: Best possible reward (hindsight)
            actual_reward: Reward from selected arm
        """
        self.round += 1
        
        # Calculate instantaneous regret
        instant_regret = max(0, oracle_reward - actual_reward)
        self.data.instantaneous.append(instant_regret)
        
        # Calculate cumulative regret
        cum_regret = sum(self.data.instantaneous)
        self.data.cumulative.append(cum_regret)
        
        # Calculate theoretical bound
        bound = self.theoretical_bound(self.round)
        self.data.theoretical_bounds.append(bound)
        
        # Store rewards
        self.data.oracle_rewards.append(oracle_reward)
        self.data.actual_rewards.append(actual_reward)
    
    def theoretical_bound(self, t: int) -> float:
        """
        Calculate theoretical upper bound on regret at round t.
        
        Thompson Sampling achieves O(√(K * T * log(T))) regret.
        """
        if t <= 0:
            return 0.0
        
        K = self.num_arms
        return self.bound_constant * np.sqrt(K * t * np.log(t + 1))
    
    def get_regret_ratio(self) -> float:
        """
        Get ratio of actual cumulative regret to theoretical bound.
        Values < 1 indicate algorithm is performing better than worst-case.
        """
        if self.round == 0:
            return 0.0
        
        actual = self.data.cumulative[-1]
        bound = self.data.theoretical_bounds[-1]
        
        return actual / bound if bound > 0 else 0.0
    
    def is_sublinear(self, min_rounds: int = 100) -> bool:
        """
        Check if regret growth is sublinear (flattening over time).
        
        Compares growth rate in first half vs second half.
        """
        if self.round < min_rounds:
            return None  # Not enough data
        
        midpoint = self.round // 2
        
        # Growth rate = regret / rounds for each half
        first_half_rate = self.data.cumulative[midpoint - 1] / midpoint
        second_half_rate = (
            (self.data.cumulative[-1] - self.data.cumulative[midpoint - 1]) / 
            (self.round - midpoint)
        )
        
        return second_half_rate < first_half_rate
    
    def get_moving_average(self, window: int = 50) -> np.ndarray:
        """Calculate moving average of instantaneous regret"""
        if self.round < window:
            return None
        
        regrets = np.array(self.data.instantaneous)
        return np.convolve(regrets, np.ones(window) / window, mode='valid')
    
    def get_stats(self) -> Dict:
        """Get summary statistics"""
        if self.round == 0:
            return {'round': 0}
        
        cum_regret = self.data.cumulative[-1]
        
        return {
            'round': self.round,
            'cumulative_regret': cum_regret,
            'average_regret': cum_regret / self.round,
            'theoretical_bound': self.data.theoretical_bounds[-1],
            'regret_ratio': self.get_regret_ratio(),
            'is_sublinear': self.is_sublinear(),
            'total_oracle_reward': sum(self.data.oracle_rewards),
            'total_actual_reward': sum(self.data.actual_rewards),
            'efficiency': sum(self.data.actual_rewards) / sum(self.data.oracle_rewards) * 100 
                         if sum(self.data.oracle_rewards) > 0 else 0
        }
    
    def get_chart_data(self) -> Dict:
        """Get data formatted for plotting"""
        return {
            'rounds': list(range(1, self.round + 1)),
            'cumulative': self.data.cumulative.copy(),
            'instantaneous': self.data.instantaneous.copy(),
            'theoretical_bound': self.data.theoretical_bounds.copy(),
            'oracle': self.data.oracle_rewards.copy(),
            'actual': self.data.actual_rewards.copy()
        }
    
    def reset(self) -> None:
        """Reset the engine"""
        self.data = RegretData()
        self.round = 0


if __name__ == "__main__":
    # Test regret calculation
    engine = RegretEngine(num_arms=3)
    
    np.random.seed(42)
    for _ in range(500):
        oracle_reward = np.random.beta(3, 2)  # Oracle always picks well
        actual_reward = np.random.beta(2, 2)  # Algorithm learns
        engine.record(oracle_reward, actual_reward)
    
    stats = engine.get_stats()
    print("Regret Statistics after 500 rounds:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
