"""
Thompson Sampling Engine for Multi-Armed Bandit Trading
Uses Beta-distributed priors for Bayesian strategy selection
"""

import numpy as np
from scipy import stats
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


@dataclass
class Arm:
    """Represents a single arm (strategy) in the bandit"""
    name: str
    alpha: float = 1.0  # Beta prior parameter
    beta: float = 1.0   # Beta prior parameter
    selections: int = 0
    total_reward: float = 0.0
    rewards: List[float] = field(default_factory=list)
    
    @property
    def mean(self) -> float:
        """Expected value of Beta distribution"""
        return self.alpha / (self.alpha + self.beta)
    
    @property
    def variance(self) -> float:
        """Variance of Beta distribution"""
        ab = self.alpha + self.beta
        return (self.alpha * self.beta) / (ab ** 2 * (ab + 1))
    
    @property
    def std(self) -> float:
        """Standard deviation"""
        return np.sqrt(self.variance)


class ThompsonSamplingEngine:
    """
    Thompson Sampling for multi-armed bandits with Beta-Bernoulli conjugate priors.
    
    The algorithm:
    1. For each arm, maintain Beta(α, β) posterior
    2. Sample from each posterior
    3. Select arm with highest sample
    4. Observe reward r ∈ [0, 1]
    5. Update: α += r, β += (1 - r)
    """
    
    def __init__(self, strategy_names: List[str]):
        self.arms: Dict[str, Arm] = {
            name: Arm(name=name) for name in strategy_names
        }
        self.history: List[Dict] = []
        self.round = 0
        
    def sample(self, arm_name: str) -> float:
        """Sample from the Beta posterior of an arm"""
        arm = self.arms[arm_name]
        return stats.beta.rvs(arm.alpha, arm.beta)
    
    def select_arm(self) -> Tuple[str, Dict[str, float]]:
        """
        Select the best arm using Thompson Sampling.
        Returns: (selected_arm_name, {arm_name: sampled_value})
        """
        samples = {name: self.sample(name) for name in self.arms}
        best_arm = max(samples, key=samples.get)
        return best_arm, samples
    
    def update(self, arm_name: str, reward: float) -> None:
        """
        Update the posterior after observing a reward.
        
        Bayesian update for Beta-Bernoulli:
        - α_new = α + reward
        - β_new = β + (1 - reward)
        """
        arm = self.arms[arm_name]
        
        # Bayesian update
        arm.alpha += reward
        arm.beta += (1 - reward)
        arm.selections += 1
        arm.total_reward += reward
        arm.rewards.append(reward)
        
        self.round += 1
        
        # Record history for visualization
        self.history.append({
            'round': self.round,
            'selected_arm': arm_name,
            'reward': reward,
            'posteriors': self.get_posteriors()
        })
    
    def get_posteriors(self) -> Dict[str, Dict]:
        """Get current posterior parameters for all arms"""
        return {
            name: {
                'alpha': arm.alpha,
                'beta': arm.beta,
                'mean': arm.mean,
                'variance': arm.variance,
                'selections': arm.selections
            }
            for name, arm in self.arms.items()
        }
    
    def get_credible_interval(self, arm_name: str, confidence: float = 0.95) -> Tuple[float, float, float]:
        """
        Get credible interval for an arm's true reward probability.
        Returns: (lower, upper, mean)
        """
        arm = self.arms[arm_name]
        lower = stats.beta.ppf((1 - confidence) / 2, arm.alpha, arm.beta)
        upper = stats.beta.ppf(1 - (1 - confidence) / 2, arm.alpha, arm.beta)
        return lower, upper, arm.mean
    
    def get_beta_pdf(self, arm_name: str, x: np.ndarray) -> np.ndarray:
        """Get Beta PDF values for plotting"""
        arm = self.arms[arm_name]
        return stats.beta.pdf(x, arm.alpha, arm.beta)
    
    def get_selection_history(self) -> Dict[str, List[bool]]:
        """Get selection history as boolean arrays for each arm"""
        selections = {name: [] for name in self.arms}
        for h in self.history:
            for name in self.arms:
                selections[name].append(h['selected_arm'] == name)
        return selections
    
    def get_stats(self) -> Dict:
        """Get summary statistics"""
        return {
            'round': self.round,
            'arms': [
                {
                    'name': arm.name,
                    'selections': arm.selections,
                    'total_reward': arm.total_reward,
                    'avg_reward': arm.total_reward / arm.selections if arm.selections > 0 else 0,
                    'alpha': arm.alpha,
                    'beta': arm.beta,
                    'mean': arm.mean
                }
                for arm in self.arms.values()
            ]
        }
    
    def reset(self) -> None:
        """Reset engine to initial state"""
        for arm in self.arms.values():
            arm.alpha = 1.0
            arm.beta = 1.0
            arm.selections = 0
            arm.total_reward = 0.0
            arm.rewards = []
        self.history = []
        self.round = 0


if __name__ == "__main__":
    # Quick test
    engine = ThompsonSamplingEngine(['Momentum', 'Mean Reversion', 'Trend Following'])
    
    # Simulate some rounds
    for _ in range(100):
        arm, samples = engine.select_arm()
        # Fake reward - in practice this comes from strategy execution
        reward = np.random.beta(2, 2)  
        engine.update(arm, reward)
    
    print("Stats after 100 rounds:")
    for arm_stat in engine.get_stats()['arms']:
        print(f"  {arm_stat['name']}: {arm_stat['selections']} selections, "
              f"avg reward: {arm_stat['avg_reward']:.3f}, "
              f"posterior mean: {arm_stat['mean']:.3f}")
