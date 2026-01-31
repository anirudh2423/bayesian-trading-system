"""
Main Simulation Script - Bayesian Multi-Armed Bandit Trading System
Runs Thompson Sampling on SPY 5-year data with complete visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Tuple

# Local imports
from thompson_sampling import ThompsonSamplingEngine
from strategies import (
    create_strategies, 
    MomentumStrategy, 
    MeanReversionStrategy, 
    TrendFollowingStrategy,
    Oracle
)
from market_data import load_spy_data, MarketRegime
from regret import RegretEngine
from visualizations import BayesianTradingVisualizer, create_standalone_figures


class BayesianTradingSimulation:
    """
    Main simulation class that orchestrates:
    - Data loading and preprocessing
    - Thompson Sampling strategy selection
    - Regret tracking and analysis
    - Visualization generation
    """
    
    def __init__(self, years: int = 5, start_idx: int = 50):
        """
        Initialize simulation.
        
        Args:
            years: Years of historical SPY data to use
            start_idx: Starting index (need history for indicators)
        """
        print("=" * 60)
        print("BAYESIAN MULTI-ARMED BANDIT TRADING SYSTEM")
        print("Thompson Sampling for Optimal Strategy Selection")
        print("=" * 60)
        
        self.years = years
        self.start_idx = start_idx
        
        # Load data
        print("\n[1/5] Loading SPY market data...")
        self.data, self.regimes = load_spy_data(years=years)
        print(f"      Loaded {len(self.data)} trading days")
        
        # Create strategies
        print("\n[2/5] Initializing trading strategies...")
        self.strategies = create_strategies()
        self.strategy_names = [s.name for s in self.strategies]
        print(f"      Strategies: {', '.join(self.strategy_names)}")
        
        # Create Thompson Sampling engine
        print("\n[3/5] Initializing Thompson Sampling engine...")
        self.thompson = ThompsonSamplingEngine(self.strategy_names)
        
        # Create Oracle (perfect hindsight)
        self.oracle = Oracle(self.strategies)
        
        # Create Regret engine
        print("\n[4/5] Initializing regret tracking...")
        self.regret_engine = RegretEngine(num_arms=len(self.strategies))
        
        # Initialize tracking
        self.results = {
            'rounds': [],
            'selected_strategy': [],
            'rewards': [],
            'oracle_rewards': [],
            'oracle_strategy': [],
            'regimes': [],
            'cumulative_rewards': {'Thompson': []},
            'posteriors_history': []
        }
        
        for name in self.strategy_names:
            self.results['cumulative_rewards'][name] = []
        self.results['cumulative_rewards']['Oracle'] = []
        
        # Track per-regime performance
        self.regime_rewards = defaultdict(lambda: defaultdict(list))
        
        print("\n[5/5] Initialization complete!")
        
    def run(self, max_rounds: int = None, verbose: bool = True) -> Dict:
        """
        Run the simulation.
        
        Args:
            max_rounds: Maximum rounds to simulate (None = all available)
            verbose: Print progress
            
        Returns:
            Dictionary with all results
        """
        n_rounds = len(self.data) - self.start_idx - 1
        if max_rounds:
            n_rounds = min(n_rounds, max_rounds)
        
        print(f"\n{'='*60}")
        print(f"RUNNING SIMULATION: {n_rounds} rounds")
        print(f"{'='*60}\n")
        
        # Cumulative trackers
        cumulative = {name: 0.0 for name in self.strategy_names}
        cumulative['Thompson'] = 0.0
        cumulative['Oracle'] = 0.0
        
        for round_num in range(n_rounds):
            idx = self.start_idx + round_num
            
            # Get current market state
            regime = self.regimes[idx].value
            
            # Thompson Sampling: Select strategy
            selected_name, samples = self.thompson.select_arm()
            selected_strategy = next(s for s in self.strategies if s.name == selected_name)
            
            # Execute selected strategy and get reward
            reward = selected_strategy.calculate_reward(self.data, idx)
            
            # Get oracle reward (best possible)
            oracle_reward, oracle_strategy = self._get_oracle_reward(idx)
            
            # Update Thompson Sampling posterior
            self.thompson.update(selected_name, reward)
            
            # Update regret tracking
            self.regret_engine.record(oracle_reward, reward)
            
            # Track cumulative rewards
            cumulative['Thompson'] += reward
            cumulative['Oracle'] += oracle_reward
            
            # Track individual strategy performance (what they WOULD have earned)
            for strategy in self.strategies:
                strat_reward = strategy.calculate_reward(self.data, idx)
                cumulative[strategy.name] += strat_reward
            
            # Record results
            self.results['rounds'].append(round_num + 1)
            self.results['selected_strategy'].append(selected_name)
            self.results['rewards'].append(reward)
            self.results['oracle_rewards'].append(oracle_reward)
            self.results['oracle_strategy'].append(oracle_strategy)
            self.results['regimes'].append(regime)
            self.results['posteriors_history'].append(self.thompson.get_posteriors())
            
            for name in list(cumulative.keys()):
                self.results['cumulative_rewards'][name].append(cumulative[name])
            
            # Track regime-specific performance
            self.regime_rewards[regime][selected_name].append(reward)
            
            # Progress
            if verbose and (round_num + 1) % 200 == 0:
                stats = self.regret_engine.get_stats()
                print(f"Round {round_num + 1:4d}/{n_rounds} | "
                      f"Regime: {regime:8s} | "
                      f"Selected: {selected_name:15s} | "
                      f"Regret Ratio: {stats['regret_ratio']:.2%}")
        
        print(f"\n{'='*60}")
        print("SIMULATION COMPLETE")
        print(f"{'='*60}")
        
        return self.results
    
    def _get_oracle_reward(self, idx: int) -> Tuple[float, str]:
        """Get oracle's best possible reward"""
        best_reward = 0
        best_strategy = None
        
        for strategy in self.strategies:
            reward = strategy.calculate_reward(self.data, idx)
            if reward > best_reward:
                best_reward = reward
                best_strategy = strategy.name
        
        return best_reward, best_strategy
    
    def get_visualization_data(self) -> Dict:
        """Prepare data for all 5 visualizations"""
        
        # Viz 1: Regret data
        regret_data = self.regret_engine.get_chart_data()
        
        # Viz 2: Performance data
        performance_data = {
            'rounds': self.results['rounds'],
            'oracle': self.results['cumulative_rewards']['Oracle'],
            'thompson': self.results['cumulative_rewards']['Thompson']
        }
        
        # Viz 3: Selection heatmap data
        selections = self.thompson.get_selection_history()
        heatmap_data = {
            'selections': selections,
            'regimes': self.results['regimes']
        }
        
        # Viz 4: Posterior data (current state)
        posteriors = self.thompson.get_posteriors()
        
        # Viz 5: Attribution data
        regime_performance = {}
        for regime, strategies in self.regime_rewards.items():
            regime_performance[regime] = {}
            for strategy, rewards in strategies.items():
                regime_performance[regime][strategy] = np.mean(rewards) if rewards else 0
        
        # Fill missing strategies with 0
        for regime in regime_performance:
            for strategy in self.strategy_names:
                if strategy not in regime_performance[regime]:
                    regime_performance[regime][strategy] = 0
        
        attribution_data = {'regime_performance': regime_performance}
        
        return {
            'regret': regret_data,
            'performance': performance_data,
            'heatmap': heatmap_data,
            'posteriors': posteriors,
            'attribution': attribution_data
        }
    
    def print_summary(self) -> None:
        """Print detailed summary of results"""
        print("\n" + "=" * 60)
        print("SIMULATION SUMMARY")
        print("=" * 60)
        
        stats = self.regret_engine.get_stats()
        
        print(f"\n[STATS] OVERALL STATISTICS:")
        print(f"   Total Rounds: {stats['round']}")
        print(f"   Cumulative Regret: {stats['cumulative_regret']:.2f}")
        print(f"   Theoretical Bound: {stats['theoretical_bound']:.2f}")
        print(f"   Regret Ratio: {stats['regret_ratio']:.2%}")
        print(f"   Is Sublinear: {stats['is_sublinear']}")
        print(f"   Efficiency: {stats['efficiency']:.1f}%")
        
        print(f"\n[SELECTION] STRATEGY SELECTION BREAKDOWN:")
        ts_stats = self.thompson.get_stats()
        for arm in ts_stats['arms']:
            pct = arm['selections'] / stats['round'] * 100 if stats['round'] > 0 else 0
            print(f"   {arm['name']:15s}: {arm['selections']:4d} selections ({pct:5.1f}%) | "
                  f"Avg Reward: {arm['avg_reward']:.3f} | "
                  f"Posterior Mean: {arm['mean']:.3f}")
        
        print(f"\n[REWARDS] CUMULATIVE REWARDS:")
        final = self.results['cumulative_rewards']
        print(f"   Thompson Sampling: {final['Thompson'][-1]:.2f}")
        print(f"   Oracle (Best):     {final['Oracle'][-1]:.2f}")
        for name in self.strategy_names:
            print(f"   {name:18s}: {final[name][-1]:.2f}")
        
        print(f"\n[REGIMES] REGIME DISTRIBUTION:")
        regime_counts = pd.Series(self.results['regimes']).value_counts()
        for regime, count in regime_counts.items():
            pct = count / len(self.results['regimes']) * 100
            print(f"   {regime:10s}: {count:4d} days ({pct:.1f}%)")
        
        print("\n" + "=" * 60)
    
    def visualize(self, save_path: str = None, show: bool = True) -> None:
        """Create and optionally save the visualization dashboard"""
        
        print("\n[VIZ] Generating visualizations...")
        
        viz_data = self.get_visualization_data()
        
        visualizer = BayesianTradingVisualizer(figsize=(20, 12))
        visualizer.create_dashboard()
        visualizer.update_all(viz_data)
        
        if save_path:
            visualizer.save(save_path)
            print(f"   Dashboard saved to: {save_path}")
            
            # Also save individual high-res figures
            import os
            output_dir = os.path.dirname(save_path) or '.'
            create_standalone_figures(viz_data, output_dir)
        
        if show:
            print("   Displaying dashboard...")
            visualizer.show()


def main():
    """Main entry point"""
    
    # Create and run simulation
    sim = BayesianTradingSimulation(years=5, start_idx=50)
    
    # Run simulation
    results = sim.run(max_rounds=None, verbose=True)
    
    # Print summary
    sim.print_summary()
    
    # Generate visualizations (save only, don't try to display)
    sim.visualize(
        save_path='bayesian_trading_dashboard.png',
        show=False  # Set to True if you want interactive display
    )
    
    return sim


if __name__ == "__main__":
    sim = main()
