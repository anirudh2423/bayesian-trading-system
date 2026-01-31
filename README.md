# Bayesian Multi-Armed Bandit Trading System

An intelligent trading system that uses Thompson Sampling to dynamically select between competing trading strategies while minimizing Bayesian regret.

![Dashboard](bayesian_trading_dashboard.png)

## Overview

This project implements a decision-making engine based on the Multi-Armed Bandit (MAB) framework to optimize trading strategy selection. Unlike traditional backtesting which selects a static "best" strategy, this system learns online, adapting to changing market regimes (Bull, Bear, Ranging) in real-time.

The core algorithm, Thompson Sampling, treats the expected performance of each strategy as a random variable governed by a probability distribution. By constantly updating these beliefs with new market data, the system naturally balances:
*   **Exploration**: Trying strategies about which we have high uncertainty.
*   **Exploitation**: Selecting strategies that have performed well historically.

## Mathematical Foundation

The system models the trading problem as a stochastic bandit problem with $K$ arms (strategies) and imperfect information.

### 1. The Algorithm: Thompson Sampling
Thompson Sampling is a randomized probability matching algorithm. For each trading strategy $k \in \{1, ..., K\}$, we model the probability of receiving a positive reward (or the mean reward) as a parameter $\theta_k$.

We calculate the posterior distribution $P(\theta_k | \mathcal{D})$ based on observed data $\mathcal{D}$. At each time step $t$:
1.  Sample expected rewards $\hat{\theta}_k \sim P(\theta_k | \mathcal{D})$ for all strategies.
2.  Select the strategy with the highest sample: $a_t = \arg\max_k \hat{\theta}_k$.
3.  Observe the actual reward $r_t$.
4.  Update the posterior distribution with the new observation.

### 2. Beta-Bernoulli Conjugate Priors
Since our rewards are bounded in $[0, 1]$, we use the **Beta distribution** as the conjugate prior for the Bernoulli likelihood. This allows for efficient, closed-form Bayesian updates without expensive numerical integration.

*   **Prior**: $\theta_k \sim \text{Beta}(\alpha_k, \beta_k)$
*   **Likelihood**: $P(r | \theta_k) = \theta_k^r (1-\theta_k)^{1-r}$
*   **Posterior**: After observing reward $r_t$, the new parameters are:
    $$ \alpha_{new} = \alpha_{old} + r_t $$
    $$ \beta_{new} = \beta_{old} + (1 - r_t) $$

We initialize all strategies with a uniform prior $\text{Beta}(1, 1)$, implying no initial knowledge of performance.

### 3. Regret Analysis
The metric for success is **Bayesian Regret** ($R_T$), defined as the difference between the reward of the optimal strategy (Oracle) and the selected strategy over $T$ rounds:

$$ R_T = \sum_{t=1}^{T} \left( \mathbb{E}[r_t | a^*] - \mathbb{E}[r_t | a_t] \right) $$

where $a^*$ is the optimal arm.

Theoretical bounds for Thompson Sampling prove that expected regret grows logarithmically with time, or $O(\sqrt{KT \log T})$ in distribution-independent settings. This implies that the average regret per round tends to zero: $\lim_{T \to \infty} \frac{R_T}{T} = 0$, meaning the system converges to the optimal strategy.

## Empirical Results

Tested on 5 years of SPY (S&P 500 ETF) historical data (1,155 trading days).

### Performance Metrics
*   **Regret Ratio**: 23.44%
    *   Our system achieved a cumulative regret that was only 23.44% of the theoretical upper bound. This demonstrates efficient learning; the system identified optimal strategies faster than the worst-case theoretical prediction.
*   **Efficiency**: 86.4%
    *   The system captured 86.4% of the total cumulative return that a perfect Oracle (with hindsight) would have achieved.
*   **Convergence**:
    *   As seen in the posterior evolution charts, the probability distributions for strategy rewards started as flat lines (high uncertainty) and converged to sharp peaks (high confidence) as data accumulated.

## Usage

### Dependencies
Install the required Python packages:
```bash
pip install -r requirements.txt
```

### Running the Simulation
Execute the main script to run the simulation on historical data:
```bash
python main.py
```

## Project Structure

*   **main.py**: Orchestrator that runs the simulation loop.
*   **thompson_sampling.py**: Implementation of the Bayesian update engine using SciPy's Beta distribution.
*   **strategies.py**: Definition of Momentum, Mean Reversion, and Trend Following strategies.
*   **regret.py**: Engine for calculating instantaneous and cumulative regret against theoretical bounds.
*   **visualizations.py**: Matplotlib plotting code for generating the dashboard and analytics.

## License

MIT License
