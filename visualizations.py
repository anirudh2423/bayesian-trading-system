"""
Visualization Module - Five Interconnected Matplotlib Visualizations
Creates the cinematic visual narrative of Bayesian learning
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from scipy import stats
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style for premium dark theme
plt.style.use('dark_background')


class BayesianTradingVisualizer:
    """
    Creates all five visualizations for the Bayesian trading system:
    1. Regret accumulation vs theoretical bounds
    2. Strategy performance comparison
    3. Temporal selection heatmap
    4. Bayesian posterior evolution
    5. Performance attribution by regime
    """
    
    # Color palette
    COLORS = {
        'Momentum': '#f472b6',        # Pink
        'Mean Reversion': '#a78bfa',  # Purple
        'Trend Following': '#34d399', # Green
        'Oracle': '#fbbf24',          # Amber
        'Thompson': '#38bdf8',        # Cyan
        'regret': '#38bdf8',
        'bound': '#fbbf24',
        'BULL': '#4ade80',
        'BEAR': '#f87171',
        'RANGING': '#a78bfa',
        'VOLATILE': '#fbbf24'
    }
    
    def __init__(self, figsize: tuple = (20, 12)):
        self.figsize = figsize
        self.fig = None
        self.axes = {}
        
    def create_dashboard(self) -> plt.Figure:
        """Create the main dashboard layout with all 5 visualizations"""
        self.fig = plt.figure(figsize=self.figsize, facecolor='#0f172a')
        self.fig.suptitle(
            'Bayesian Multi-Armed Bandit Trading System\n'
            'Thompson Sampling Learning Journey',
            fontsize=16, fontweight='bold', color='white', y=0.98
        )
        
        # Create grid layout
        gs = GridSpec(3, 3, figure=self.fig, hspace=0.35, wspace=0.3,
                      left=0.06, right=0.94, top=0.90, bottom=0.06)
        
        # Viz 1: Regret (top-left, spans 2 columns)
        self.axes['regret'] = self.fig.add_subplot(gs[0, :2])
        self.axes['regret'].set_facecolor('#1e293b')
        
        # Viz 2: Performance (top-right)
        self.axes['performance'] = self.fig.add_subplot(gs[0, 2])
        self.axes['performance'].set_facecolor('#1e293b')
        
        # Viz 3: Heatmap (middle-left, spans 2 columns)
        self.axes['heatmap'] = self.fig.add_subplot(gs[1, :2])
        self.axes['heatmap'].set_facecolor('#1e293b')
        
        # Viz 4: Posterior (middle-right)
        self.axes['posterior'] = self.fig.add_subplot(gs[1, 2])
        self.axes['posterior'].set_facecolor('#1e293b')
        
        # Viz 5: Attribution (bottom, spans all columns)
        self.axes['attribution'] = self.fig.add_subplot(gs[2, :])
        self.axes['attribution'].set_facecolor('#1e293b')
        
        return self.fig
    
    def plot_regret(self, regret_data: Dict, ax: Optional[plt.Axes] = None) -> None:
        """
        Viz 1: Real-time regret accumulation vs theoretical bounds
        Shows cumulative regret staying below O(√(K·T·log(T))) bound
        """
        ax = ax or self.axes.get('regret')
        ax.clear()
        ax.set_facecolor('#1e293b')
        
        rounds = regret_data.get('rounds', [])
        cumulative = regret_data.get('cumulative', [])
        bound = regret_data.get('theoretical_bound', [])
        
        if not rounds:
            ax.text(0.5, 0.5, 'Waiting for data...', transform=ax.transAxes,
                   ha='center', va='center', fontsize=12, color='#94a3b8')
            return
        
        # Plot theoretical bound as filled area
        ax.fill_between(rounds, 0, bound, alpha=0.15, color=self.COLORS['bound'],
                        label=r'$O(\sqrt{K \cdot T \cdot \log(T)})$ Bound')
        ax.plot(rounds, bound, '--', color=self.COLORS['bound'], linewidth=2, alpha=0.8)
        
        # Plot cumulative regret
        ax.plot(rounds, cumulative, color=self.COLORS['regret'], linewidth=2.5,
                label='Cumulative Regret')
        
        # Highlight current point
        if len(rounds) > 0:
            ax.scatter([rounds[-1]], [cumulative[-1]], color=self.COLORS['regret'],
                      s=100, zorder=5, edgecolor='white', linewidth=2)
        
        # Stats box
        if len(cumulative) > 0:
            ratio = cumulative[-1] / bound[-1] * 100 if bound[-1] > 0 else 0
            color = '#4ade80' if ratio < 100 else '#f87171'
            stats_text = f'Regret: {cumulative[-1]:.1f}\nBound: {bound[-1]:.1f}\nRatio: {ratio:.1f}%'
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='#0f172a', edgecolor=color, alpha=0.9),
                   color=color, family='monospace')
        
        ax.set_xlabel('Round (t)', fontsize=10, color='#94a3b8')
        ax.set_ylabel('Cumulative Regret', fontsize=10, color='#94a3b8')
        ax.set_title('Regret Accumulation vs Theoretical Bound', fontsize=11, 
                    fontweight='bold', color='white', pad=10)
        ax.legend(loc='upper left', fontsize=8, framealpha=0.8)
        ax.grid(True, alpha=0.1)
        ax.tick_params(colors='#94a3b8')
        
    def plot_performance(self, performance_data: Dict, ax: Optional[plt.Axes] = None) -> None:
        """
        Viz 2: Strategy performance comparison against oracle baseline
        Shows cumulative rewards for all strategies + Thompson + Oracle
        """
        ax = ax or self.axes.get('performance')
        ax.clear()
        ax.set_facecolor('#1e293b')
        
        rounds = performance_data.get('rounds', [])
        
        if not rounds:
            ax.text(0.5, 0.5, 'Waiting for data...', transform=ax.transAxes,
                   ha='center', va='center', fontsize=12, color='#94a3b8')
            return
        
        # Plot oracle baseline (dashed)
        oracle = performance_data.get('oracle', [])
        if oracle:
            ax.plot(rounds, oracle, '--', color=self.COLORS['Oracle'], 
                   linewidth=2, label='Oracle (Best)', alpha=0.9)
        
        # Plot Thompson Sampling (prominent)
        thompson = performance_data.get('thompson', [])
        if thompson:
            ax.plot(rounds, thompson, color=self.COLORS['Thompson'],
                   linewidth=2.5, label='Thompson Sampling')
            ax.scatter([rounds[-1]], [thompson[-1]], color=self.COLORS['Thompson'],
                      s=80, zorder=5, edgecolor='white', linewidth=2)
        
        # Efficiency
        if oracle and thompson and len(oracle) > 0:
            efficiency = thompson[-1] / oracle[-1] * 100 if oracle[-1] > 0 else 0
            color = '#4ade80' if efficiency > 90 else '#fbbf24' if efficiency > 75 else '#f87171'
            ax.text(0.98, 0.02, f'Efficiency: {efficiency:.1f}%', transform=ax.transAxes,
                   fontsize=9, ha='right', va='bottom', color=color, fontweight='bold')
        
        ax.set_xlabel('Round (t)', fontsize=10, color='#94a3b8')
        ax.set_ylabel('Cumulative Reward', fontsize=10, color='#94a3b8')
        ax.set_title('Performance vs Oracle', fontsize=11,
                    fontweight='bold', color='white', pad=10)
        ax.legend(loc='upper left', fontsize=8, framealpha=0.8)
        ax.grid(True, alpha=0.1)
        ax.tick_params(colors='#94a3b8')
        
    def plot_heatmap(self, selection_data: Dict, ax: Optional[plt.Axes] = None) -> None:
        """
        Viz 3: Temporal selection heatmap showing exploration-exploitation balance
        Time windows on X, strategies on Y, color = selection frequency
        """
        ax = ax or self.axes.get('heatmap')
        ax.clear()
        ax.set_facecolor('#1e293b')
        
        selections = selection_data.get('selections', {})
        regimes = selection_data.get('regimes', [])
        
        if not selections or not any(len(v) > 0 for v in selections.values()):
            ax.text(0.5, 0.5, 'Waiting for data...', transform=ax.transAxes,
                   ha='center', va='center', fontsize=12, color='#94a3b8')
            return
        
        strategies = list(selections.keys())
        n_rounds = len(list(selections.values())[0])
        window_size = max(1, n_rounds // 50)  # ~50 windows
        n_windows = (n_rounds + window_size - 1) // window_size
        
        # Build frequency matrix
        freq_matrix = np.zeros((len(strategies), n_windows))
        
        for i, strategy in enumerate(strategies):
            sel_array = np.array(selections[strategy])
            for w in range(n_windows):
                start = w * window_size
                end = min((w + 1) * window_size, n_rounds)
                if end > start:
                    freq_matrix[i, w] = sel_array[start:end].mean()
        
        # Plot heatmap
        im = ax.imshow(freq_matrix, aspect='auto', cmap='viridis',
                      vmin=0, vmax=1, interpolation='nearest')
        
        # Plot regime bands at top
        if regimes:
            regime_colors = [self.COLORS.get(r, '#64748b') for r in regimes[::window_size]]
            for w, color in enumerate(regime_colors[:n_windows]):
                ax.add_patch(Rectangle((w - 0.5, -0.7), 1, 0.4, 
                            facecolor=color, alpha=0.6, clip_on=False))
        
        ax.set_yticks(range(len(strategies)))
        ax.set_yticklabels(strategies, fontsize=9, color='#e2e8f0')
        ax.set_xlabel('Time Window', fontsize=10, color='#94a3b8')
        ax.set_title('Strategy Selection Frequency Over Time', fontsize=11,
                    fontweight='bold', color='white', pad=15)
        ax.tick_params(colors='#94a3b8')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('Selection Frequency', fontsize=9, color='#94a3b8')
        cbar.ax.tick_params(colors='#94a3b8')
        
    def plot_posteriors(self, posteriors: Dict, ax: Optional[plt.Axes] = None) -> None:
        """
        Viz 4: Bayesian posterior evolution with shrinking credible intervals
        Shows Beta distributions for each arm
        """
        ax = ax or self.axes.get('posterior')
        ax.clear()
        ax.set_facecolor('#1e293b')
        
        if not posteriors:
            ax.text(0.5, 0.5, 'Waiting for data...', transform=ax.transAxes,
                   ha='center', va='center', fontsize=12, color='#94a3b8')
            return
        
        x = np.linspace(0.001, 0.999, 200)
        
        for name, params in posteriors.items():
            alpha = params['alpha']
            beta = params['beta']
            color = self.COLORS.get(name, '#64748b')
            
            # Plot Beta PDF
            y = stats.beta.pdf(x, alpha, beta)
            ax.fill_between(x, y, alpha=0.3, color=color)
            ax.plot(x, y, color=color, linewidth=2, label=f'{name} (α={alpha:.1f}, β={beta:.1f})')
            
            # Mark mean
            mean = alpha / (alpha + beta)
            ax.axvline(mean, color=color, linestyle='--', alpha=0.7, linewidth=1)
            
            # Mark 95% CI
            ci_low = stats.beta.ppf(0.025, alpha, beta)
            ci_high = stats.beta.ppf(0.975, alpha, beta)
            y_max = stats.beta.pdf(mean, alpha, beta)
            ax.plot([ci_low, ci_high], [y_max * 0.1] * 2, color=color, 
                   linewidth=4, alpha=0.5, solid_capstyle='round')
        
        ax.set_xlabel('Reward Probability', fontsize=10, color='#94a3b8')
        ax.set_ylabel('Density', fontsize=10, color='#94a3b8')
        ax.set_title('Posterior Distributions', fontsize=11,
                    fontweight='bold', color='white', pad=10)
        ax.legend(loc='upper right', fontsize=7, framealpha=0.8)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.1)
        ax.tick_params(colors='#94a3b8')
        
    def plot_attribution(self, attribution_data: Dict, ax: Optional[plt.Axes] = None) -> None:
        """
        Viz 5: Performance attribution across market regimes
        Grouped bar chart showing strategy performance per regime
        """
        ax = ax or self.axes.get('attribution')
        ax.clear()
        ax.set_facecolor('#1e293b')
        
        regime_performance = attribution_data.get('regime_performance', {})
        
        if not regime_performance:
            ax.text(0.5, 0.5, 'Waiting for data...', transform=ax.transAxes,
                   ha='center', va='center', fontsize=12, color='#94a3b8')
            return
        
        regimes = list(regime_performance.keys())
        strategies = list(list(regime_performance.values())[0].keys()) if regime_performance else []
        
        if not strategies:
            return
        
        x = np.arange(len(regimes))
        width = 0.2
        
        for i, strategy in enumerate(strategies):
            values = [regime_performance[r].get(strategy, 0) for r in regimes]
            offset = (i - len(strategies) / 2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, 
                         label=strategy, color=self.COLORS.get(strategy, '#64748b'),
                         alpha=0.85, edgecolor='white', linewidth=0.5)
        
        ax.set_xlabel('Market Regime', fontsize=10, color='#94a3b8')
        ax.set_ylabel('Average Reward', fontsize=10, color='#94a3b8')
        ax.set_title('Performance Attribution by Market Regime', fontsize=11,
                    fontweight='bold', color='white', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(regimes, fontsize=10, color='#e2e8f0')
        ax.legend(loc='upper right', fontsize=8, framealpha=0.8)
        ax.grid(True, alpha=0.1, axis='y')
        ax.tick_params(colors='#94a3b8')
        
        # Add regime colors to x-axis labels
        for i, regime in enumerate(regimes):
            ax.get_xticklabels()[i].set_color(self.COLORS.get(regime, '#94a3b8'))
    
    def update_all(self, data: Dict) -> None:
        """Update all visualizations with new data"""
        self.plot_regret(data.get('regret', {}))
        self.plot_performance(data.get('performance', {}))
        self.plot_heatmap(data.get('heatmap', {}))
        self.plot_posteriors(data.get('posteriors', {}))
        self.plot_attribution(data.get('attribution', {}))
        self.fig.canvas.draw_idle()
        
    def save(self, filename: str, dpi: int = 150) -> None:
        """Save the dashboard to file"""
        self.fig.savefig(filename, dpi=dpi, facecolor='#0f172a', 
                        edgecolor='none', bbox_inches='tight')
        print(f"Saved visualization to {filename}")
        
    def show(self) -> None:
        """Display the dashboard"""
        plt.show()


def create_standalone_figures(data: Dict, output_dir: str = '.') -> None:
    """Create individual high-resolution figures for each visualization"""
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    viz = BayesianTradingVisualizer()
    
    # Figure 1: Regret
    fig1, ax1 = plt.subplots(figsize=(12, 6), facecolor='#0f172a')
    ax1.set_facecolor('#1e293b')
    viz.plot_regret(data.get('regret', {}), ax1)
    fig1.tight_layout()
    fig1.savefig(f'{output_dir}/01_regret_analysis.png', dpi=200, facecolor='#0f172a')
    plt.close(fig1)
    
    # Figure 2: Performance
    fig2, ax2 = plt.subplots(figsize=(10, 6), facecolor='#0f172a')
    ax2.set_facecolor('#1e293b')
    viz.plot_performance(data.get('performance', {}), ax2)
    fig2.tight_layout()
    fig2.savefig(f'{output_dir}/02_performance_comparison.png', dpi=200, facecolor='#0f172a')
    plt.close(fig2)
    
    # Figure 3: Heatmap
    fig3, ax3 = plt.subplots(figsize=(14, 4), facecolor='#0f172a')
    ax3.set_facecolor('#1e293b')
    viz.plot_heatmap(data.get('heatmap', {}), ax3)
    fig3.tight_layout()
    fig3.savefig(f'{output_dir}/03_selection_heatmap.png', dpi=200, facecolor='#0f172a')
    plt.close(fig3)
    
    # Figure 4: Posteriors
    fig4, ax4 = plt.subplots(figsize=(10, 6), facecolor='#0f172a')
    ax4.set_facecolor('#1e293b')
    viz.plot_posteriors(data.get('posteriors', {}), ax4)
    fig4.tight_layout()
    fig4.savefig(f'{output_dir}/04_posterior_evolution.png', dpi=200, facecolor='#0f172a')
    plt.close(fig4)
    
    # Figure 5: Attribution
    fig5, ax5 = plt.subplots(figsize=(12, 5), facecolor='#0f172a')
    ax5.set_facecolor('#1e293b')
    viz.plot_attribution(data.get('attribution', {}), ax5)
    fig5.tight_layout()
    fig5.savefig(f'{output_dir}/05_regime_attribution.png', dpi=200, facecolor='#0f172a')
    plt.close(fig5)
    
    print(f"Saved 5 individual figures to {output_dir}/")


if __name__ == "__main__":
    # Demo with random data
    np.random.seed(42)
    n = 500
    
    demo_data = {
        'regret': {
            'rounds': list(range(1, n + 1)),
            'cumulative': np.cumsum(np.random.uniform(0, 0.3, n)).tolist(),
            'theoretical_bound': [2.5 * np.sqrt(3 * t * np.log(t + 1)) for t in range(1, n + 1)]
        },
        'performance': {
            'rounds': list(range(1, n + 1)),
            'oracle': np.cumsum(np.random.uniform(0.4, 0.7, n)).tolist(),
            'thompson': np.cumsum(np.random.uniform(0.35, 0.65, n)).tolist()
        },
        'heatmap': {
            'selections': {
                'Momentum': (np.random.random(n) > 0.5).tolist(),
                'Mean Reversion': (np.random.random(n) > 0.6).tolist(),
                'Trend Following': (np.random.random(n) > 0.55).tolist()
            },
            'regimes': np.random.choice(['BULL', 'BEAR', 'RANGING', 'VOLATILE'], n).tolist()
        },
        'posteriors': {
            'Momentum': {'alpha': 45, 'beta': 30},
            'Mean Reversion': {'alpha': 35, 'beta': 40},
            'Trend Following': {'alpha': 55, 'beta': 35}
        },
        'attribution': {
            'regime_performance': {
                'BULL': {'Momentum': 0.65, 'Mean Reversion': 0.45, 'Trend Following': 0.60},
                'BEAR': {'Momentum': 0.55, 'Mean Reversion': 0.50, 'Trend Following': 0.58},
                'RANGING': {'Momentum': 0.40, 'Mean Reversion': 0.62, 'Trend Following': 0.42},
                'VOLATILE': {'Momentum': 0.48, 'Mean Reversion': 0.52, 'Trend Following': 0.45}
            }
        }
    }
    
    viz = BayesianTradingVisualizer()
    viz.create_dashboard()
    viz.update_all(demo_data)
    plt.show()
