"""
Visualize the distributions used for data generation
Show user exactly what probability distributions were used
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Create figure with subplots
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle('Distribution Parameters Used for Synthetic Data Generation', fontsize=16, fontweight='bold')

# Industry parameters
industries = {
    'Wholesale (46)': {'mu': 19.0, 'sigma': 0.8, 'color': 'blue'},
    'Food Mfg (10)': {'mu': 19.5, 'sigma': 0.9, 'color': 'green'},
    'F&B (56)': {'mu': 18.8, 'sigma': 1.0, 'color': 'red'}
}

# Row 1: Revenue distributions (Lognormal)
for idx, (industry, params) in enumerate(industries.items()):
    ax = axes[0, idx]
    
    # Generate samples
    samples = np.random.lognormal(params['mu'], params['sigma'], 10000)
    samples_millions = samples / 1e6  # Convert to millions VND
    
    ax.hist(samples_millions, bins=50, alpha=0.7, color=params['color'], edgecolor='black')
    ax.axvline(samples_millions.mean(), color='darkred', linestyle='--', linewidth=2, 
               label=f'Mean: {samples_millions.mean():.1f}M')
    ax.set_title(f'{industry}\nRevenue Distribution', fontweight='bold')
    ax.set_xlabel('Revenue (Million VND)')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.set_xlim(0, 1000)

# Row 2: Profit Margin distributions
profit_margins = {
    'Wholesale (46)': 0.03,
    'Food Mfg (10)': 0.08,
    'F&B (56)': 0.05
}

for idx, (industry, pm) in enumerate(profit_margins.items()):
    ax = axes[1, idx]
    
    # Simulate variation (±20%)
    samples = np.random.uniform(pm * 0.8, pm * 1.2, 10000)
    
    ax.hist(samples * 100, bins=30, alpha=0.7, color=list(industries.values())[idx]['color'], 
            edgecolor='black')
    ax.axvline(pm * 100, color='darkred', linestyle='--', linewidth=2,
               label=f'Target: {pm*100:.1f}%')
    ax.set_title(f'{industry}\nProfit Margin', fontweight='bold')
    ax.set_xlabel('Profit Margin (%)')
    ax.set_ylabel('Frequency')
    ax.legend()

# Row 3: Current Ratio distributions
current_ratios = {
    'Wholesale (46)': 1.8,
    'Food Mfg (10)': 1.5,
    'F&B (56)': 1.3
}

for idx, (industry, cr) in enumerate(current_ratios.items()):
    ax = axes[2, idx]
    
    # Simulate with normal noise
    samples = np.random.normal(cr, 0.2, 10000)
    samples = samples[samples > 0]  # Remove negative
    
    ax.hist(samples, bins=30, alpha=0.7, color=list(industries.values())[idx]['color'],
            edgecolor='black')
    ax.axvline(cr, color='darkred', linestyle='--', linewidth=2,
               label=f'Target: {cr:.1f}')
    ax.set_title(f'{industry}\nCurrent Ratio', fontweight='bold')
    ax.set_xlabel('Current Ratio')
    ax.set_ylabel('Frequency')
    ax.legend()

plt.tight_layout()
plt.savefig('data/processed/distribution_visualizations.png', dpi=150, bbox_inches='tight')
print("✓ Saved: data/processed/distribution_visualizations.png")

# Create summary table
print("\n" + "="*70)
print("DISTRIBUTION PARAMETERS SUMMARY")
print("="*70)

summary_data = []
for industry, params in industries.items():
    summary_data.append({
        'Industry': industry,
        'Revenue μ (log)': params['mu'],
        'Revenue σ (log)': params['sigma'],
        'Mean Revenue (M)': f"{np.exp(params['mu'] + params['sigma']**2/2)/1e6:.1f}",
        'Profit Margin': f"{profit_margins[industry]*100:.1f}%",
        'Current Ratio': current_ratios[industry]
    })

df_summary = pd.DataFrame(summary_data)
print(df_summary.to_string(index=False))
print("="*70)

print("\n✅ Visualization complete!")
print("📊 Check: data/processed/distribution_visualizations.png")
