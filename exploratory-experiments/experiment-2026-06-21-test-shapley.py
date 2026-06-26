import json
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from itertools import combinations

# --- Parameters ---
CONFIDENCE = 0.8
N_BOOTSTRAPS = 1000

def compute_shapley_for_means(means_dict, mechanisms):
    """Helper function to calculate Shapley values given a dictionary of config means."""
    n = len(mechanisms)
    shapley_values = {m: 0.0 for m in mechanisms}
    
    def get_score(subset):
        config_name = "m5"
        if subset:
            config_name += "_" + "_".join(sorted(list(subset)))
        return means_dict.get(config_name, 0.0)

    for m in mechanisms:
        for size in range(n):
            for subset in combinations([x for x in mechanisms if x != m], size):
                weight = (math.factorial(size) * math.factorial(n - size - 1)) / math.factorial(n)
                mean_with = get_score(set(subset) | {m})
                mean_without = get_score(set(subset))
                shapley_values[m] += weight * (mean_with - mean_without)
                
    return shapley_values

def calculate_shapley_bootstrap(file_path, confidence=0.80, n_bootstraps=1000):
    # 1. Load and parse raw data
    with open(file_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    
    # 2. Extract unique mechanisms (excluding base 'm5')
    all_mechanisms = set()
    for config in df['mechanism_config'].unique():
        parts = config.split('_')
        all_mechanisms.update([p for p in parts if p != 'm5'])
    mechanisms = sorted(list(all_mechanisms))
    
    # 3. Base Effect Sizes (Calculate exactly once on the original, un-resampled data)
    original_means = df.groupby('mechanism_config')['score'].mean().to_dict()
    base_shapley = compute_shapley_for_means(original_means, mechanisms)
    
    # 4. Bootstrapping
    # Pre-group data by config to speed up the resampling loop
    grouped = df.groupby('mechanism_config')
    bootstrap_shapley_results = {m: [] for m in mechanisms}
    
    for _ in range(n_bootstraps):
        resampled_means = {}
        # Resample within each specific mechanism configuration (Stratified Resampling)
        for name, group in grouped:
            scores = group['score'].values
            resampled_scores = np.random.choice(scores, size=len(scores), replace=True)
            resampled_means[name] = np.mean(resampled_scores)
            
        # Calculate Shapley for this specific resampled "parallel universe"
        b_shapley = compute_shapley_for_means(resampled_means, mechanisms)
        for m in mechanisms:
            bootstrap_shapley_results[m].append(b_shapley[m])

    # 5. Calculate Empirical Confidence Intervals
    alpha = 1.0 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1.0 - (alpha / 2)) * 100
    
    final_results = {}
    for m in mechanisms:
        b_values = np.array(bootstrap_shapley_results[m])
        
        # Grab exact percentiles from our 1,000 simulated iterations
        ci_lower = np.percentile(b_values, lower_percentile)
        ci_upper = np.percentile(b_values, upper_percentile)
        
        # Calculate asymmetric margin of errors for plotting
        moe_lower = base_shapley[m] - ci_lower
        moe_upper = ci_upper - base_shapley[m]
        
        final_results[m] = {
            'shapley_value': base_shapley[m],
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'moe_lower': moe_lower,
            'moe_upper': moe_upper
        }
        
    return final_results

def plot_shapley_results(shapley_results, confidence):
    # Mapping mechanism codes to human-readable names
    name_map = {
        'm2': 'Memory',
        'm3': 'State Externalisation',
        'm6': 'Planning'
    }
    
    mechanisms = []
    means = []
    errors_lower = []
    errors_upper = []
    
    for m, metrics in shapley_results.items():
        mechanisms.append(name_map.get(m, m))
        means.append(metrics['shapley_value'])
        errors_lower.append(metrics['moe_lower'])
        errors_upper.append(metrics['moe_upper'])
        
    # Sort the data by effect size (descending order)
    sorted_indices = np.argsort(means)[::-1]
    
    sorted_mechanisms = [mechanisms[i] for i in sorted_indices]
    sorted_means = [means[i] for i in sorted_indices]
    
    # Asymmetric error arrays need to be sorted to match the bars
    sorted_err_lower = [errors_lower[i] for i in sorted_indices]
    sorted_err_upper = [errors_upper[i] for i in sorted_indices]
    
    # Matplotlib expects a 2xN array for asymmetric error bars: [lower_errors, upper_errors]
    asymmetric_errors = [sorted_err_lower, sorted_err_upper]
    
    # Create the Bar Plot
    plt.figure(figsize=(9, 6))
    
    bars = plt.bar(
        sorted_mechanisms, 
        sorted_means, 
        yerr=asymmetric_errors, 
        capsize=8, 
        color='#4C72B0', 
        edgecolor='black',
        alpha=0.8
    )
    
    # Formatting
    ci_percentage = int(confidence * 100)
    plt.title(f'Mechanism Effect Sizes ({ci_percentage}% Bootstrap CI)', fontsize=14, pad=15)
    plt.ylabel('Effect Size (Mean Score Contribution)', fontsize=12)
    plt.xlabel('Mechanism', fontsize=12)
    
    plt.axhline(0, color='black', linewidth=1.2, linestyle='--')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    output_filename = 'shapley_effect_sizes_bootstrap.png'
    plt.savefig(output_filename, dpi=300)
    print(f"\nPlot successfully saved to: {output_filename}")

if __name__ == "__main__":
    file_path = 'inspect_data/experiments/experiment-output/powerset_dataset_gemini-flash-lite-latest_2026-06-19_00-50-07.json'
    
    print(f"Calculating Shapley values using {N_BOOTSTRAPS} bootstrapped iterations...")
    shapley_results = calculate_shapley_bootstrap(file_path, confidence=CONFIDENCE, n_bootstraps=N_BOOTSTRAPS)
    
    print(f"\n--- Shapley Values ({int(CONFIDENCE * 100)}% Bootstrap CI) ---")
    for m, metrics in shapley_results.items():
        val = metrics['shapley_value']
        lower = metrics['ci_lower']
        upper = metrics['ci_upper']
        print(f"{m}: {val:.4f}  (CI: [{lower:.4f}, {upper:.4f}])")
        
    plot_shapley_results(shapley_results, CONFIDENCE)
