import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import glob
import numpy as np


def load_results_by_pattern(results_dir, pattern='*_results.csv'):
    """
    Load all result CSV files matching pattern from directory.
    
    Args:
        results_dir: Directory containing result CSV files
        pattern: Glob pattern for CSV files
    Returns:
        DataFrame with all results combined
    """
    filepath_pattern = os.path.join(results_dir, pattern)
    result_files = glob.glob(filepath_pattern)
    
    if not result_files:
        print(f"Warning: No files found matching pattern: {filepath_pattern}")
        return pd.DataFrame()
    
    dfs = []
    for filepath in result_files:
        try:
            df = pd.read_csv(filepath)
            dfs.append(df)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
    
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


def plot_model_comparison(df, output_dir, metric='test_f1_macro'):
    """
    Create comprehensive comparison plot for all models and preprocessing strategies.
    
    Args:
        df: DataFrame with results
        output_dir: Directory to save plot
        metric: Metric to plot
    """
    if df.empty:
        print("No data to plot")
        return
    
    # Get unique preprocessing strategies and models
    preprocessing_strategies = sorted(df['preprocessing'].unique())
    models = sorted(df['model'].unique())
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(preprocessing_strategies))
    width = 0.8 / len(models)  # Dynamic width based on number of models
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    
    for i, model in enumerate(models):
        model_data = df[df['model'] == model].sort_values('preprocessing')
        if not model_data.empty:
            values = [model_data[model_data['preprocessing']==prep][metric].values[0] 
                     if prep in model_data['preprocessing'].values else 0
                     for prep in preprocessing_strategies]
            
            offset = (i - len(models)/2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, 
                         label=model, color=colors[i], alpha=0.85)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}',
                           ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Preprocessing Strategy', fontsize=13, fontweight='bold')
    ax.set_ylabel(f'{metric.replace("_", " ").title()}', fontsize=13, fontweight='bold')
    ax.set_title('Model Performance Comparison Across Preprocessing Strategies', 
                fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([p.capitalize() for p in preprocessing_strategies], fontsize=11)
    ax.legend(fontsize=10, loc='best', ncol=2)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'all_models_{metric}_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Model comparison plot saved to {output_path}")


def plot_comprehensive_heatmap(df, output_dir, metric='test_f1_macro'):
    """
    Create heatmap for all models and preprocessing strategies.
    
    Args:
        df: DataFrame with results
        output_dir: Directory to save plot
        metric: Metric to visualize
    """
    if df.empty:
        print("No data to plot")
        return
    
    # Create pivot table
    heatmap_data = df.pivot_table(
        values=metric,
        index='model',
        columns='preprocessing',
        aggfunc='first'
    )
    
    # Ensure proper column ordering
    preprocessing_order = ['raw', 'clean', 'aggressive']
    existing_cols = [col for col in preprocessing_order if col in heatmap_data.columns]
    heatmap_data = heatmap_data[existing_cols]
    
    plt.figure(figsize=(10, max(6, len(heatmap_data) * 0.8)))
    sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='RdYlGn', 
                cbar_kws={'label': metric.replace('_', ' ').title()},
                linewidths=2, linecolor='white',
                vmin=heatmap_data.min().min() * 0.95,
                vmax=heatmap_data.max().max() * 1.02)
    
    plt.title(f'{metric.replace("_", " ").title()} Heatmap: All Models vs Preprocessing', 
              fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Preprocessing Strategy', fontsize=12, fontweight='bold')
    plt.ylabel('Model', fontsize=12, fontweight='bold')
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'all_models_{metric}_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved to {output_path}")


def plot_best_models_summary(df, output_dir):
    """
    Create summary plot showing best configuration for each model family.
    
    Args:
        df: DataFrame with results
        output_dir: Directory to save plot
    """
    if df.empty:
        print("No data to plot")
        return
    
    # Get best configuration for each model
    best_configs = df.loc[df.groupby('model')['test_f1_macro'].idxmax()]
    best_configs = best_configs.sort_values('test_f1_macro', ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, max(6, len(best_configs) * 0.5)))
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(best_configs)))
    bars = ax.barh(range(len(best_configs)), best_configs['test_f1_macro'], 
                   color=colors, alpha=0.85)
    
    # Add value and preprocessing labels
    for i, (bar, (_, row)) in enumerate(zip(bars, best_configs.iterrows())):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
               f' {width:.4f} ({row["preprocessing"]})',
               ha='left', va='center', fontsize=10, fontweight='bold')
    
    ax.set_yticks(range(len(best_configs)))
    ax.set_yticklabels(best_configs['model'], fontsize=11)
    ax.set_xlabel('Test Macro F1-Score', fontsize=13, fontweight='bold')
    ax.set_title('Best Configuration for Each Model (with preprocessing strategy)', 
                fontsize=14, fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_xlim(0, max(best_configs['test_f1_macro']) * 1.1)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'best_models_summary.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Best models summary saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate comprehensive visualization plots for all experiment results."
    )
    parser.add_argument("--results_dir", type=str, default="results/",
                       help="Directory containing result CSV files")
    parser.add_argument("--output_dir", type=str, default="results/",
                       help="Directory to save visualization plots")
    parser.add_argument("--pattern", type=str, default="*_results.csv",
                       help="Pattern to match result files")
    
    args = parser.parse_args()
    
    print("="*70)
    print("GENERATING COMPREHENSIVE VISUALIZATIONS FOR ALL EXPERIMENTS")
    print("="*70)
    
    # Load data
    print(f"\nLoading results from {args.results_dir}...")
    df = load_results_by_pattern(args.results_dir, args.pattern)
    
    if df.empty:
        print("No results found. Exiting.")
        return
    
    print(f"Loaded {len(df)} experiment results")
    print(f"Models found: {', '.join(df['model'].unique())}")
    print(f"Preprocessing strategies: {', '.join(df['preprocessing'].unique())}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate plots
    print("\nGenerating plots...")
    plot_model_comparison(df, args.output_dir, 'test_f1_macro')
    plot_comprehensive_heatmap(df, args.output_dir, 'test_f1_macro')
    plot_best_models_summary(df, args.output_dir)
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETED")
    print("="*70)
    print(f"\nAll plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
