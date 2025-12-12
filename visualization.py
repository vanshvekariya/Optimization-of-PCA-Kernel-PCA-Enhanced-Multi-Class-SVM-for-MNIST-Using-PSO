"""
Visualization Module
====================
This module provides advanced visualization utilities for the MNIST classification project.
Includes functions for plotting results, comparisons, and generating figures for the report.

"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import json


class ResultsVisualizer:
    """
    Class for creating comprehensive visualizations of experimental results.
    """
    
    def __init__(self, style='seaborn-v0_8-darkgrid'):
        """
        Initialize the visualizer.
        
        Parameters:
            style (str): Matplotlib style to use
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        
        # Set default parameters
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
    
    @staticmethod
    def plot_accuracy_comparison(results_dict, save_path=None):
        """
        Plot accuracy comparison across different methods.
        
        Parameters:
            results_dict (dict): Dictionary mapping method names to accuracy values
            save_path (str, optional): Path to save figure
        """
        methods = list(results_dict.keys())
        accuracies = list(results_dict.values())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
        bars = ax.bar(methods, accuracies, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{acc:.4f}\n({acc*100:.2f}%)',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Classification Accuracy Comparison', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1.0])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Accuracy comparison plot saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_time_comparison(results_dict, save_path=None):
        """
        Plot computation time comparison.
        
        Parameters:
            results_dict (dict): Dictionary mapping method names to time values
            save_path (str, optional): Path to save figure
        """
        methods = list(results_dict.keys())
        times = list(results_dict.values())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = plt.cm.Pastel1(np.linspace(0, 1, len(methods)))
        bars = ax.barh(methods, times, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar, t in zip(bars, times):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f'{t:.2f}s',
                   ha='left', va='center', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Computation Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_title('Computation Time Comparison', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Time comparison plot saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_pso_vs_grid_comparison(pso_results, grid_results, save_path=None):
        """
        Create comprehensive comparison between PSO and Grid Search.
        
        Parameters:
            pso_results (dict): PSO results
            grid_results (dict): Grid search results
            save_path (str, optional): Path to save figure
        """
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Accuracy comparison
        ax1 = fig.add_subplot(gs[0, 0])
        methods = ['PSO', 'Grid Search']
        accuracies = [pso_results['test_accuracy'], grid_results['test_accuracy']]
        colors = ['#FF6B6B', '#4ECDC4']
        
        bars = ax1.bar(methods, accuracies, color=colors, alpha=0.8, edgecolor='black')
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.4f}',
                    ha='center', va='bottom', fontweight='bold')
        
        ax1.set_ylabel('Test Accuracy', fontweight='bold')
        ax1.set_title('Accuracy Comparison', fontweight='bold')
        ax1.set_ylim([min(accuracies) - 0.01, max(accuracies) + 0.01])
        ax1.grid(axis='y', alpha=0.3)
        
        # Time comparison
        ax2 = fig.add_subplot(gs[0, 1])
        times = [pso_results['optimization_time'], grid_results['optimization_time']]
        
        bars = ax2.bar(methods, times, color=colors, alpha=0.8, edgecolor='black')
        for bar, t in zip(bars, times):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{t:.2f}s',
                    ha='center', va='bottom', fontweight='bold')
        
        ax2.set_ylabel('Optimization Time (seconds)', fontweight='bold')
        ax2.set_title('Optimization Time Comparison', fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # Parameter comparison
        ax3 = fig.add_subplot(gs[1, :])
        
        param_names = ['C', 'gamma']
        pso_params = [pso_results['best_params']['C'], pso_results['best_params']['gamma']]
        grid_params = [grid_results['best_params']['C'], grid_results['best_params']['gamma']]
        
        x = np.arange(len(param_names))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, pso_params, width, label='PSO', 
                       color=colors[0], alpha=0.8, edgecolor='black')
        bars2 = ax3.bar(x + width/2, grid_params, width, label='Grid Search',
                       color=colors[1], alpha=0.8, edgecolor='black')
        
        ax3.set_ylabel('Parameter Value', fontweight='bold')
        ax3.set_title('Best Hyperparameters Comparison', fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(param_names)
        ax3.legend()
        ax3.set_yscale('log')
        ax3.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}',
                        ha='center', va='bottom', fontsize=8)
        
        plt.suptitle('PSO vs Grid Search Optimization Comparison', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"PSO vs Grid comparison plot saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_pca_vs_kpca_comparison(pca_results, kpca_results, save_path=None):
        """
        Create comparison between PCA and Kernel PCA methods.
        
        Parameters:
            pca_results (dict): PCA results
            kpca_results (dict): KPCA results
            save_path (str, optional): Path to save figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        methods = ['PCA', 'Kernel PCA']
        colors = ['#95E1D3', '#F38181']
        
        # Accuracy
        accuracies = [pca_results['test_accuracy'], kpca_results['test_accuracy']]
        bars = axes[0].bar(methods, accuracies, color=colors, alpha=0.8, edgecolor='black')
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{acc:.4f}',
                        ha='center', va='bottom', fontweight='bold')
        axes[0].set_ylabel('Test Accuracy', fontweight='bold')
        axes[0].set_title('Accuracy Comparison', fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Reduction time
        red_times = [pca_results['reduction_time'], kpca_results['reduction_time']]
        bars = axes[1].bar(methods, red_times, color=colors, alpha=0.8, edgecolor='black')
        for bar, t in zip(bars, red_times):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{t:.2f}s',
                        ha='center', va='bottom', fontweight='bold')
        axes[1].set_ylabel('Reduction Time (seconds)', fontweight='bold')
        axes[1].set_title('Dimensionality Reduction Time', fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        
        # Total time
        total_times = [pca_results['total_time'], kpca_results['total_time']]
        bars = axes[2].bar(methods, total_times, color=colors, alpha=0.8, edgecolor='black')
        for bar, t in zip(bars, total_times):
            height = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width()/2., height,
                        f'{t:.2f}s',
                        ha='center', va='bottom', fontweight='bold')
        axes[2].set_ylabel('Total Time (seconds)', fontweight='bold')
        axes[2].set_title('Total Pipeline Time', fontweight='bold')
        axes[2].grid(axis='y', alpha=0.3)
        
        plt.suptitle('PCA vs Kernel PCA Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"PCA vs KPCA comparison plot saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def create_results_summary_table(results_list, save_path=None):
        """
        Create a summary table of all results.
        
        Parameters:
            results_list (list): List of result dictionaries
            save_path (str, optional): Path to save figure
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare table data
        headers = ['Method', 'Test Accuracy', 'Train Time (s)', 'Total Time (s)', 'Best C', 'Best Gamma']
        rows = []
        
        for result in results_list:
            row = [
                result.get('name', 'Unknown'),
                f"{result['test_accuracy']:.4f} ({result['test_accuracy']*100:.2f}%)",
                f"{result['training_time']:.2f}",
                f"{result['total_time']:.2f}",
                f"{result['best_params']['C']:.4f}",
                f"{result['best_params'].get('gamma', 'N/A')}"
            ]
            rows.append(row)
        
        # Create table
        table = ax.table(cellText=rows, colLabels=headers, cellLoc='center',
                        loc='center', bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header
        for i in range(len(headers)):
            cell = table[(0, i)]
            cell.set_facecolor('#4ECDC4')
            cell.set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(rows) + 1):
            for j in range(len(headers)):
                cell = table[(i, j)]
                if i % 2 == 0:
                    cell.set_facecolor('#F0F0F0')
                else:
                    cell.set_facecolor('white')
        
        plt.title('Experimental Results Summary', fontsize=16, fontweight='bold', pad=20)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Results summary table saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def load_and_visualize_results(results_dir):
        """
        Load results from JSON files and create visualizations.
        
        Parameters:
            results_dir (str): Directory containing result JSON files
        """
        import os
        
        results = {}
        
        # Load all JSON files
        for filename in os.listdir(results_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(results_dir, filename)
                with open(filepath, 'r') as f:
                    method_name = filename.replace('_results.json', '').replace('_', ' ').title()
                    results[method_name] = json.load(f)
        
        if not results:
            print(f"No result files found in {results_dir}")
            return
        
        # Create visualizations
        print(f"\nLoaded {len(results)} result files")
        print("Creating visualizations...")
        
        # Accuracy comparison
        accuracies = {name: res['test_accuracy'] for name, res in results.items()}
        ResultsVisualizer.plot_accuracy_comparison(accuracies)
        
        # Time comparison
        times = {name: res['total_time'] for name, res in results.items()}
        ResultsVisualizer.plot_time_comparison(times)
        
        print("Visualizations created successfully!")


if __name__ == "__main__":
    # Example usage
    print("=" * 70)
    print("Visualization Module - Test")
    print("=" * 70)
    
    # Create sample data for testing
    sample_results = {
        'PCA + PSO': 0.9756,
        'PCA + Grid': 0.9742,
        'KPCA + PSO': 0.9680
    }
    
    print("\nCreating sample accuracy comparison plot...")
    ResultsVisualizer.plot_accuracy_comparison(sample_results)
    
    sample_times = {
        'PCA + PSO': 245.3,
        'PCA + Grid': 312.7,
        'KPCA + PSO': 428.9
    }
    
    print("\nCreating sample time comparison plot...")
    ResultsVisualizer.plot_time_comparison(sample_times)
    
    print("\n" + "=" * 70)
    print("Visualization module test completed!")
    print("=" * 70)
