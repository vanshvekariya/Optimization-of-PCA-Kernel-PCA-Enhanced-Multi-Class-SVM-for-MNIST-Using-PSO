"""
Complete Pipeline Module
=========================
This module provides end-to-end pipelines for MNIST classification combining:
- Dimensionality reduction (PCA or Kernel PCA)
- Multi-class SVM classification
- Hyperparameter optimization (PSO or Grid Search)

"""

import numpy as np
import time
import json
from sklearn.model_selection import train_test_split
from data_loader import MNISTDataLoader
from dimensionality_reduction import DimensionalityReducer
from multiclass_svm import MultiClassSVM, SVMGridSearch
from pso_optimizer import SVMPSOOptimizer
import matplotlib.pyplot as plt


class MNISTClassificationPipeline:
    """
    Complete pipeline for MNIST classification.
    
    Combines dimensionality reduction, SVM classification, and hyperparameter optimization.
    """
    
    def __init__(self, reduction_method='pca', optimization_method='pso', 
                 n_components=50, kernel='rbf'):
        """
        Initialize the classification pipeline.
        
        Parameters:
            reduction_method (str): 'pca' or 'kpca'
            optimization_method (str): 'pso' or 'grid'
            n_components (int): Number of PCA/KPCA components
            kernel (str): SVM kernel type ('linear' or 'rbf')
        """
        self.reduction_method = reduction_method
        self.optimization_method = optimization_method
        self.n_components = n_components
        self.kernel = kernel
        
        self.reducer = None
        self.classifier = None
        self.best_params = None
        
        self.results = {
            'reduction_time': 0,
            'optimization_time': 0,
            'training_time': 0,
            'prediction_time': 0,
            'total_time': 0,
            'train_accuracy': 0,
            'test_accuracy': 0,
            'best_params': None
        }
    
    def run(self, X_train, y_train, X_test, y_test, 
            pso_particles=30, pso_iterations=50,
            grid_params=None, verbose=True):
        """
        Run the complete pipeline.
        
        Parameters:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
            pso_particles (int): Number of PSO particles
            pso_iterations (int): Number of PSO iterations
            grid_params (dict): Grid search parameters
            verbose (bool): Whether to print progress
            
        Returns:
            dict: Results dictionary
        """
        total_start = time.time()
        
        if verbose:
            print("\n" + "=" * 70)
            print(f"MNIST Classification Pipeline")
            print("=" * 70)
            print(f"Dimensionality Reduction: {self.reduction_method.upper()}")
            print(f"Optimization Method: {self.optimization_method.upper()}")
            print(f"SVM Kernel: {self.kernel}")
            print(f"Components: {self.n_components}")
            print("=" * 70)
        
        # Step 1: Dimensionality Reduction
        if verbose:
            print("\n[Step 1/4] Dimensionality Reduction")
            print("-" * 70)
        
        reduction_start = time.time()
        
        self.reducer = DimensionalityReducer(
            method=self.reduction_method,
            n_components=self.n_components,
            kernel='rbf' if self.reduction_method == 'kpca' else None,
            gamma=0.001 if self.reduction_method == 'kpca' else None
        )
        
        X_train_reduced = self.reducer.fit_transform(X_train)
        X_test_reduced = self.reducer.transform(X_test)
        
        self.results['reduction_time'] = time.time() - reduction_start
        
        if verbose:
            print(f"Original shape: {X_train.shape}")
            print(f"Reduced shape: {X_train_reduced.shape}")
            print(f"Time: {self.results['reduction_time']:.2f} seconds")
        
        # Step 2: Split for validation (for hyperparameter tuning)
        X_train_opt, X_val, y_train_opt, y_val = train_test_split(
            X_train_reduced, y_train, 
            test_size=0.15, 
            random_state=42,
            stratify=y_train
        )
        
        # Step 3: Hyperparameter Optimization
        if verbose:
            print("\n[Step 2/4] Hyperparameter Optimization")
            print("-" * 70)
        
        opt_start = time.time()
        
        if self.optimization_method == 'pso':
            # PSO Optimization
            pso_optimizer = SVMPSOOptimizer(
                X_train_opt, y_train_opt, X_val, y_val,
                optimize_pca=False,
                kernel=self.kernel,
                n_particles=pso_particles,
                n_iterations=pso_iterations
            )
            
            self.best_params = pso_optimizer.optimize(verbose=verbose)
            
        else:  # grid search
            # Grid Search
            if grid_params is None:
                grid_params = {
                    'C': [0.1, 1, 10, 100],
                    'gamma': [0.001, 0.01, 0.1, 1] if self.kernel == 'rbf' else [0.001]
                }
            
            grid_search = SVMGridSearch(param_grid=grid_params)
            self.best_params = grid_search.search(
                X_train_opt, y_train_opt, X_val, y_val,
                kernel=self.kernel,
                verbose=verbose
            )
            self.best_params['accuracy'] = grid_search.best_score
        
        self.results['optimization_time'] = time.time() - opt_start
        self.results['best_params'] = self.best_params
        
        if verbose:
            print(f"Optimization time: {self.results['optimization_time']:.2f} seconds")
        
        # Step 4: Train final model with best parameters
        if verbose:
            print("\n[Step 3/4] Training Final Model")
            print("-" * 70)
        
        train_start = time.time()
        
        self.classifier = MultiClassSVM(
            C=self.best_params['C'],
            kernel=self.kernel,
            gamma=self.best_params.get('gamma', 'scale')
        )
        
        self.classifier.fit(X_train_reduced, y_train, verbose=verbose)
        
        self.results['training_time'] = time.time() - train_start
        
        # Step 5: Evaluation
        if verbose:
            print("\n[Step 4/4] Evaluation")
            print("-" * 70)
        
        # Training accuracy
        y_train_pred = self.classifier.predict(X_train_reduced)
        self.results['train_accuracy'] = np.mean(y_train_pred == y_train)
        
        # Test accuracy
        y_test_pred = self.classifier.predict(X_test_reduced)
        self.results['test_accuracy'] = np.mean(y_test_pred == y_test)
        
        self.results['prediction_time'] = self.classifier.predict_time
        self.results['total_time'] = time.time() - total_start
        
        if verbose:
            print(f"\nTraining Accuracy: {self.results['train_accuracy']:.4f} ({self.results['train_accuracy']*100:.2f}%)")
            print(f"Test Accuracy: {self.results['test_accuracy']:.4f} ({self.results['test_accuracy']*100:.2f}%)")
            print(f"\nTotal Pipeline Time: {self.results['total_time']:.2f} seconds")
            print("=" * 70)
        
        return self.results
    
    def evaluate_detailed(self, X_test, y_test):
        """
        Get detailed evaluation metrics.
        
        Parameters:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
            
        Returns:
            dict: Detailed metrics
        """
        X_test_reduced = self.reducer.transform(X_test)
        return self.classifier.evaluate(X_test_reduced, y_test, verbose=True)
    
    def plot_confusion_matrix(self, X_test, y_test, save_path=None):
        """
        Plot confusion matrix.
        
        Parameters:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
            save_path (str, optional): Path to save figure
        """
        X_test_reduced = self.reducer.transform(X_test)
        self.classifier.plot_confusion_matrix(X_test_reduced, y_test, save_path=save_path)
    
    def save_results(self, filepath):
        """
        Save results to JSON file.
        
        Parameters:
            filepath (str): Path to save results
        """
        # Convert numpy types to Python types for JSON serialization
        results_serializable = {}
        for key, value in self.results.items():
            if isinstance(value, (np.integer, np.floating)):
                results_serializable[key] = float(value)
            elif isinstance(value, dict):
                results_serializable[key] = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                                            for k, v in value.items()}
            else:
                results_serializable[key] = value
        
        with open(filepath, 'w') as f:
            json.dump(results_serializable, f, indent=4)
        
        print(f"Results saved to {filepath}")


class PipelineComparison:
    """
    Compare different pipeline configurations.
    """
    
    def __init__(self):
        """Initialize pipeline comparison."""
        self.pipelines = []
        self.results = []
    
    def add_pipeline(self, name, reduction_method, optimization_method, 
                    n_components, kernel='rbf'):
        """
        Add a pipeline configuration to compare.
        
        Parameters:
            name (str): Pipeline name
            reduction_method (str): 'pca' or 'kpca'
            optimization_method (str): 'pso' or 'grid'
            n_components (int): Number of components
            kernel (str): SVM kernel
        """
        self.pipelines.append({
            'name': name,
            'reduction_method': reduction_method,
            'optimization_method': optimization_method,
            'n_components': n_components,
            'kernel': kernel
        })
    
    def run_comparison(self, X_train, y_train, X_test, y_test, 
                      pso_particles=20, pso_iterations=30, verbose=True):
        """
        Run all pipeline configurations and compare.
        
        Parameters:
            X_train, y_train, X_test, y_test: Data
            pso_particles (int): PSO particles
            pso_iterations (int): PSO iterations
            verbose (bool): Print progress
            
        Returns:
            list: Results for all pipelines
        """
        self.results = []
        
        for i, config in enumerate(self.pipelines, 1):
            if verbose:
                print("\n" + "=" * 70)
                print(f"Running Pipeline {i}/{len(self.pipelines)}: {config['name']}")
                print("=" * 70)
            
            pipeline = MNISTClassificationPipeline(
                reduction_method=config['reduction_method'],
                optimization_method=config['optimization_method'],
                n_components=config['n_components'],
                kernel=config['kernel']
            )
            
            result = pipeline.run(
                X_train, y_train, X_test, y_test,
                pso_particles=pso_particles,
                pso_iterations=pso_iterations,
                verbose=verbose
            )
            
            result['name'] = config['name']
            result['config'] = config
            self.results.append(result)
        
        return self.results
    
    def plot_comparison(self, save_path=None):
        """
        Plot comparison of all pipelines.
        
        Parameters:
            save_path (str, optional): Path to save figure
        """
        if not self.results:
            print("No results to plot. Run comparison first.")
            return
        
        names = [r['name'] for r in self.results]
        test_acc = [r['test_accuracy'] for r in self.results]
        total_time = [r['total_time'] for r in self.results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy comparison
        colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
        bars1 = ax1.bar(range(len(names)), test_acc, color=colors, alpha=0.8)
        ax1.set_xlabel('Pipeline', fontsize=12)
        ax1.set_ylabel('Test Accuracy', fontsize=12)
        ax1.set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(names, rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, acc in zip(bars1, test_acc):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.4f}',
                    ha='center', va='bottom', fontsize=9)
        
        # Time comparison
        bars2 = ax2.bar(range(len(names)), total_time, color=colors, alpha=0.8)
        ax2.set_xlabel('Pipeline', fontsize=12)
        ax2.set_ylabel('Total Time (seconds)', fontsize=12)
        ax2.set_title('Computation Time Comparison', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels(names, rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, t in zip(bars2, total_time):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{t:.1f}s',
                    ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to {save_path}")
        
        plt.show()
    
    def print_summary(self):
        """Print summary of all pipeline results."""
        if not self.results:
            print("No results available.")
            return
        
        print("\n" + "=" * 70)
        print("PIPELINE COMPARISON SUMMARY")
        print("=" * 70)
        
        for i, result in enumerate(self.results, 1):
            print(f"\n{i}. {result['name']}")
            print("-" * 70)
            print(f"   Configuration:")
            print(f"     - Reduction: {result['config']['reduction_method'].upper()}")
            print(f"     - Optimization: {result['config']['optimization_method'].upper()}")
            print(f"     - Components: {result['config']['n_components']}")
            print(f"     - Kernel: {result['config']['kernel']}")
            print(f"   Results:")
            print(f"     - Test Accuracy: {result['test_accuracy']:.4f} ({result['test_accuracy']*100:.2f}%)")
            print(f"     - Total Time: {result['total_time']:.2f} seconds")
            print(f"   Best Parameters:")
            for param, value in result['best_params'].items():
                if param != 'accuracy':
                    print(f"     - {param}: {value}")
        
        # Find best pipeline
        best_idx = np.argmax([r['test_accuracy'] for r in self.results])
        best_result = self.results[best_idx]
        
        print("\n" + "=" * 70)
        print(f"BEST PIPELINE: {best_result['name']}")
        print(f"Test Accuracy: {best_result['test_accuracy']:.4f} ({best_result['test_accuracy']*100:.2f}%)")
        print("=" * 70)


if __name__ == "__main__":
    # Example usage
    print("=" * 70)
    print("Pipeline Module - Test")
    print("=" * 70)
    
    # This is a minimal test - full testing requires MNIST data
    print("\nPipeline module loaded successfully!")
    print("Use main.py to run complete experiments.")
