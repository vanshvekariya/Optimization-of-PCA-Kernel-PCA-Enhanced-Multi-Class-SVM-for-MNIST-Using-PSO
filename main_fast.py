"""
FAST VERSION - Main Execution Script
=====================================
This is a faster version that uses smaller datasets and fewer iterations
to complete in reasonable time (10-15 minutes instead of hours).

Use this if main.py is taking too long!
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime

# Import project modules
from data_loader import MNISTDataLoader
from dimensionality_reduction import DimensionalityReducer, PCAAnalyzer
from multiclass_svm import MultiClassSVM, SVMGridSearch
from pso_optimizer import SVMPSOOptimizer, ParticleSwarmOptimizer
from pipeline import MNISTClassificationPipeline, PipelineComparison
from visualization import ResultsVisualizer


def create_output_directory():
    """Create output directory for results and figures."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results_fast_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "data"), exist_ok=True)
    
    print(f"\nOutput directory created: {output_dir}")
    return output_dir


def experiment_1_pca_analysis(loader, output_dir):
    """Experiment 1: PCA Analysis (FAST VERSION)"""
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: PCA Analysis (Fast Version)")
    print("=" * 80)
    
    # Use subset for faster analysis
    X_subset, y_subset = loader.get_subset(10000, 'train', random_state=42)
    
    print("\n[1.1] Analyzing PCA Explained Variance")
    print("-" * 80)
    
    reducer = DimensionalityReducer(method='pca', n_components=100)
    reducer.fit(X_subset)
    
    reducer.plot_explained_variance(
        save_path=os.path.join(output_dir, "figures", "pca_explained_variance.png")
    )
    
    print("\n[1.2] 2D PCA Visualization")
    print("-" * 80)
    
    reducer_2d = DimensionalityReducer(method='pca', n_components=2)
    reducer_2d.plot_2d_projection(
        X_subset[:2000], 
        y_subset[:2000],
        save_path=os.path.join(output_dir, "figures", "pca_2d_projection.png")
    )
    
    print("\nExperiment 1 completed!")


def experiment_2_pca_svm_pso(loader, output_dir):
    """Experiment 2: PCA + SVM + PSO (FAST VERSION)"""
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: PCA + SVM with PSO Optimization (Fast Version)")
    print("=" * 80)
    print("Using 15,000 training samples for speed")
    
    # Use smaller subset
    X_train_subset, y_train_subset = loader.get_subset(15000, 'train', random_state=42)
    
    # Create pipeline with REDUCED parameters
    pipeline = MNISTClassificationPipeline(
        reduction_method='pca',
        optimization_method='pso',
        n_components=50,  # Reduced from 100
        kernel='rbf'
    )
    
    # Run pipeline with FEWER particles and iterations
    results = pipeline.run(
        X_train_subset, y_train_subset,
        loader.X_test, loader.y_test,
        pso_particles=10,      # Reduced from 30
        pso_iterations=20,     # Reduced from 50
        verbose=True
    )
    
    # Confusion matrix
    print("\n[2.1] Confusion Matrix")
    print("-" * 80)
    pipeline.plot_confusion_matrix(
        loader.X_test, loader.y_test,
        save_path=os.path.join(output_dir, "figures", "pca_svm_pso_confusion_matrix.png")
    )
    
    # Save results
    pipeline.save_results(
        os.path.join(output_dir, "data", "pca_svm_pso_results.json")
    )
    
    print("\nExperiment 2 completed!")
    return results


def experiment_3_pca_svm_grid(loader, output_dir):
    """Experiment 3: PCA + SVM + Grid Search (FAST VERSION)"""
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: PCA + SVM with Grid Search (Fast Version)")
    print("=" * 80)
    print("Using 10,000 training samples for speed")
    
    # Use smaller subset
    X_train_subset, y_train_subset = loader.get_subset(10000, 'train', random_state=42)
    
    # SMALLER grid search
    grid_params = {
        'C': [1, 10],              # Only 2 values instead of 4
        'gamma': [0.001, 0.01]     # Only 2 values instead of 4
    }
    
    pipeline = MNISTClassificationPipeline(
        reduction_method='pca',
        optimization_method='grid',
        n_components=50,
        kernel='rbf'
    )
    
    results = pipeline.run(
        X_train_subset, y_train_subset,
        loader.X_test, loader.y_test,
        grid_params=grid_params,
        verbose=True
    )
    
    pipeline.plot_confusion_matrix(
        loader.X_test, loader.y_test,
        save_path=os.path.join(output_dir, "figures", "pca_svm_grid_confusion_matrix.png")
    )
    
    pipeline.save_results(
        os.path.join(output_dir, "data", "pca_svm_grid_results.json")
    )
    
    print("\nExperiment 3 completed!")
    return results


def experiment_4_kpca_svm_pso(loader, output_dir):
    """Experiment 4: Kernel PCA + SVM + PSO (FAST VERSION)"""
    print("\n" + "=" * 80)
    print("EXPERIMENT 4: Kernel PCA + SVM with PSO (Fast Version)")
    print("=" * 80)
    print("Using 5,000 training samples for speed")
    
    # Use small subset for KPCA
    X_train_subset, y_train_subset = loader.get_subset(5000, 'train', random_state=42)
    
    # 2D KPCA visualization
    print("\n[4.1] 2D Kernel PCA Visualization")
    print("-" * 80)
    
    kpca_2d = DimensionalityReducer(method='kpca', n_components=2, 
                                    kernel='rbf', gamma=0.001)
    kpca_2d.plot_2d_projection(
        X_train_subset[:1000], 
        y_train_subset[:1000],
        save_path=os.path.join(output_dir, "figures", "kpca_2d_projection.png")
    )
    
    # Create pipeline with KPCA
    print("\n[4.2] Training KPCA + SVM Pipeline")
    print("-" * 80)
    
    pipeline = MNISTClassificationPipeline(
        reduction_method='kpca',
        optimization_method='pso',
        n_components=30,  # Reduced from 50
        kernel='rbf'
    )
    
    results = pipeline.run(
        X_train_subset, y_train_subset,
        loader.X_test, loader.y_test,
        pso_particles=8,       # Reduced from 20
        pso_iterations=15,     # Reduced from 30
        verbose=True
    )
    
    pipeline.plot_confusion_matrix(
        loader.X_test, loader.y_test,
        save_path=os.path.join(output_dir, "figures", "kpca_svm_pso_confusion_matrix.png")
    )
    
    pipeline.save_results(
        os.path.join(output_dir, "data", "kpca_svm_pso_results.json")
    )
    
    print("\nExperiment 4 completed!")
    return results


def experiment_5_comparison(loader, output_dir):
    """Experiment 5: Quick Comparison (FAST VERSION)"""
    print("\n" + "=" * 80)
    print("EXPERIMENT 5: Method Comparison (Fast Version)")
    print("=" * 80)
    
    # Use subset
    X_train_subset, y_train_subset = loader.get_subset(10000, 'train', random_state=42)
    
    comparison = PipelineComparison()
    
    comparison.add_pipeline(
        name="PCA + PSO",
        reduction_method='pca',
        optimization_method='pso',
        n_components=50,
        kernel='rbf'
    )
    
    comparison.add_pipeline(
        name="PCA + Grid",
        reduction_method='pca',
        optimization_method='grid',
        n_components=50,
        kernel='rbf'
    )
    
    print("\nRunning comparison...")
    
    results = []
    for i, config in enumerate(comparison.pipelines, 1):
        print(f"\n[{i}/{len(comparison.pipelines)}] Running: {config['name']}")
        
        pipeline = MNISTClassificationPipeline(
            reduction_method=config['reduction_method'],
            optimization_method=config['optimization_method'],
            n_components=config['n_components'],
            kernel=config['kernel']
        )
        
        result = pipeline.run(
            X_train_subset, y_train_subset,
            loader.X_test, loader.y_test,
            pso_particles=10,
            pso_iterations=15,
            grid_params={'C': [1, 10], 'gamma': [0.001, 0.01]},
            verbose=True
        )
        
        result['name'] = config['name']
        result['config'] = config
        results.append(result)
    
    comparison.results = results
    comparison.print_summary()
    comparison.plot_comparison(
        save_path=os.path.join(output_dir, "figures", "method_comparison.png")
    )
    
    print("\nExperiment 5 completed!")
    return results


def main():
    """Main execution function - FAST VERSION"""
    print("=" * 80)
    print("MNIST CLASSIFICATION PROJECT - FAST VERSION")
    print("=" * 80)
    print("This version uses smaller datasets and fewer iterations")
    print("Expected completion time: 10-15 minutes")
    print("=" * 80)
    print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create output directory
    output_dir = create_output_directory()
    
    # Load MNIST data
    print("\n" + "=" * 80)
    print("LOADING MNIST DATASET")
    print("=" * 80)
    
    loader = MNISTDataLoader(test_size=10000, random_state=42)
    X_train, X_test, y_train, y_test = loader.load_data(normalize=True)
    
    # Visualize samples
    print("\nVisualizing sample digits...")
    loader.visualize_samples(
        n_samples=10, 
        dataset='train',
        save_path=os.path.join(output_dir, "figures", "sample_digits.png")
    )
    
    loader.plot_class_distribution(
        save_path=os.path.join(output_dir, "figures", "class_distribution.png")
    )
    
    # Run experiments
    try:
        # Experiment 1: PCA Analysis
        experiment_1_pca_analysis(loader, output_dir)
        
        # Experiment 2: PCA + SVM + PSO
        results_pca_pso = experiment_2_pca_svm_pso(loader, output_dir)
        
        # Experiment 3: PCA + SVM + Grid Search
        results_pca_grid = experiment_3_pca_svm_grid(loader, output_dir)
        
        # Experiment 4: Kernel PCA + SVM + PSO
        results_kpca_pso = experiment_4_kpca_svm_pso(loader, output_dir)
        
        # Experiment 5: Comparison
        comparison_results = experiment_5_comparison(loader, output_dir)
        
        # Generate final summary
        print("\n" + "=" * 80)
        print("FINAL SUMMARY")
        print("=" * 80)
        
        print("\nAll experiments completed successfully!")
        print(f"\nResults saved to: {output_dir}")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print("\n" + "=" * 80)
        print("PROJECT EXECUTION COMPLETED")
        print("=" * 80)
        print("\nNOTE: This was the FAST version with reduced parameters.")
        print("Accuracy may be slightly lower than full version, but still good!")
        print("\nNext step: Run generate_report.py to create your report data")
        
    except Exception as e:
        print(f"\n\nERROR: An exception occurred during execution:")
        print(f"{type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        print(f"\nPartial results may be available in: {output_dir}")


if __name__ == "__main__":
    main()
