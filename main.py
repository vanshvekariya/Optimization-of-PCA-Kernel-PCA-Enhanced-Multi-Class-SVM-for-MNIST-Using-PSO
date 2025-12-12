"""
Main Execution Script for MNIST Classification Project
=======================================================
This is the main entry point for running all experiments and generating results
for the project: "Optimization of PCA/Kernel PCA-Enhanced Multi-Class SVM for 
MNIST Handwritten Digit Classification Using Particle Swarm Optimization"

Experiments:
1. PCA + SVM with PSO optimization
2. PCA + SVM with Grid Search optimization
3. Kernel PCA + SVM with PSO optimization (on subset)
4. Comparison of all methods
5. Detailed analysis and visualization

Author: ML Project
Date: December 2025
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
    """
    Create output directory for results and figures.
    
    Returns:
        str: Path to output directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results_{timestamp}"
    
    # Create subdirectories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "data"), exist_ok=True)
    
    print(f"\nOutput directory created: {output_dir}")
    return output_dir


def experiment_1_pca_analysis(loader, output_dir):
    """
    Experiment 1: PCA Analysis and Optimal Component Selection
    
    Parameters:
        loader (MNISTDataLoader): Data loader
        output_dir (str): Output directory
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: PCA Analysis and Optimal Component Selection")
    print("=" * 80)
    
    # Analyze explained variance
    print("\n[1.1] Analyzing PCA Explained Variance")
    print("-" * 80)
    
    reducer = DimensionalityReducer(method='pca', n_components=200)
    reducer.fit(loader.X_train)
    
    # Plot explained variance
    reducer.plot_explained_variance(
        save_path=os.path.join(output_dir, "figures", "pca_explained_variance.png")
    )
    
    # Find optimal components for different variance thresholds
    print("\n[1.2] Finding Optimal Components for Different Variance Thresholds")
    print("-" * 80)
    
    thresholds = [0.90, 0.95, 0.99]
    for threshold in thresholds:
        n_comp = PCAAnalyzer.find_optimal_components(
            loader.X_train, 
            variance_threshold=threshold,
            max_components=200
        )
    
    # 2D visualization
    print("\n[1.3] 2D PCA Visualization")
    print("-" * 80)
    
    reducer_2d = DimensionalityReducer(method='pca', n_components=2)
    reducer_2d.plot_2d_projection(
        loader.X_train[:5000], 
        loader.y_train[:5000],
        save_path=os.path.join(output_dir, "figures", "pca_2d_projection.png")
    )
    
    print("\nExperiment 1 completed!")


def experiment_2_pca_svm_pso(loader, output_dir):
    """
    Experiment 2: PCA + SVM with PSO Optimization
    
    Parameters:
        loader (MNISTDataLoader): Data loader
        output_dir (str): Output directory
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: PCA + SVM with PSO Optimization")
    print("=" * 80)
    
    # Create pipeline
    pipeline = MNISTClassificationPipeline(
        reduction_method='pca',
        optimization_method='pso',
        n_components=100,  # Based on PCA analysis
        kernel='rbf'
    )
    
    # Run pipeline
    results = pipeline.run(
        loader.X_train, loader.y_train,
        loader.X_test, loader.y_test,
        pso_particles=30,
        pso_iterations=50,
        verbose=True
    )
    
    # Detailed evaluation
    print("\n[2.1] Detailed Evaluation")
    print("-" * 80)
    pipeline.evaluate_detailed(loader.X_test, loader.y_test)
    
    # Confusion matrix
    print("\n[2.2] Confusion Matrix")
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
    """
    Experiment 3: PCA + SVM with Grid Search Optimization
    
    Parameters:
        loader (MNISTDataLoader): Data loader
        output_dir (str): Output directory
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: PCA + SVM with Grid Search Optimization")
    print("=" * 80)
    
    # Define grid search parameters
    grid_params = {
        'C': [0.1, 1, 10, 100],
        'gamma': [0.0001, 0.001, 0.01, 0.1]
    }
    
    # Create pipeline
    pipeline = MNISTClassificationPipeline(
        reduction_method='pca',
        optimization_method='grid',
        n_components=100,
        kernel='rbf'
    )
    
    # Run pipeline
    results = pipeline.run(
        loader.X_train, loader.y_train,
        loader.X_test, loader.y_test,
        grid_params=grid_params,
        verbose=True
    )
    
    # Confusion matrix
    pipeline.plot_confusion_matrix(
        loader.X_test, loader.y_test,
        save_path=os.path.join(output_dir, "figures", "pca_svm_grid_confusion_matrix.png")
    )
    
    # Save results
    pipeline.save_results(
        os.path.join(output_dir, "data", "pca_svm_grid_results.json")
    )
    
    print("\nExperiment 3 completed!")
    return results


def experiment_4_kpca_svm_pso(loader, output_dir):
    """
    Experiment 4: Kernel PCA + SVM with PSO Optimization (on subset)
    
    Parameters:
        loader (MNISTDataLoader): Data loader
        output_dir (str): Output directory
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 4: Kernel PCA + SVM with PSO Optimization")
    print("=" * 80)
    print("Note: Using subset of data due to computational complexity of Kernel PCA")
    
    # Use subset for Kernel PCA (computational efficiency)
    subset_size = 10000
    print(f"\nUsing {subset_size} training samples for Kernel PCA")
    
    X_train_subset, y_train_subset = loader.get_subset(
        subset_size, dataset='train', random_state=42
    )
    
    # 2D KPCA visualization
    print("\n[4.1] 2D Kernel PCA Visualization")
    print("-" * 80)
    
    kpca_2d = DimensionalityReducer(method='kpca', n_components=2, 
                                    kernel='rbf', gamma=0.001)
    kpca_2d.plot_2d_projection(
        X_train_subset[:2000], 
        y_train_subset[:2000],
        save_path=os.path.join(output_dir, "figures", "kpca_2d_projection.png")
    )
    
    # Create pipeline with KPCA
    print("\n[4.2] Training KPCA + SVM Pipeline")
    print("-" * 80)
    
    pipeline = MNISTClassificationPipeline(
        reduction_method='kpca',
        optimization_method='pso',
        n_components=50,  # Fewer components for KPCA
        kernel='rbf'
    )
    
    # Run pipeline on subset
    results = pipeline.run(
        X_train_subset, y_train_subset,
        loader.X_test, loader.y_test,
        pso_particles=20,  # Fewer particles due to computational cost
        pso_iterations=30,  # Fewer iterations
        verbose=True
    )
    
    # Confusion matrix
    pipeline.plot_confusion_matrix(
        loader.X_test, loader.y_test,
        save_path=os.path.join(output_dir, "figures", "kpca_svm_pso_confusion_matrix.png")
    )
    
    # Save results
    pipeline.save_results(
        os.path.join(output_dir, "data", "kpca_svm_pso_results.json")
    )
    
    print("\nExperiment 4 completed!")
    return results


def experiment_5_comparison(loader, output_dir):
    """
    Experiment 5: Comprehensive Comparison of All Methods
    
    Parameters:
        loader (MNISTDataLoader): Data loader
        output_dir (str): Output directory
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 5: Comprehensive Method Comparison")
    print("=" * 80)
    
    # Create comparison object
    comparison = PipelineComparison()
    
    # Add pipeline configurations
    comparison.add_pipeline(
        name="PCA + PSO",
        reduction_method='pca',
        optimization_method='pso',
        n_components=100,
        kernel='rbf'
    )
    
    comparison.add_pipeline(
        name="PCA + Grid",
        reduction_method='pca',
        optimization_method='grid',
        n_components=100,
        kernel='rbf'
    )
    
    # Note: KPCA comparison uses subset for computational efficiency
    print("\nNote: KPCA pipelines use 10,000 training samples for efficiency")
    
    X_train_subset, y_train_subset = loader.get_subset(
        10000, dataset='train', random_state=42
    )
    
    comparison.add_pipeline(
        name="KPCA + PSO",
        reduction_method='kpca',
        optimization_method='pso',
        n_components=50,
        kernel='rbf'
    )
    
    # Run comparison (using subset for KPCA)
    # For fair comparison, we'll run PCA methods on full data
    # and KPCA on subset separately
    
    print("\n[5.1] Running PCA-based pipelines on full training set")
    print("-" * 80)
    
    results_pca = []
    for i in range(2):  # First two are PCA-based
        config = comparison.pipelines[i]
        print(f"\nRunning: {config['name']}")
        
        pipeline = MNISTClassificationPipeline(
            reduction_method=config['reduction_method'],
            optimization_method=config['optimization_method'],
            n_components=config['n_components'],
            kernel=config['kernel']
        )
        
        result = pipeline.run(
            loader.X_train, loader.y_train,
            loader.X_test, loader.y_test,
            pso_particles=30,
            pso_iterations=50,
            grid_params={'C': [0.1, 1, 10, 100], 'gamma': [0.0001, 0.001, 0.01, 0.1]},
            verbose=True
        )
        
        result['name'] = config['name']
        result['config'] = config
        results_pca.append(result)
    
    print("\n[5.2] Running KPCA-based pipeline on subset")
    print("-" * 80)
    
    config = comparison.pipelines[2]
    print(f"\nRunning: {config['name']}")
    
    pipeline = MNISTClassificationPipeline(
        reduction_method=config['reduction_method'],
        optimization_method=config['optimization_method'],
        n_components=config['n_components'],
        kernel=config['kernel']
    )
    
    result = pipeline.run(
        X_train_subset, y_train_subset,
        loader.X_test, loader.y_test,
        pso_particles=20,
        pso_iterations=30,
        verbose=True
    )
    
    result['name'] = config['name']
    result['config'] = config
    
    # Combine results
    comparison.results = results_pca + [result]
    
    # Print summary
    comparison.print_summary()
    
    # Plot comparison
    comparison.plot_comparison(
        save_path=os.path.join(output_dir, "figures", "method_comparison.png")
    )
    
    print("\nExperiment 5 completed!")
    return comparison.results


def experiment_6_pso_convergence_analysis(loader, output_dir):
    """
    Experiment 6: PSO Convergence Analysis
    
    Parameters:
        loader (MNISTDataLoader): Data loader
        output_dir (str): Output directory
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 6: PSO Convergence Analysis")
    print("=" * 80)
    
    # Prepare data
    from sklearn.model_selection import train_test_split
    
    # Use PCA-reduced data
    reducer = DimensionalityReducer(method='pca', n_components=100)
    X_train_reduced = reducer.fit_transform(loader.X_train)
    X_test_reduced = reducer.transform(loader.X_test)
    
    # Split for validation
    X_train_opt, X_val, y_train_opt, y_val = train_test_split(
        X_train_reduced, loader.y_train,
        test_size=0.15,
        random_state=42,
        stratify=loader.y_train
    )
    
    # Run PSO optimization
    print("\n[6.1] Running PSO Optimization")
    print("-" * 80)
    
    pso_optimizer = SVMPSOOptimizer(
        X_train_opt, y_train_opt, X_val, y_val,
        optimize_pca=False,
        kernel='rbf',
        n_particles=30,
        n_iterations=50
    )
    
    best_params = pso_optimizer.optimize(verbose=True)
    
    # Plot convergence
    print("\n[6.2] Plotting PSO Convergence")
    print("-" * 80)
    
    pso_optimizer.plot_convergence(
        save_path=os.path.join(output_dir, "figures", "pso_convergence.png")
    )
    
    print("\nExperiment 6 completed!")
    return best_params


def main():
    """
    Main execution function - runs all experiments.
    """
    print("=" * 80)
    print("MNIST HANDWRITTEN DIGIT CLASSIFICATION PROJECT")
    print("PCA/Kernel PCA-Enhanced Multi-Class SVM with PSO Optimization")
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
    
    # Plot class distribution
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
        
        # Experiment 5: Comprehensive Comparison
        comparison_results = experiment_5_comparison(loader, output_dir)
        
        # Experiment 6: PSO Convergence Analysis
        pso_params = experiment_6_pso_convergence_analysis(loader, output_dir)
        
        # Generate final summary report
        print("\n" + "=" * 80)
        print("FINAL SUMMARY")
        print("=" * 80)
        
        print("\nAll experiments completed successfully!")
        print(f"\nResults saved to: {output_dir}")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print("\n" + "=" * 80)
        print("PROJECT EXECUTION COMPLETED")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n\nERROR: An exception occurred during execution:")
        print(f"{type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        print(f"\nPartial results may be available in: {output_dir}")


if __name__ == "__main__":
    main()
