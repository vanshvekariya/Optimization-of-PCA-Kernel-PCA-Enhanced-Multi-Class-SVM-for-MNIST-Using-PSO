"""
Multi-class SVM Classifier Module
==================================
This module implements multi-class Support Vector Machine (SVM) classification
using the One-vs-Rest (OvR) strategy with hinge loss formulation.

The implementation supports:
- Linear and RBF kernels
- Hyperparameter tuning (C, gamma)
- Performance evaluation metrics
- Confusion matrix visualization

Author: ML Project
Date: December 2025
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
import time


class MultiClassSVM:
    """
    Multi-class SVM classifier using One-vs-Rest strategy.
    
    Attributes:
        C (float): Regularization parameter (penalty parameter)
        kernel (str): Kernel type ('linear' or 'rbf')
        gamma (float): Kernel coefficient for 'rbf'
        classifier (OneVsRestClassifier): The fitted SVM classifier
        train_time (float): Time taken to train the model
        predict_time (float): Time taken to make predictions
    """
    
    def __init__(self, C=1.0, kernel='rbf', gamma='scale', class_weight='balanced'):
        """
        Initialize the Multi-class SVM classifier.
        
        Parameters:
            C (float): Regularization parameter (default: 1.0)
                      - Smaller C: more regularization, simpler decision boundary
                      - Larger C: less regularization, more complex boundary
            kernel (str): Kernel type - 'linear' or 'rbf' (default: 'rbf')
            gamma (float or str): Kernel coefficient for 'rbf' (default: 'scale')
                                 - 'scale': 1 / (n_features * X.var())
                                 - 'auto': 1 / n_features
                                 - float: custom value
            class_weight (str or dict): Weights for classes (default: 'balanced')
        """
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.class_weight = class_weight
        self.classifier = None
        self.train_time = 0
        self.predict_time = 0
        self.n_classes = None
        
    def fit(self, X_train, y_train, verbose=True):
        """
        Train the multi-class SVM classifier.
        
        Parameters:
            X_train (np.ndarray): Training features of shape (n_samples, n_features)
            y_train (np.ndarray): Training labels of shape (n_samples,)
            verbose (bool): Whether to print training information
            
        Returns:
            self: The fitted classifier
        """
        if verbose:
            print(f"\nTraining Multi-class SVM...")
            print(f"  Kernel: {self.kernel}")
            print(f"  C (penalty): {self.C}")
            if self.kernel == 'rbf':
                print(f"  Gamma: {self.gamma}")
            print(f"  Training samples: {len(X_train)}")
        
        self.n_classes = len(np.unique(y_train))
        
        start_time = time.time()
        
        # Create base SVM classifier
        base_svm = SVC(
            C=self.C,
            kernel=self.kernel,
            gamma=self.gamma,
            class_weight=self.class_weight,
            random_state=42,
            cache_size=1000,  # Increase cache for faster training
            max_iter=-1  # No iteration limit
        )
        
        # Wrap in One-vs-Rest classifier for multi-class
        self.classifier = OneVsRestClassifier(base_svm, n_jobs=-1)
        
        # Train the classifier
        self.classifier.fit(X_train, y_train)
        
        self.train_time = time.time() - start_time
        
        if verbose:
            print(f"Training completed in {self.train_time:.2f} seconds")
        
        return self
    
    def predict(self, X_test):
        """
        Predict class labels for test data.
        
        Parameters:
            X_test (np.ndarray): Test features of shape (n_samples, n_features)
            
        Returns:
            np.ndarray: Predicted labels of shape (n_samples,)
        """
        if self.classifier is None:
            raise ValueError("Classifier must be trained before prediction. Call fit() first.")
        
        start_time = time.time()
        y_pred = self.classifier.predict(X_test)
        self.predict_time = time.time() - start_time
        
        return y_pred
    
    def evaluate(self, X_test, y_test, verbose=True):
        """
        Evaluate the classifier on test data.
        
        Parameters:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): True test labels
            verbose (bool): Whether to print evaluation metrics
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'train_time': self.train_time,
            'predict_time': self.predict_time
        }
        
        if verbose:
            print("\n" + "=" * 60)
            print("Evaluation Metrics")
            print("=" * 60)
            print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-Score:  {f1:.4f}")
            print(f"\nTiming:")
            print(f"Training time:   {self.train_time:.2f} seconds")
            print(f"Prediction time: {self.predict_time:.4f} seconds")
            print("=" * 60)
        
        return metrics
    
    def get_classification_report(self, X_test, y_test):
        """
        Get detailed classification report.
        
        Parameters:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): True test labels
            
        Returns:
            str: Classification report
        """
        y_pred = self.predict(X_test)
        report = classification_report(y_test, y_pred, 
                                       target_names=[f'Digit {i}' for i in range(self.n_classes)])
        return report
    
    def plot_confusion_matrix(self, X_test, y_test, save_path=None, normalize=False):
        """
        Plot confusion matrix for the classifier.
        
        Parameters:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): True test labels
            save_path (str, optional): Path to save the figure
            normalize (bool): Whether to normalize the confusion matrix
        """
        y_pred = self.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                   xticklabels=range(self.n_classes),
                   yticklabels=range(self.n_classes),
                   cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
        
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()
        
        # Print per-class accuracy
        if not normalize:
            print("\nPer-class Accuracy:")
            class_accuracy = cm.diagonal() / cm.sum(axis=1)
            for i, acc in enumerate(class_accuracy):
                print(f"  Digit {i}: {acc:.4f} ({acc*100:.2f}%)")
    
    def get_support_vectors_info(self):
        """
        Get information about support vectors (for single SVM only).
        
        Returns:
            dict: Information about support vectors
        """
        if self.classifier is None:
            raise ValueError("Classifier must be trained first")
        
        # For One-vs-Rest, we have multiple binary classifiers
        n_support_total = 0
        estimators_info = []
        
        for i, estimator in enumerate(self.classifier.estimators_):
            n_support = len(estimator.support_vectors_)
            n_support_total += n_support
            estimators_info.append({
                'class': i,
                'n_support_vectors': n_support
            })
        
        return {
            'total_support_vectors': n_support_total,
            'per_class_info': estimators_info
        }


class SVMGridSearch:
    """
    Grid search for SVM hyperparameter tuning.
    """
    
    def __init__(self, param_grid):
        """
        Initialize grid search.
        
        Parameters:
            param_grid (dict): Dictionary with parameters names (str) as keys
                              and lists of parameter settings to try as values.
                              Example: {'C': [0.1, 1, 10], 'gamma': [0.001, 0.01]}
        """
        self.param_grid = param_grid
        self.results = []
        self.best_params = None
        self.best_score = 0
    
    def search(self, X_train, y_train, X_val, y_val, kernel='rbf', verbose=True):
        """
        Perform grid search over parameter grid.
        
        Parameters:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            X_val (np.ndarray): Validation features
            y_val (np.ndarray): Validation labels
            kernel (str): Kernel type
            verbose (bool): Whether to print progress
            
        Returns:
            dict: Best parameters found
        """
        from itertools import product
        
        # Generate all parameter combinations
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        param_combinations = list(product(*param_values))
        
        total_combinations = len(param_combinations)
        
        if verbose:
            print(f"\nStarting Grid Search...")
            print(f"Total combinations to test: {total_combinations}")
            print("=" * 60)
        
        for idx, param_combo in enumerate(param_combinations, 1):
            # Create parameter dictionary
            params = dict(zip(param_names, param_combo))
            
            if verbose:
                print(f"\n[{idx}/{total_combinations}] Testing: {params}")
            
            # Train SVM with current parameters
            svm = MultiClassSVM(
                C=params.get('C', 1.0),
                kernel=kernel,
                gamma=params.get('gamma', 'scale')
            )
            
            svm.fit(X_train, y_train, verbose=False)
            
            # Evaluate on validation set
            metrics = svm.evaluate(X_val, y_val, verbose=False)
            
            # Store results
            result = {
                'params': params,
                'accuracy': metrics['accuracy'],
                'train_time': metrics['train_time']
            }
            self.results.append(result)
            
            if verbose:
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
            
            # Update best parameters
            if metrics['accuracy'] > self.best_score:
                self.best_score = metrics['accuracy']
                self.best_params = params
        
        if verbose:
            print("\n" + "=" * 60)
            print("Grid Search Completed!")
            print(f"Best Parameters: {self.best_params}")
            print(f"Best Accuracy: {self.best_score:.4f}")
            print("=" * 60)
        
        return self.best_params
    
    def plot_results(self, save_path=None):
        """
        Plot grid search results.
        
        Parameters:
            save_path (str, optional): Path to save the figure
        """
        if not self.results:
            print("No results to plot. Run search() first.")
            return
        
        # Extract data for plotting
        accuracies = [r['accuracy'] for r in self.results]
        
        # If we have C and gamma parameters, create 2D heatmap
        if 'C' in self.param_grid and 'gamma' in self.param_grid:
            C_values = self.param_grid['C']
            gamma_values = self.param_grid['gamma']
            
            # Reshape accuracies into 2D grid
            acc_grid = np.array(accuracies).reshape(len(C_values), len(gamma_values))
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(acc_grid, annot=True, fmt='.4f', cmap='YlOrRd',
                       xticklabels=[f'{g:.4f}' for g in gamma_values],
                       yticklabels=[f'{c:.2f}' for c in C_values],
                       cbar_kws={'label': 'Accuracy'})
            
            plt.xlabel('Gamma', fontsize=12)
            plt.ylabel('C (Penalty Parameter)', fontsize=12)
            plt.title('Grid Search Results: Accuracy Heatmap', fontsize=14, fontweight='bold')
            
        else:
            # Simple bar plot
            plt.figure(figsize=(12, 6))
            x_labels = [str(r['params']) for r in self.results]
            plt.bar(range(len(accuracies)), accuracies, color='steelblue', alpha=0.8)
            plt.xlabel('Parameter Combination', fontsize=12)
            plt.ylabel('Accuracy', fontsize=12)
            plt.title('Grid Search Results', fontsize=14, fontweight='bold')
            plt.xticks(range(len(accuracies)), x_labels, rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Grid search results plot saved to {save_path}")
        
        plt.show()


if __name__ == "__main__":
    # Example usage and testing
    print("=" * 60)
    print("Multi-class SVM Module - Test")
    print("=" * 60)
    
    # Generate synthetic data for testing
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    
    digits = load_digits()
    X, y = digits.data, digits.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\nTest data:")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Test SVM with RBF kernel
    print("\n" + "=" * 60)
    print("Testing SVM with RBF kernel")
    print("=" * 60)
    
    svm = MultiClassSVM(C=1.0, kernel='rbf', gamma=0.001)
    svm.fit(X_train, y_train)
    metrics = svm.evaluate(X_test, y_test)
    
    # Print classification report
    print("\nDetailed Classification Report:")
    print(svm.get_classification_report(X_test, y_test))
    
    # Plot confusion matrix
    svm.plot_confusion_matrix(X_test, y_test)
    
    print("\n" + "=" * 60)
    print("Multi-class SVM test completed!")
    print("=" * 60)
