"""
Dimensionality Reduction Module
================================
This module implements PCA and Kernel PCA for feature extraction and dimensionality reduction.
Both methods are used to reduce the 784-dimensional MNIST features to a lower-dimensional space
while preserving the most important information.

"""

import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time


class DimensionalityReducer:
    """
    A class to handle dimensionality reduction using PCA and Kernel PCA.
    
    Attributes:
        method (str): 'pca' or 'kpca'
        n_components (int): Number of components to retain
        reducer (object): The fitted PCA or KernelPCA object
        scaler (StandardScaler): Data standardization scaler
    """
    
    def __init__(self, method='pca', n_components=50, kernel='rbf', gamma=None):
        """
        Initialize the dimensionality reducer.
        
        Parameters:
            method (str): 'pca' for linear PCA or 'kpca' for Kernel PCA
            n_components (int): Number of principal components to retain
            kernel (str): Kernel type for KPCA ('rbf', 'poly', 'sigmoid', 'cosine')
            gamma (float): Kernel coefficient for rbf, poly and sigmoid kernels
        """
        self.method = method.lower()
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.reducer = None
        self.scaler = StandardScaler()
        self.fit_time = 0
        self.transform_time = 0
        
        # Validate method
        if self.method not in ['pca', 'kpca']:
            raise ValueError("Method must be 'pca' or 'kpca'")
    
    def fit(self, X, y=None):
        """
        Fit the dimensionality reduction model.
        
        Parameters:
            X (np.ndarray): Training data of shape (n_samples, n_features)
            y (np.ndarray, optional): Labels (not used, for API consistency)
            
        Returns:
            self: The fitted reducer
        """
        print(f"\nFitting {self.method.upper()} with {self.n_components} components...")
        start_time = time.time()
        
        # Standardize the data (important for PCA/KPCA)
        X_scaled = self.scaler.fit_transform(X)
        
        if self.method == 'pca':
            # Linear PCA
            self.reducer = PCA(
                n_components=self.n_components,
                random_state=42,
                svd_solver='auto'
            )
            self.reducer.fit(X_scaled)
            
            # Print explained variance information
            explained_var = np.sum(self.reducer.explained_variance_ratio_)
            print(f"PCA fitted in {time.time() - start_time:.2f} seconds")
            print(f"Cumulative explained variance: {explained_var:.4f} ({explained_var*100:.2f}%)")
            
        else:  # kpca
            # Kernel PCA (nonlinear)
            self.reducer = KernelPCA(
                n_components=self.n_components,
                kernel=self.kernel,
                gamma=self.gamma,
                fit_inverse_transform=False,  # Set to False to save memory
                random_state=42,
                n_jobs=-1  # Use all available cores
            )
            self.reducer.fit(X_scaled)
            
            print(f"Kernel PCA ({self.kernel} kernel) fitted in {time.time() - start_time:.2f} seconds")
            if self.gamma:
                print(f"Gamma parameter: {self.gamma}")
        
        self.fit_time = time.time() - start_time
        return self
    
    def transform(self, X):
        """
        Transform data to reduced dimensionality space.
        
        Parameters:
            X (np.ndarray): Data to transform of shape (n_samples, n_features)
            
        Returns:
            np.ndarray: Transformed data of shape (n_samples, n_components)
        """
        if self.reducer is None:
            raise ValueError("Reducer must be fitted before transform. Call fit() first.")
        
        start_time = time.time()
        
        # Standardize using the fitted scaler
        X_scaled = self.scaler.transform(X)
        
        # Transform to lower dimensional space
        X_reduced = self.reducer.transform(X_scaled)
        
        self.transform_time = time.time() - start_time
        
        return X_reduced
    
    def fit_transform(self, X, y=None):
        """
        Fit the model and transform the data in one step.
        
        Parameters:
            X (np.ndarray): Training data
            y (np.ndarray, optional): Labels
            
        Returns:
            np.ndarray: Transformed data
        """
        self.fit(X, y)
        return self.transform(X)
    
    def get_explained_variance_ratio(self):
        """
        Get the explained variance ratio for each component (PCA only).
        
        Returns:
            np.ndarray: Explained variance ratio for each component
        """
        if self.method != 'pca':
            print("Warning: Explained variance ratio only available for linear PCA")
            return None
        
        if self.reducer is None:
            raise ValueError("Reducer must be fitted first")
        
        return self.reducer.explained_variance_ratio_
    
    def plot_explained_variance(self, save_path=None):
        """
        Plot cumulative explained variance (PCA only).
        
        Parameters:
            save_path (str, optional): Path to save the figure
        """
        if self.method != 'pca':
            print("Warning: Explained variance plot only available for linear PCA")
            return
        
        if self.reducer is None:
            raise ValueError("Reducer must be fitted first")
        
        explained_var_ratio = self.reducer.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var_ratio)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Individual explained variance
        ax1.bar(range(1, len(explained_var_ratio) + 1), explained_var_ratio, 
                alpha=0.7, color='steelblue')
        ax1.set_xlabel('Principal Component', fontsize=12)
        ax1.set_ylabel('Explained Variance Ratio', fontsize=12)
        ax1.set_title('Individual Explained Variance by Component', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Cumulative explained variance
        ax2.plot(range(1, len(cumulative_var) + 1), cumulative_var, 
                marker='o', linestyle='-', color='coral', linewidth=2, markersize=4)
        ax2.axhline(y=0.95, color='red', linestyle='--', label='95% threshold')
        ax2.set_xlabel('Number of Components', fontsize=12)
        ax2.set_ylabel('Cumulative Explained Variance', fontsize=12)
        ax2.set_title('Cumulative Explained Variance', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Explained variance plot saved to {save_path}")
        
        plt.show()
        
        # Print statistics
        print(f"\nExplained Variance Statistics:")
        print(f"Total variance explained: {cumulative_var[-1]:.4f} ({cumulative_var[-1]*100:.2f}%)")
        
        # Find number of components for 95% variance
        n_components_95 = np.argmax(cumulative_var >= 0.95) + 1
        print(f"Components needed for 95% variance: {n_components_95}")
    
    def plot_2d_projection(self, X, y, save_path=None):
        """
        Plot 2D projection of the data using the first 2 principal components.
        
        Parameters:
            X (np.ndarray): Original data
            y (np.ndarray): Labels
            save_path (str, optional): Path to save the figure
        """
        # Create a temporary reducer with 2 components for visualization
        temp_reducer = DimensionalityReducer(
            method=self.method,
            n_components=2,
            kernel=self.kernel,
            gamma=self.gamma
        )
        
        X_2d = temp_reducer.fit_transform(X)
        
        # Create scatter plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='tab10', 
                            alpha=0.6, s=10, edgecolors='none')
        plt.colorbar(scatter, label='Digit Class')
        plt.xlabel('First Principal Component', fontsize=12)
        plt.ylabel('Second Principal Component', fontsize=12)
        
        title = f'2D Projection using {self.method.upper()}'
        if self.method == 'kpca':
            title += f' ({self.kernel} kernel)'
        plt.title(title, fontsize=14, fontweight='bold')
        
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"2D projection plot saved to {save_path}")
        
        plt.show()


class PCAAnalyzer:
    """
    A utility class for analyzing optimal number of PCA components.
    """
    
    @staticmethod
    def find_optimal_components(X, variance_threshold=0.95, max_components=200):
        """
        Find the optimal number of PCA components to retain a given variance.
        
        Parameters:
            X (np.ndarray): Training data
            variance_threshold (float): Desired cumulative explained variance (0-1)
            max_components (int): Maximum components to consider
            
        Returns:
            int: Optimal number of components
        """
        print(f"\nFinding optimal components for {variance_threshold*100}% variance...")
        
        # Standardize data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit PCA with maximum components
        pca = PCA(n_components=max_components, random_state=42)
        pca.fit(X_scaled)
        
        # Find number of components
        cumulative_var = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumulative_var >= variance_threshold) + 1
        
        print(f"Optimal components: {n_components}")
        print(f"Actual variance explained: {cumulative_var[n_components-1]:.4f}")
        
        return n_components
    
    @staticmethod
    def compare_component_counts(X, y, component_counts, classifier_func, save_path=None):
        """
        Compare classification performance for different numbers of components.
        
        Parameters:
            X (np.ndarray): Training data
            y (np.ndarray): Labels
            component_counts (list): List of component counts to test
            classifier_func (callable): Function that takes (X_train, y_train) and returns accuracy
            save_path (str, optional): Path to save the figure
            
        Returns:
            dict: Results for each component count
        """
        results = {}
        
        for n_comp in component_counts:
            print(f"\nTesting with {n_comp} components...")
            
            reducer = DimensionalityReducer(method='pca', n_components=n_comp)
            X_reduced = reducer.fit_transform(X)
            
            accuracy = classifier_func(X_reduced, y)
            results[n_comp] = accuracy
            
            print(f"Accuracy: {accuracy:.4f}")
        
        # Plot results
        plt.figure(figsize=(10, 6))
        plt.plot(list(results.keys()), list(results.values()), 
                marker='o', linestyle='-', linewidth=2, markersize=8, color='steelblue')
        plt.xlabel('Number of Components', fontsize=12)
        plt.ylabel('Classification Accuracy', fontsize=12)
        plt.title('Classification Accuracy vs. Number of PCA Components', 
                 fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to {save_path}")
        
        plt.show()
        
        return results


if __name__ == "__main__":
    # Example usage and testing
    print("=" * 60)
    print("Dimensionality Reduction Module - Test")
    print("=" * 60)
    
    # Generate synthetic data for testing
    from sklearn.datasets import load_digits
    
    digits = load_digits()
    X, y = digits.data, digits.target
    
    print(f"\nTest data shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    # Test PCA
    print("\n" + "=" * 60)
    print("Testing Linear PCA")
    print("=" * 60)
    
    pca_reducer = DimensionalityReducer(method='pca', n_components=20)
    X_pca = pca_reducer.fit_transform(X)
    
    print(f"Reduced shape: {X_pca.shape}")
    pca_reducer.plot_explained_variance()
    pca_reducer.plot_2d_projection(X[:500], y[:500])
    
    # Test Kernel PCA
    print("\n" + "=" * 60)
    print("Testing Kernel PCA (RBF)")
    print("=" * 60)
    
    kpca_reducer = DimensionalityReducer(method='kpca', n_components=20, 
                                         kernel='rbf', gamma=0.001)
    X_kpca = kpca_reducer.fit_transform(X[:500])  # Use subset for speed
    
    print(f"Reduced shape: {X_kpca.shape}")
    kpca_reducer.plot_2d_projection(X[:500], y[:500])
    
    print("\n" + "=" * 60)
    print("Dimensionality reduction test completed!")
    print("=" * 60)
