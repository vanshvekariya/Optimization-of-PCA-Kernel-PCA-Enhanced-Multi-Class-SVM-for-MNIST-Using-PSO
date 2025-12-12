"""
Data Loader Module for MNIST Dataset
=====================================
This module handles loading and preprocessing of the MNIST handwritten digit dataset.
It provides utilities for data normalization, train-test splitting, and data exploration.

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import os


class MNISTDataLoader:
    """
    A class to handle MNIST dataset loading and preprocessing.
    
    Attributes:
        X_train (np.ndarray): Training features (normalized)
        X_test (np.ndarray): Test features (normalized)
        y_train (np.ndarray): Training labels
        y_test (np.ndarray): Test labels
        n_samples (int): Total number of samples
        n_features (int): Number of features per sample (784 for 28x28 images)
        n_classes (int): Number of classes (10 for digits 0-9)
    """
    
    def __init__(self, data_path=None, test_size=10000, random_state=42):
        """
        Initialize the MNIST data loader.
        
        Parameters:
            data_path (str, optional): Path to local MNIST data files
            test_size (int): Number of samples for test set (default: 10000)
            random_state (int): Random seed for reproducibility (default: 42)
        """
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.n_samples = 0
        self.n_features = 784  # 28x28 pixels
        self.n_classes = 10    # Digits 0-9
        
    def load_data(self, normalize=True):
        """
        Load MNIST dataset from sklearn or local files.
        
        Parameters:
            normalize (bool): Whether to normalize pixel values to [0, 1] range
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        print("Loading MNIST dataset...")
        
        if self.data_path and os.path.exists(self.data_path):
            # Load from local files if available
            X, y = self._load_from_local()
        else:
            # Load from sklearn/openml
            X, y = self._load_from_sklearn()
        
        # Convert labels to integers
        y = y.astype(np.int32)
        
        # Normalize pixel values to [0, 1] range
        if normalize:
            X = X.astype(np.float64) / 255.0
            print("Data normalized to [0, 1] range")
        
        # Split into train and test sets
        # MNIST standard: 60,000 train, 10,000 test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=y  # Maintain class distribution
        )
        
        self.n_samples = len(X)
        
        print(f"Dataset loaded successfully!")
        print(f"Training samples: {len(self.X_train)}")
        print(f"Test samples: {len(self.X_test)}")
        print(f"Features per sample: {self.n_features}")
        print(f"Number of classes: {self.n_classes}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def _load_from_sklearn(self):
        """
        Load MNIST from sklearn's fetch_openml.
        
        Returns:
            tuple: (X, y) features and labels
        """
        print("Fetching MNIST from OpenML (this may take a moment)...")
        mnist = fetch_openml('mnist_784', version=1, parser='auto')
        X = mnist.data.to_numpy() if hasattr(mnist.data, 'to_numpy') else mnist.data
        y = mnist.target.to_numpy() if hasattr(mnist.target, 'to_numpy') else mnist.target
        return X, y
    
    def _load_from_local(self):
        """
        Load MNIST from local numpy files.
        
        Returns:
            tuple: (X, y) features and labels
        """
        print(f"Loading MNIST from local path: {self.data_path}")
        X = np.load(os.path.join(self.data_path, 'mnist_features.npy'))
        y = np.load(os.path.join(self.data_path, 'mnist_labels.npy'))
        return X, y
    
    def get_subset(self, n_samples, dataset='train', random_state=None):
        """
        Get a random subset of the data (useful for Kernel PCA experiments).
        
        Parameters:
            n_samples (int): Number of samples to extract
            dataset (str): 'train' or 'test'
            random_state (int, optional): Random seed
            
        Returns:
            tuple: (X_subset, y_subset)
        """
        if dataset == 'train':
            X, y = self.X_train, self.y_train
        else:
            X, y = self.X_test, self.y_test
        
        if n_samples >= len(X):
            return X, y
        
        np.random.seed(random_state)
        indices = np.random.choice(len(X), n_samples, replace=False)
        
        return X[indices], y[indices]
    
    def visualize_samples(self, n_samples=10, dataset='train', save_path=None):
        """
        Visualize random samples from the dataset.
        
        Parameters:
            n_samples (int): Number of samples to display
            dataset (str): 'train' or 'test'
            save_path (str, optional): Path to save the figure
        """
        if dataset == 'train':
            X, y = self.X_train, self.y_train
        else:
            X, y = self.X_test, self.y_test
        
        # Select random samples
        indices = np.random.choice(len(X), n_samples, replace=False)
        
        # Create subplot grid
        fig, axes = plt.subplots(2, 5, figsize=(12, 6))
        axes = axes.ravel()
        
        for i, idx in enumerate(indices):
            # Reshape to 28x28 for visualization
            img = X[idx].reshape(28, 28)
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'Label: {y[idx]}')
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def get_class_distribution(self, dataset='train'):
        """
        Get the distribution of classes in the dataset.
        
        Parameters:
            dataset (str): 'train' or 'test'
            
        Returns:
            dict: Class distribution
        """
        y = self.y_train if dataset == 'train' else self.y_test
        unique, counts = np.unique(y, return_counts=True)
        return dict(zip(unique, counts))
    
    def plot_class_distribution(self, save_path=None):
        """
        Plot the class distribution for both train and test sets.
        
        Parameters:
            save_path (str, optional): Path to save the figure
        """
        train_dist = self.get_class_distribution('train')
        test_dist = self.get_class_distribution('test')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Training set distribution
        ax1.bar(train_dist.keys(), train_dist.values(), color='steelblue', alpha=0.8)
        ax1.set_xlabel('Digit Class', fontsize=12)
        ax1.set_ylabel('Number of Samples', fontsize=12)
        ax1.set_title('Training Set Class Distribution', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Test set distribution
        ax2.bar(test_dist.keys(), test_dist.values(), color='coral', alpha=0.8)
        ax2.set_xlabel('Digit Class', fontsize=12)
        ax2.set_ylabel('Number of Samples', fontsize=12)
        ax2.set_title('Test Set Class Distribution', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Distribution plot saved to {save_path}")
        
        plt.show()


if __name__ == "__main__":
    # Example usage and testing
    print("=" * 60)
    print("MNIST Data Loader - Test Module")
    print("=" * 60)
    
    # Initialize loader
    loader = MNISTDataLoader(test_size=10000, random_state=42)
    
    # Load data
    X_train, X_test, y_train, y_test = loader.load_data(normalize=True)
    
    # Display statistics
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Feature range: [{X_train.min():.2f}, {X_train.max():.2f}]")
    
    # Show class distribution
    print("\nTraining set class distribution:")
    train_dist = loader.get_class_distribution('train')
    for digit, count in sorted(train_dist.items()):
        print(f"  Digit {digit}: {count} samples")
    
    # Visualize samples
    print("\nVisualizing sample digits...")
    loader.visualize_samples(n_samples=10, dataset='train')
    
    # Plot class distribution
    loader.plot_class_distribution()
    
    print("\n" + "=" * 60)
    print("Data loader test completed successfully!")
    print("=" * 60)
