"""
Installation and Setup Test Script
===================================
This script tests that all dependencies are properly installed
and that the project modules can be imported successfully.

Run this before executing main.py to ensure everything is set up correctly.

Author: ML Project
Date: December 2025
"""

import sys
import importlib


def test_python_version():
    """Test Python version."""
    print("Testing Python version...")
    version = sys.version_info
    print(f"  Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("  ❌ FAILED: Python 3.8 or higher required")
        return False
    else:
        print("  ✅ PASSED: Python version is compatible")
        return True


def test_package(package_name, import_name=None):
    """Test if a package can be imported."""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"  ✅ {package_name:20s} version {version}")
        return True
    except ImportError as e:
        print(f"  ❌ {package_name:20s} NOT FOUND")
        print(f"     Error: {e}")
        return False


def test_dependencies():
    """Test all required dependencies."""
    print("\nTesting required packages...")
    
    packages = [
        ('numpy', 'numpy'),
        ('scikit-learn', 'sklearn'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('pandas', 'pandas'),
        ('scipy', 'scipy'),
        ('joblib', 'joblib'),
        ('tqdm', 'tqdm'),
    ]
    
    results = []
    for package_name, import_name in packages:
        results.append(test_package(package_name, import_name))
    
    return all(results)


def test_project_modules():
    """Test if project modules can be imported."""
    print("\nTesting project modules...")
    
    modules = [
        'data_loader',
        'dimensionality_reduction',
        'multiclass_svm',
        'pso_optimizer',
        'pipeline',
        'visualization'
    ]
    
    results = []
    for module_name in modules:
        try:
            importlib.import_module(module_name)
            print(f"  ✅ {module_name}")
            results.append(True)
        except ImportError as e:
            print(f"  ❌ {module_name}")
            print(f"     Error: {e}")
            results.append(False)
    
    return all(results)


def test_data_loading():
    """Test basic data loading functionality."""
    print("\nTesting data loading...")
    
    try:
        from data_loader import MNISTDataLoader
        
        print("  Creating data loader...")
        loader = MNISTDataLoader(test_size=100, random_state=42)
        
        print("  Loading MNIST dataset (this may take a moment)...")
        X_train, X_test, y_train, y_test = loader.load_data(normalize=True)
        
        print(f"  ✅ Data loaded successfully!")
        print(f"     Training samples: {len(X_train)}")
        print(f"     Test samples: {len(X_test)}")
        print(f"     Features: {X_train.shape[1]}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Data loading failed")
        print(f"     Error: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality of key modules."""
    print("\nTesting basic functionality...")
    
    try:
        import numpy as np
        from sklearn.datasets import load_digits
        from dimensionality_reduction import DimensionalityReducer
        from multiclass_svm import MultiClassSVM
        
        # Load small test dataset
        print("  Loading test dataset...")
        digits = load_digits()
        X, y = digits.data[:100], digits.target[:100]
        
        # Test PCA
        print("  Testing PCA...")
        pca = DimensionalityReducer(method='pca', n_components=10)
        X_pca = pca.fit_transform(X)
        print(f"     Original shape: {X.shape}")
        print(f"     Reduced shape: {X_pca.shape}")
        
        # Test SVM
        print("  Testing SVM...")
        svm = MultiClassSVM(C=1.0, kernel='rbf', gamma=0.001)
        svm.fit(X_pca[:80], y[:80], verbose=False)
        y_pred = svm.predict(X_pca[80:])
        accuracy = np.mean(y_pred == y[80:])
        print(f"     Test accuracy: {accuracy:.4f}")
        
        print("  ✅ Basic functionality test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Functionality test failed")
        print(f"     Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("INSTALLATION AND SETUP TEST")
    print("=" * 70)
    
    results = []
    
    # Test Python version
    results.append(test_python_version())
    
    # Test dependencies
    results.append(test_dependencies())
    
    # Test project modules
    results.append(test_project_modules())
    
    # Test basic functionality
    results.append(test_basic_functionality())
    
    # Test data loading (optional, can be slow)
    print("\n" + "=" * 70)
    response = input("Test MNIST data loading? (may take 1-2 minutes) [y/N]: ")
    if response.lower() == 'y':
        results.append(test_data_loading())
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    if all(results):
        print("✅ ALL TESTS PASSED!")
        print("\nYou are ready to run the main experiments:")
        print("  python main.py")
    else:
        print("❌ SOME TESTS FAILED")
        print("\nPlease fix the issues above before running main.py")
        print("\nTo install missing packages:")
        print("  pip install -r requirements.txt")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
