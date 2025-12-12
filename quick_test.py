"""
ULTRA-FAST TEST VERSION
=======================
This runs in 3-5 minutes to verify everything works.
Use this to test before running the full version.

Author: ML Project
Date: December 2025
"""

import numpy as np
from datetime import datetime
from data_loader import MNISTDataLoader
from pipeline import MNISTClassificationPipeline

print("=" * 70)
print("ULTRA-FAST TEST - MNIST Classification")
print("=" * 70)
print("Expected time: 3-5 minutes")
print("=" * 70)

start_time = datetime.now()
print(f"\nStart time: {start_time.strftime('%H:%M:%S')}")

# Load data
print("\n[1/4] Loading MNIST dataset...")
loader = MNISTDataLoader(test_size=10000, random_state=42)
X_train, X_test, y_train, y_test = loader.load_data(normalize=True)

# Use SMALL subset
print("\n[2/4] Creating training subset (5,000 samples)...")
X_train_small, y_train_small = loader.get_subset(5000, 'train', random_state=42)

# Create pipeline with MINIMAL parameters
print("\n[3/4] Running PCA + SVM + PSO pipeline...")
print("  - PCA components: 30")
print("  - PSO particles: 5")
print("  - PSO iterations: 10")

pipeline = MNISTClassificationPipeline(
    reduction_method='pca',
    optimization_method='pso',
    n_components=30,    # Very small
    kernel='rbf'
)

results = pipeline.run(
    X_train_small, y_train_small,
    X_test, y_test,
    pso_particles=5,    # Very few particles
    pso_iterations=10,  # Very few iterations
    verbose=True
)

# Results
print("\n[4/4] Results:")
print("=" * 70)
print(f"Test Accuracy:    {results['test_accuracy']:.4f} ({results['test_accuracy']*100:.2f}%)")
print(f"Training Time:    {results['training_time']:.2f} seconds")
print(f"Total Time:       {results['total_time']:.2f} seconds")
print(f"Best C:           {results['best_params']['C']:.4f}")
print(f"Best gamma:       {results['best_params']['gamma']:.6f}")

end_time = datetime.now()
elapsed = (end_time - start_time).total_seconds()

print("\n" + "=" * 70)
print(f"COMPLETED in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
print("=" * 70)

print("\n✅ If this worked, you can now run:")
print("   - main_fast.py (10-15 minutes, better accuracy)")
print("   - main.py (30-45 minutes, best accuracy)")
