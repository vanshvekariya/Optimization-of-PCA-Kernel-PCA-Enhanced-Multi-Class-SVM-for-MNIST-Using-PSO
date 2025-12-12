# MNIST Handwritten Digit Classification Project

## Optimization of PCA/Kernel PCA-Enhanced Multi-Class SVM Using Particle Swarm Optimization

### Project Overview

This project implements a comprehensive machine learning pipeline for classifying MNIST handwritten digits using:
- **Principal Component Analysis (PCA)** for linear dimensionality reduction
- **Kernel PCA (RBF kernel)** for nonlinear feature extraction
- **Multi-class Support Vector Machine (SVM)** with hinge-loss formulation
- **Particle Swarm Optimization (PSO)** for hyperparameter tuning

The project compares different combinations of these techniques and evaluates their performance on the MNIST dataset.

---

## Table of Contents

1. [Installation](#installation)
2. [Project Structure](#project-structure)
3. [Usage](#usage)
4. [Modules Description](#modules-description)
5. [Experiments](#experiments)
6. [Results](#results)
7. [References](#references)

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone or download the project**
   ```bash
   git clone <repository-url>
   cd Optimization-of-PCA-Kernel-PCA-Enhanced-Multi-Class-SVM-for-MNIST-Using-PSO
   ```

2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python test_installation.py
   ```
   
   Or manually check:
   ```bash
   python -c "import numpy, sklearn, matplotlib; print('All packages installed successfully!')"
   ```

---

## Project Structure

```
Optimization-of-PCA-Kernel-PCA-Enhanced-Multi-Class-SVM-for-MNIST-Using-PSO/
│
├── data_loader.py              # MNIST data loading and preprocessing
├── dimensionality_reduction.py # PCA and Kernel PCA implementation
├── multiclass_svm.py           # Multi-class SVM classifier
├── pso_optimizer.py            # Particle Swarm Optimization
├── pipeline.py                 # Complete classification pipelines
├── visualization.py            # Results visualization utilities
├── main.py                     # Main execution script (full experiments)
├── main_fast.py                # Fast execution script (quick tests)
├── quick_test.py               # Quick validation script
├── test_installation.py        # Installation verification script
├── generate_report.py          # Report generation utilities
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── .gitignore                  # Git ignore rules

results_YYYYMMDD_HHMMSS/       # Generated results directory
├── figures/                    # Generated plots and visualizations
└── data/                       # Experimental results (JSON)
```

---

## Usage

### Quick Start

#### Option 1: Full Experiments (Recommended for complete analysis)

Run all experiments with default settings:

```bash
python main.py
```

This will:
1. Load the MNIST dataset (70,000 samples)
2. Run 6 comprehensive experiments
3. Generate visualizations and save results
4. Create a timestamped results directory

**Note:** This may take 10-20 minutes depending on your hardware.

#### Option 2: Fast Experiments (Quick validation)

Run a faster version with reduced dataset:

```bash
python main_fast.py
```

This runs experiments on a smaller subset for quick validation.

#### Option 3: Quick Test (Installation check)

Run a minimal test to verify everything works:

```bash
python quick_test.py
```

### Running Individual Modules

#### Test Data Loader
```bash
python data_loader.py
```

#### Test Dimensionality Reduction
```bash
python dimensionality_reduction.py
```

#### Test Multi-class SVM
```bash
python multiclass_svm.py
```

#### Test PSO Optimizer
```bash
python pso_optimizer.py
```

#### Test Visualization
```bash
python visualization.py
```

### Custom Experiments

You can create custom pipelines programmatically:

```python
from data_loader import MNISTDataLoader
from pipeline import MNISTClassificationPipeline

# Load data
loader = MNISTDataLoader(test_size=10000, random_state=42)
X_train, X_test, y_train, y_test = loader.load_data(normalize=True)

# Create pipeline
pipeline = MNISTClassificationPipeline(
    reduction_method='pca',      # 'pca' or 'kpca'
    optimization_method='pso',   # 'pso' or 'grid'
    n_components=100,            # Number of PCA components
    kernel='rbf'                 # SVM kernel type
)

# Run pipeline
results = pipeline.run(
    X_train, y_train, X_test, y_test,
    pso_particles=30,
    pso_iterations=50,
    verbose=True
)

print(f"Test Accuracy: {results['test_accuracy']:.4f}")
```

---

## Modules Description

### 1. `data_loader.py`

**Purpose:** Handle MNIST dataset loading and preprocessing

**Key Features:**
- Automatic download from OpenML
- Data normalization (0-1 range)
- Train-test splitting with stratification
- Subset extraction for experiments
- Visualization of sample digits
- Class distribution analysis

**Main Class:** `MNISTDataLoader`

**Example:**
```python
loader = MNISTDataLoader(test_size=10000, random_state=42)
X_train, X_test, y_train, y_test = loader.load_data(normalize=True)
loader.visualize_samples(n_samples=10)
```

---

### 2. `dimensionality_reduction.py`

**Purpose:** Implement PCA and Kernel PCA for feature extraction

**Key Features:**
- Linear PCA with explained variance analysis
- Kernel PCA with RBF kernel
- Automatic data standardization
- 2D projection visualization
- Optimal component selection

**Main Classes:**
- `DimensionalityReducer`: Main reduction class
- `PCAAnalyzer`: Analysis utilities

**Example:**
```python
# Linear PCA
reducer = DimensionalityReducer(method='pca', n_components=100)
X_reduced = reducer.fit_transform(X_train)
reducer.plot_explained_variance()

# Kernel PCA
kpca_reducer = DimensionalityReducer(
    method='kpca', 
    n_components=50, 
    kernel='rbf', 
    gamma=0.001
)
X_kpca = kpca_reducer.fit_transform(X_train)
```

---

### 3. `multiclass_svm.py`

**Purpose:** Multi-class SVM classification using One-vs-Rest strategy

**Key Features:**
- One-vs-Rest (OvR) multi-class strategy
- Support for linear and RBF kernels
- Comprehensive evaluation metrics
- Confusion matrix visualization
- Grid search for hyperparameter tuning

**Main Classes:**
- `MultiClassSVM`: SVM classifier
- `SVMGridSearch`: Grid search optimizer

**Example:**
```python
# Train SVM
svm = MultiClassSVM(C=10.0, kernel='rbf', gamma=0.001)
svm.fit(X_train, y_train)

# Evaluate
metrics = svm.evaluate(X_test, y_test)
svm.plot_confusion_matrix(X_test, y_test)
```

---

### 4. `pso_optimizer.py`

**Purpose:** Particle Swarm Optimization for hyperparameter tuning

**Key Features:**
- Standard PSO algorithm implementation
- Adaptive inertia weight
- Velocity clamping
- Convergence visualization
- Specialized SVM optimizer

**Main Classes:**
- `Particle`: Individual particle representation
- `ParticleSwarmOptimizer`: General PSO algorithm
- `SVMPSOOptimizer`: SVM-specific optimizer

**PSO Parameters:**
- `n_particles`: Number of particles in swarm (default: 30)
- `n_iterations`: Maximum iterations (default: 50)
- `w`: Inertia weight (default: 0.7)
- `c1`: Cognitive coefficient (default: 1.5)
- `c2`: Social coefficient (default: 1.5)

**Example:**
```python
# Optimize SVM hyperparameters
pso_optimizer = SVMPSOOptimizer(
    X_train, y_train, X_val, y_val,
    kernel='rbf',
    n_particles=30,
    n_iterations=50
)

best_params = pso_optimizer.optimize(verbose=True)
print(f"Best C: {best_params['C']}")
print(f"Best gamma: {best_params['gamma']}")
```

---

### 5. `pipeline.py`

**Purpose:** End-to-end classification pipelines

**Key Features:**
- Complete pipeline integration
- Automatic hyperparameter optimization
- Performance comparison
- Results serialization

**Main Classes:**
- `MNISTClassificationPipeline`: Single pipeline
- `PipelineComparison`: Compare multiple pipelines

**Example:**
```python
# Compare multiple pipelines
comparison = PipelineComparison()

comparison.add_pipeline('PCA + PSO', 'pca', 'pso', 100, 'rbf')
comparison.add_pipeline('PCA + Grid', 'pca', 'grid', 100, 'rbf')

results = comparison.run_comparison(X_train, y_train, X_test, y_test)
comparison.plot_comparison()
comparison.print_summary()
```

---

### 6. `visualization.py`

**Purpose:** Advanced visualization utilities

**Key Features:**
- Accuracy comparison plots
- Time comparison plots
- PSO vs Grid Search comparison
- PCA vs KPCA comparison
- Results summary tables

**Main Class:** `ResultsVisualizer`

**Example:**
```python
visualizer = ResultsVisualizer()

# Plot accuracy comparison
accuracies = {'PCA+PSO': 0.975, 'PCA+Grid': 0.972}
visualizer.plot_accuracy_comparison(accuracies)

# Compare PSO vs Grid
visualizer.plot_pso_vs_grid_comparison(pso_results, grid_results)
```

---

## Experiments

The `main.py` script runs 6 comprehensive experiments:

### Experiment 1: PCA Analysis
- Analyze explained variance
- Find optimal number of components
- 2D visualization of digit classes

### Experiment 2: PCA + SVM + PSO
- 100 PCA components
- PSO optimization (30 particles, 50 iterations)
- RBF kernel SVM
- Full training set (60,000 samples)

### Experiment 3: PCA + SVM + Grid Search
- 100 PCA components
- Grid search over C and gamma
- RBF kernel SVM
- Comparison with PSO

### Experiment 4: Kernel PCA + SVM + PSO
- 50 KPCA components (RBF kernel)
- PSO optimization (20 particles, 30 iterations)
- Subset of 10,000 training samples (computational efficiency)
- Nonlinear feature extraction

### Experiment 5: Comprehensive Comparison
- Compare all methods
- Performance metrics
- Time analysis
- Best parameter identification

### Experiment 6: PSO Convergence Analysis
- Detailed PSO convergence study
- Iteration-by-iteration analysis
- Swarm behavior visualization

---

## Results

### Expected Outputs

After running `main.py`, you will find:

#### 1. Figures Directory
- `sample_digits.png` - Sample MNIST digits
- `class_distribution.png` - Class distribution plots
- `pca_explained_variance.png` - PCA variance analysis
- `pca_2d_projection.png` - 2D PCA visualization
- `kpca_2d_projection.png` - 2D KPCA visualization
- `pca_svm_pso_confusion_matrix.png` - Confusion matrix (PCA+PSO)
- `pca_svm_grid_confusion_matrix.png` - Confusion matrix (PCA+Grid)
- `kpca_svm_pso_confusion_matrix.png` - Confusion matrix (KPCA+PSO)
- `pso_convergence.png` - PSO convergence plot
- `method_comparison.png` - Overall method comparison

#### 2. Data Directory
- `pca_svm_pso_results.json` - PCA+PSO results
- `pca_svm_grid_results.json` - PCA+Grid results
- `kpca_svm_pso_results.json` - KPCA+PSO results

### Typical Performance

Based on the MNIST dataset:

| Method | Expected Accuracy | Typical Time |
|--------|------------------|--------------|
| PCA + SVM + PSO | 97-98% | 3-5 minutes |
| PCA + SVM + Grid | 97-98% | 5-8 minutes |
| KPCA + SVM + PSO | 96-97% | 8-12 minutes |

*Note: Times are approximate and depend on hardware*

---

## Technical Details

### Hyperparameter Search Spaces

**PSO Search Space:**
- C (penalty parameter): [0.1, 100] (log scale)
- gamma (RBF kernel): [0.0001, 1] (log scale)
- n_components (if optimized): [20, 200]

**Grid Search Space (default):**
- C: [0.1, 1, 10, 100]
- gamma: [0.0001, 0.001, 0.01, 0.1]

### PSO Algorithm Details

**Velocity Update Equation:**
```
v(t+1) = w*v(t) + c1*r1*(pbest - x(t)) + c2*r2*(gbest - x(t))
```

Where:
- `w`: Inertia weight (controls exploration vs exploitation)
- `c1`: Cognitive coefficient (attraction to personal best)
- `c2`: Social coefficient (attraction to global best)
- `r1, r2`: Random numbers in [0, 1]
- `pbest`: Personal best position
- `gbest`: Global best position

**Position Update:**
```
x(t+1) = x(t) + v(t+1)
```

### SVM Formulation

**Multi-class Strategy:** One-vs-Rest (OvR)

For each class k, solve:
```
min (1/2)||w||² + C * Σ max(0, 1 - y_i * f(x_i))
```

Where:
- `C`: Penalty parameter
- `f(x)`: Decision function
- Hinge loss: `max(0, 1 - y*f(x))`

**RBF Kernel:**
```
K(x, x') = exp(-gamma * ||x - x'||²)
```

---

## Troubleshooting

### Common Issues

**1. Memory Error**
- Reduce training set size
- Decrease number of PCA components
- Use subset for KPCA experiments

**2. Slow Execution**
- Reduce PSO particles/iterations
- Use fewer grid search points
- Enable parallel processing (already configured)

**3. Import Errors**
- Verify all packages installed: `pip install -r requirements.txt`
- Check Python version: `python --version` (should be 3.8+)

**4. Dataset Download Issues**
- Check internet connection
- Manually download MNIST from: https://www.openml.org/d/554
- Place in local directory and update `data_loader.py`

---

## Performance Optimization Tips

1. **For faster experimentation:**
   - Use smaller subset of data
   - Reduce PSO iterations
   - Use linear kernel instead of RBF

2. **For better accuracy:**
   - Increase PCA components (up to 200)
   - Increase PSO particles and iterations
   - Fine-tune search space bounds

3. **For production use:**
   - Save trained models using `joblib`
   - Implement early stopping in PSO
   - Use cross-validation for robust evaluation

---

## References

### Papers

1. J. Weston and C. Watkins, "Support vector machines for multi-class pattern recognition," ESANN'1999 Proceedings, 1999.

2. K. Crammer and Y. Singer, "On the algorithmic implementation of multiclass kernel-based vector machines," Journal of Machine Learning Research, 2001.

3. J. Kennedy and R. Eberhart, "Particle swarm optimization," Proceedings of IEEE International Conference on Neural Networks, 1995.

4. B. Schölkopf, A. Smola, and K.-R. Müller, "Nonlinear component analysis as a kernel eigenvalue problem," Neural Computation, 1998.

### Dataset

- Y. LeCun, C. Cortes, and C. J. Burges, "MNIST handwritten digit database," 1998.
- Available at: http://yann.lecun.com/exdb/mnist/

### Libraries

- scikit-learn: https://scikit-learn.org/
- NumPy: https://numpy.org/
- Matplotlib: https://matplotlib.org/

---


## License

This project is for educational purposes as part of a machine learning course project.

---

## Acknowledgments

- MNIST dataset creators
- scikit-learn developers
- Course instructors and TAs

---

## Contact

For questions or issues, please refer to the project documentation or contact the course instructor.

---

**Last Updated:** December 2025
