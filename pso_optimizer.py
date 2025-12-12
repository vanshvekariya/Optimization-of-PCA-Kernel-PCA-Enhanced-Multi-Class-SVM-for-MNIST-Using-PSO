"""
Particle Swarm Optimization (PSO) Module
=========================================
This module implements Particle Swarm Optimization for hyperparameter tuning
of machine learning models, specifically designed for SVM hyperparameter optimization.

PSO is a population-based stochastic optimization technique inspired by the
social behavior of bird flocking or fish schooling.

Key Concepts:
- Particles: Candidate solutions in the search space
- Velocity: Direction and speed of particle movement
- Personal Best (pbest): Best position found by each particle
- Global Best (gbest): Best position found by the entire swarm
- Inertia Weight (w): Controls exploration vs exploitation
- Cognitive Component (c1): Attraction to personal best
- Social Component (c2): Attraction to global best

Author: ML Project
Date: December 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time


class Particle:
    """
    Represents a single particle in the PSO swarm.
    
    Attributes:
        position (np.ndarray): Current position in search space
        velocity (np.ndarray): Current velocity
        best_position (np.ndarray): Personal best position
        best_fitness (float): Personal best fitness value
        fitness (float): Current fitness value
    """
    
    def __init__(self, dim, bounds):
        """
        Initialize a particle with random position and velocity.
        
        Parameters:
            dim (int): Dimensionality of the search space
            bounds (list): List of (min, max) tuples for each dimension
        """
        self.dim = dim
        self.bounds = bounds
        
        # Initialize position randomly within bounds
        self.position = np.array([
            np.random.uniform(bounds[i][0], bounds[i][1]) 
            for i in range(dim)
        ])
        
        # Initialize velocity randomly (small values)
        velocity_range = [(bounds[i][1] - bounds[i][0]) * 0.1 for i in range(dim)]
        self.velocity = np.array([
            np.random.uniform(-velocity_range[i], velocity_range[i])
            for i in range(dim)
        ])
        
        # Initialize personal best
        self.best_position = self.position.copy()
        self.best_fitness = -np.inf
        self.fitness = -np.inf
    
    def update_velocity(self, global_best_position, w, c1, c2):
        """
        Update particle velocity using PSO velocity update equation.
        
        v_new = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)
        
        Parameters:
            global_best_position (np.ndarray): Global best position
            w (float): Inertia weight
            c1 (float): Cognitive (personal) coefficient
            c2 (float): Social (global) coefficient
        """
        r1 = np.random.random(self.dim)
        r2 = np.random.random(self.dim)
        
        # Cognitive component (personal best)
        cognitive = c1 * r1 * (self.best_position - self.position)
        
        # Social component (global best)
        social = c2 * r2 * (global_best_position - self.position)
        
        # Update velocity
        self.velocity = w * self.velocity + cognitive + social
        
        # Apply velocity clamping to prevent explosion
        max_velocity = [(self.bounds[i][1] - self.bounds[i][0]) * 0.2 
                       for i in range(self.dim)]
        for i in range(self.dim):
            self.velocity[i] = np.clip(self.velocity[i], 
                                      -max_velocity[i], 
                                      max_velocity[i])
    
    def update_position(self):
        """
        Update particle position and enforce boundary constraints.
        """
        self.position = self.position + self.velocity
        
        # Enforce bounds
        for i in range(self.dim):
            if self.position[i] < self.bounds[i][0]:
                self.position[i] = self.bounds[i][0]
                self.velocity[i] = 0  # Stop at boundary
            elif self.position[i] > self.bounds[i][1]:
                self.position[i] = self.bounds[i][1]
                self.velocity[i] = 0  # Stop at boundary


class ParticleSwarmOptimizer:
    """
    Particle Swarm Optimization algorithm for hyperparameter tuning.
    
    Attributes:
        n_particles (int): Number of particles in the swarm
        n_iterations (int): Maximum number of iterations
        bounds (list): Search space bounds for each parameter
        w (float): Inertia weight
        c1 (float): Cognitive coefficient
        c2 (float): Social coefficient
        global_best_position (np.ndarray): Best position found by swarm
        global_best_fitness (float): Best fitness value found
        history (dict): Optimization history
    """
    
    def __init__(self, n_particles=30, n_iterations=50, bounds=None,
                 w=0.7, c1=1.5, c2=1.5, w_decay=0.99):
        """
        Initialize PSO optimizer.
        
        Parameters:
            n_particles (int): Number of particles in swarm (default: 30)
            n_iterations (int): Maximum iterations (default: 50)
            bounds (list): List of (min, max) tuples for each parameter
            w (float): Inertia weight (default: 0.7)
                      - Controls exploration vs exploitation
                      - Higher w: more exploration
                      - Lower w: more exploitation
            c1 (float): Cognitive coefficient (default: 1.5)
                       - Attraction to personal best
            c2 (float): Social coefficient (default: 1.5)
                       - Attraction to global best
            w_decay (float): Decay rate for inertia weight (default: 0.99)
        """
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.bounds = bounds
        self.w_initial = w
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.w_decay = w_decay
        
        self.dim = len(bounds) if bounds else 0
        self.particles = []
        self.global_best_position = None
        self.global_best_fitness = -np.inf
        
        # History tracking
        self.history = {
            'global_best_fitness': [],
            'mean_fitness': [],
            'best_position': [],
            'iteration_time': []
        }
    
    def optimize(self, objective_function, verbose=True):
        """
        Run PSO optimization.
        
        Parameters:
            objective_function (callable): Function to maximize
                                          Takes position array, returns fitness value
            verbose (bool): Whether to print progress
            
        Returns:
            tuple: (best_position, best_fitness)
        """
        if verbose:
            print("\n" + "=" * 60)
            print("Particle Swarm Optimization")
            print("=" * 60)
            print(f"Particles: {self.n_particles}")
            print(f"Iterations: {self.n_iterations}")
            print(f"Search space dimensions: {self.dim}")
            print(f"Inertia weight (w): {self.w}")
            print(f"Cognitive coef (c1): {self.c1}")
            print(f"Social coef (c2): {self.c2}")
            print("=" * 60)
        
        # Initialize swarm
        self.particles = [Particle(self.dim, self.bounds) for _ in range(self.n_particles)]
        
        # Evaluate initial positions
        if verbose:
            print("\nInitializing swarm...")
        
        for particle in self.particles:
            particle.fitness = objective_function(particle.position)
            particle.best_fitness = particle.fitness
            
            if particle.fitness > self.global_best_fitness:
                self.global_best_fitness = particle.fitness
                self.global_best_position = particle.position.copy()
        
        if verbose:
            print(f"Initial best fitness: {self.global_best_fitness:.4f}")
        
        # Main PSO loop
        iterator = tqdm(range(self.n_iterations), desc="PSO Progress") if verbose else range(self.n_iterations)
        
        for iteration in iterator:
            iter_start_time = time.time()
            
            # Update each particle
            for particle in self.particles:
                # Update velocity and position
                particle.update_velocity(self.global_best_position, self.w, self.c1, self.c2)
                particle.update_position()
                
                # Evaluate new position
                particle.fitness = objective_function(particle.position)
                
                # Update personal best
                if particle.fitness > particle.best_fitness:
                    particle.best_fitness = particle.fitness
                    particle.best_position = particle.position.copy()
                
                # Update global best
                if particle.fitness > self.global_best_fitness:
                    self.global_best_fitness = particle.fitness
                    self.global_best_position = particle.position.copy()
            
            # Calculate mean fitness
            mean_fitness = np.mean([p.fitness for p in self.particles])
            
            # Store history
            self.history['global_best_fitness'].append(self.global_best_fitness)
            self.history['mean_fitness'].append(mean_fitness)
            self.history['best_position'].append(self.global_best_position.copy())
            self.history['iteration_time'].append(time.time() - iter_start_time)
            
            # Update inertia weight (linearly decreasing)
            self.w = self.w * self.w_decay
            
            # Update progress bar
            if verbose and hasattr(iterator, 'set_postfix'):
                iterator.set_postfix({
                    'Best': f'{self.global_best_fitness:.4f}',
                    'Mean': f'{mean_fitness:.4f}',
                    'w': f'{self.w:.3f}'
                })
        
        if verbose:
            print("\n" + "=" * 60)
            print("Optimization Completed!")
            print(f"Best fitness: {self.global_best_fitness:.4f}")
            print(f"Best position: {self.global_best_position}")
            print(f"Total time: {sum(self.history['iteration_time']):.2f} seconds")
            print("=" * 60)
        
        return self.global_best_position, self.global_best_fitness
    
    def plot_convergence(self, save_path=None):
        """
        Plot PSO convergence history.
        
        Parameters:
            save_path (str, optional): Path to save the figure
        """
        if not self.history['global_best_fitness']:
            print("No history to plot. Run optimize() first.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        iterations = range(1, len(self.history['global_best_fitness']) + 1)
        
        # Plot fitness convergence
        ax1.plot(iterations, self.history['global_best_fitness'], 
                label='Global Best', linewidth=2, color='red', marker='o', markersize=3)
        ax1.plot(iterations, self.history['mean_fitness'], 
                label='Swarm Mean', linewidth=2, color='blue', alpha=0.7)
        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('Fitness (Accuracy)', fontsize=12)
        ax1.set_title('PSO Convergence', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Plot parameter evolution (if 2D or 3D)
        if self.dim == 2:
            positions = np.array(self.history['best_position'])
            ax2.plot(positions[:, 0], positions[:, 1], 
                    marker='o', linestyle='-', linewidth=2, markersize=4, color='green')
            ax2.scatter(positions[0, 0], positions[0, 1], 
                       color='blue', s=100, marker='s', label='Start', zorder=5)
            ax2.scatter(positions[-1, 0], positions[-1, 1], 
                       color='red', s=100, marker='*', label='End', zorder=5)
            ax2.set_xlabel('Parameter 1', fontsize=12)
            ax2.set_ylabel('Parameter 2', fontsize=12)
            ax2.set_title('Best Position Evolution', fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(alpha=0.3)
        else:
            # Plot iteration time
            ax2.plot(iterations, self.history['iteration_time'], 
                    linewidth=2, color='purple')
            ax2.set_xlabel('Iteration', fontsize=12)
            ax2.set_ylabel('Time (seconds)', fontsize=12)
            ax2.set_title('Iteration Time', fontsize=14, fontweight='bold')
            ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Convergence plot saved to {save_path}")
        
        plt.show()
    
    def get_optimization_summary(self):
        """
        Get summary statistics of the optimization run.
        
        Returns:
            dict: Summary statistics
        """
        return {
            'best_fitness': self.global_best_fitness,
            'best_position': self.global_best_position,
            'total_iterations': len(self.history['global_best_fitness']),
            'total_time': sum(self.history['iteration_time']),
            'avg_iteration_time': np.mean(self.history['iteration_time']),
            'final_mean_fitness': self.history['mean_fitness'][-1] if self.history['mean_fitness'] else None,
            'improvement': self.global_best_fitness - self.history['global_best_fitness'][0] if self.history['global_best_fitness'] else 0
        }


class SVMPSOOptimizer:
    """
    Specialized PSO optimizer for SVM hyperparameter tuning.
    Optimizes C, gamma, and optionally n_components for PCA.
    """
    
    def __init__(self, X_train, y_train, X_val, y_val, 
                 optimize_pca=False, kernel='rbf',
                 n_particles=30, n_iterations=50):
        """
        Initialize SVM PSO optimizer.
        
        Parameters:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            X_val (np.ndarray): Validation features
            y_val (np.ndarray): Validation labels
            optimize_pca (bool): Whether to optimize PCA components
            kernel (str): SVM kernel type
            n_particles (int): Number of PSO particles
            n_iterations (int): Number of PSO iterations
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.optimize_pca = optimize_pca
        self.kernel = kernel
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        
        # Define search space bounds
        # C: [0.1, 100] in log scale
        # gamma: [0.0001, 1] in log scale
        # n_components: [20, min(200, n_features)] if optimizing PCA
        
        if optimize_pca:
            max_components = min(200, X_train.shape[1])
            self.bounds = [
                (-1, 2),      # log10(C): 0.1 to 100
                (-4, 0),      # log10(gamma): 0.0001 to 1
                (20, max_components)  # n_components
            ]
            self.param_names = ['C', 'gamma', 'n_components']
        else:
            self.bounds = [
                (-1, 2),      # log10(C): 0.1 to 100
                (-4, 0)       # log10(gamma): 0.0001 to 1
            ]
            self.param_names = ['C', 'gamma']
        
        self.pso = ParticleSwarmOptimizer(
            n_particles=n_particles,
            n_iterations=n_iterations,
            bounds=self.bounds,
            w=0.7,
            c1=1.5,
            c2=1.5,
            w_decay=0.99
        )
        
        self.evaluation_count = 0
    
    def _objective_function(self, position):
        """
        Objective function for PSO (accuracy on validation set).
        
        Parameters:
            position (np.ndarray): Particle position [log10(C), log10(gamma), n_components]
            
        Returns:
            float: Validation accuracy
        """
        from multiclass_svm import MultiClassSVM
        from dimensionality_reduction import DimensionalityReducer
        
        # Decode parameters
        C = 10 ** position[0]
        gamma = 10 ** position[1]
        
        X_train_use = self.X_train
        X_val_use = self.X_val
        
        # Apply PCA if optimizing components
        if self.optimize_pca:
            n_components = int(position[2])
            reducer = DimensionalityReducer(method='pca', n_components=n_components)
            X_train_use = reducer.fit_transform(self.X_train)
            X_val_use = reducer.transform(self.X_val)
        
        # Train SVM
        svm = MultiClassSVM(C=C, kernel=self.kernel, gamma=gamma)
        svm.fit(X_train_use, self.y_train, verbose=False)
        
        # Evaluate on validation set
        y_pred = svm.predict(X_val_use)
        accuracy = np.mean(y_pred == self.y_val)
        
        self.evaluation_count += 1
        
        return accuracy
    
    def optimize(self, verbose=True):
        """
        Run PSO optimization for SVM hyperparameters.
        
        Parameters:
            verbose (bool): Whether to print progress
            
        Returns:
            dict: Best parameters found
        """
        if verbose:
            print("\nOptimizing SVM hyperparameters using PSO...")
            print(f"Parameters to optimize: {self.param_names}")
        
        self.evaluation_count = 0
        
        # Run PSO
        best_position, best_fitness = self.pso.optimize(
            self._objective_function, 
            verbose=verbose
        )
        
        # Decode best parameters
        best_params = {
            'C': 10 ** best_position[0],
            'gamma': 10 ** best_position[1]
        }
        
        if self.optimize_pca:
            best_params['n_components'] = int(best_position[2])
        
        best_params['accuracy'] = best_fitness
        
        if verbose:
            print("\nBest parameters found:")
            for param, value in best_params.items():
                if param == 'accuracy':
                    print(f"  {param}: {value:.4f}")
                elif param == 'n_components':
                    print(f"  {param}: {int(value)}")
                else:
                    print(f"  {param}: {value:.6f}")
            print(f"\nTotal SVM evaluations: {self.evaluation_count}")
        
        return best_params
    
    def plot_convergence(self, save_path=None):
        """
        Plot PSO convergence.
        
        Parameters:
            save_path (str, optional): Path to save figure
        """
        self.pso.plot_convergence(save_path)


if __name__ == "__main__":
    # Example usage and testing
    print("=" * 60)
    print("PSO Optimizer Module - Test")
    print("=" * 60)
    
    # Test with a simple 2D optimization problem (Sphere function)
    def sphere_function(x):
        """Simple test function: minimize sum of squares (we maximize negative)"""
        return -np.sum(x**2)
    
    print("\nTest 1: Optimizing 2D Sphere Function")
    print("(Global optimum at [0, 0])")
    
    pso = ParticleSwarmOptimizer(
        n_particles=20,
        n_iterations=30,
        bounds=[(-5, 5), (-5, 5)],
        w=0.7,
        c1=1.5,
        c2=1.5
    )
    
    best_pos, best_fit = pso.optimize(sphere_function, verbose=True)
    
    print(f"\nFound optimum at: {best_pos}")
    print(f"Function value: {best_fit:.6f}")
    
    # Plot convergence
    pso.plot_convergence()
    
    print("\n" + "=" * 60)
    print("PSO optimizer test completed!")
    print("=" * 60)
