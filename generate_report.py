"""
Report Generation Script
========================
This script helps generate a complete project report by loading experimental
results and filling in the report template with actual values.

"""

import os
import json
import glob
from datetime import datetime


class ReportGenerator:
    """
    Generate project report from experimental results.
    """
    
    def __init__(self, results_dir):
        """
        Initialize report generator.
        
        Parameters:
            results_dir (str): Directory containing experimental results
        """
        self.results_dir = results_dir
        self.results = {}
        self.figures_dir = os.path.join(results_dir, "figures")
        self.data_dir = os.path.join(results_dir, "data")
    
    def load_results(self):
        """Load all result JSON files."""
        print(f"Loading results from {self.data_dir}...")
        
        if not os.path.exists(self.data_dir):
            print(f"Error: Data directory not found: {self.data_dir}")
            return False
        
        json_files = glob.glob(os.path.join(self.data_dir, "*.json"))
        
        if not json_files:
            print("Error: No result files found!")
            return False
        
        for filepath in json_files:
            filename = os.path.basename(filepath)
            method_name = filename.replace('_results.json', '')
            
            with open(filepath, 'r') as f:
                self.results[method_name] = json.load(f)
            
            print(f"  Loaded: {method_name}")
        
        print(f"\nTotal results loaded: {len(self.results)}")
        return True
    
    def print_summary(self):
        """Print summary of all results."""
        print("\n" + "=" * 80)
        print("EXPERIMENTAL RESULTS SUMMARY")
        print("=" * 80)
        
        for method_name, result in self.results.items():
            print(f"\n{method_name.upper().replace('_', ' ')}")
            print("-" * 80)
            
            # Accuracy
            test_acc = result.get('test_accuracy', 0)
            train_acc = result.get('train_accuracy', 0)
            print(f"Test Accuracy:       {test_acc:.4f} ({test_acc*100:.2f}%)")
            print(f"Training Accuracy:   {train_acc:.4f} ({train_acc*100:.2f}%)")
            
            # Timing
            print(f"\nTiming:")
            print(f"  Reduction Time:    {result.get('reduction_time', 0):.2f} seconds")
            print(f"  Optimization Time: {result.get('optimization_time', 0):.2f} seconds")
            print(f"  Training Time:     {result.get('training_time', 0):.2f} seconds")
            print(f"  Total Time:        {result.get('total_time', 0):.2f} seconds")
            
            # Best parameters
            best_params = result.get('best_params', {})
            if best_params:
                print(f"\nBest Parameters:")
                for param, value in best_params.items():
                    if param != 'accuracy':
                        if isinstance(value, float):
                            print(f"  {param}: {value:.6f}")
                        else:
                            print(f"  {param}: {value}")
    
    def generate_latex_table(self):
        """Generate LaTeX table for report."""
        print("\n" + "=" * 80)
        print("LATEX TABLE (Copy to your report)")
        print("=" * 80)
        
        print("""
\\begin{table}[h]
\\centering
\\caption{Experimental Results Summary}
\\begin{tabular}{|l|c|c|c|c|c|}
\\hline
\\textbf{Method} & \\textbf{Test Acc.} & \\textbf{Train Time} & \\textbf{Opt. Time} & \\textbf{Best C} & \\textbf{Best $\\gamma$} \\\\
\\hline""")
        
        for method_name, result in self.results.items():
            method_display = method_name.replace('_', ' ').title()
            test_acc = result.get('test_accuracy', 0) * 100
            train_time = result.get('training_time', 0)
            opt_time = result.get('optimization_time', 0)
            best_c = result.get('best_params', {}).get('C', 0)
            best_gamma = result.get('best_params', {}).get('gamma', 0)
            
            print(f"{method_display} & {test_acc:.2f}\\% & {train_time:.2f}s & {opt_time:.2f}s & {best_c:.4f} & {best_gamma:.6f} \\\\")
        
        print("""\\hline
\\end{tabular}
\\end{table}
""")
    
    def generate_markdown_table(self):
        """Generate Markdown table for report."""
        print("\n" + "=" * 80)
        print("MARKDOWN TABLE (Copy to your report)")
        print("=" * 80)
        
        print("\n| Method | Test Accuracy | Training Time | Optimization Time | Best C | Best γ |")
        print("|--------|--------------|---------------|-------------------|--------|--------|")
        
        for method_name, result in self.results.items():
            method_display = method_name.replace('_', ' ').title()
            test_acc = result.get('test_accuracy', 0)
            train_time = result.get('training_time', 0)
            opt_time = result.get('optimization_time', 0)
            best_c = result.get('best_params', {}).get('C', 0)
            best_gamma = result.get('best_params', {}).get('gamma', 0)
            
            print(f"| {method_display} | {test_acc:.4f} ({test_acc*100:.2f}%) | {train_time:.2f}s | {opt_time:.2f}s | {best_c:.4f} | {best_gamma:.6f} |")
    
    def list_figures(self):
        """List all generated figures."""
        print("\n" + "=" * 80)
        print("GENERATED FIGURES")
        print("=" * 80)
        
        if not os.path.exists(self.figures_dir):
            print("No figures directory found!")
            return
        
        figures = glob.glob(os.path.join(self.figures_dir, "*.png"))
        
        if not figures:
            print("No figures found!")
            return
        
        print(f"\nTotal figures: {len(figures)}")
        print("\nFigures for your report:")
        
        for i, fig_path in enumerate(sorted(figures), 1):
            fig_name = os.path.basename(fig_path)
            print(f"\n{i}. {fig_name}")
            print(f"   Path: {fig_path}")
            
            # Suggest caption based on filename
            caption = self._suggest_caption(fig_name)
            print(f"   Suggested caption: {caption}")
    
    def _suggest_caption(self, filename):
        """Suggest figure caption based on filename."""
        captions = {
            'sample_digits': 'Sample MNIST handwritten digits from the dataset',
            'class_distribution': 'Class distribution in training and test sets',
            'pca_explained_variance': 'PCA explained variance analysis',
            'pca_2d_projection': '2D PCA projection of MNIST digits',
            'kpca_2d_projection': '2D Kernel PCA projection of MNIST digits',
            'pca_svm_pso_confusion_matrix': 'Confusion matrix for PCA + SVM + PSO method',
            'pca_svm_grid_confusion_matrix': 'Confusion matrix for PCA + SVM + Grid Search method',
            'kpca_svm_pso_confusion_matrix': 'Confusion matrix for Kernel PCA + SVM + PSO method',
            'pso_convergence': 'PSO convergence behavior over iterations',
            'method_comparison': 'Comparison of different classification methods'
        }
        
        for key, caption in captions.items():
            if key in filename:
                return caption
        
        return "Figure caption"
    
    def generate_key_findings(self):
        """Generate key findings section."""
        print("\n" + "=" * 80)
        print("KEY FINDINGS (For your report)")
        print("=" * 80)
        
        if not self.results:
            print("No results loaded!")
            return
        
        # Find best method
        best_method = max(self.results.items(), 
                         key=lambda x: x[1].get('test_accuracy', 0))
        best_name, best_result = best_method
        
        # Find fastest method
        fastest_method = min(self.results.items(),
                           key=lambda x: x[1].get('total_time', float('inf')))
        fastest_name, fastest_result = fastest_method
        
        print("\n1. Best Accuracy:")
        print(f"   Method: {best_name.replace('_', ' ').title()}")
        print(f"   Accuracy: {best_result['test_accuracy']:.4f} ({best_result['test_accuracy']*100:.2f}%)")
        print(f"   Parameters: C={best_result['best_params']['C']:.4f}, "
              f"γ={best_result['best_params'].get('gamma', 'N/A')}")
        
        print("\n2. Fastest Method:")
        print(f"   Method: {fastest_name.replace('_', ' ').title()}")
        print(f"   Time: {fastest_result['total_time']:.2f} seconds")
        print(f"   Accuracy: {fastest_result['test_accuracy']:.4f} ({fastest_result['test_accuracy']*100:.2f}%)")
        
        # Compare PSO vs Grid if both available
        pso_results = {k: v for k, v in self.results.items() if 'pso' in k.lower()}
        grid_results = {k: v for k, v in self.results.items() if 'grid' in k.lower()}
        
        if pso_results and grid_results:
            print("\n3. PSO vs Grid Search Comparison:")
            
            pso_name, pso_res = list(pso_results.items())[0]
            grid_name, grid_res = list(grid_results.items())[0]
            
            acc_diff = pso_res['test_accuracy'] - grid_res['test_accuracy']
            time_diff = pso_res['optimization_time'] - grid_res['optimization_time']
            
            print(f"   Accuracy difference: {acc_diff:+.4f} ({acc_diff*100:+.2f}%)")
            print(f"   Time difference: {time_diff:+.2f} seconds")
            
            if acc_diff > 0:
                print(f"   → PSO achieved {abs(acc_diff)*100:.2f}% higher accuracy")
            else:
                print(f"   → Grid Search achieved {abs(acc_diff)*100:.2f}% higher accuracy")
            
            if time_diff < 0:
                print(f"   → PSO was {abs(time_diff):.2f}s faster")
            else:
                print(f"   → Grid Search was {abs(time_diff):.2f}s faster")
    
    def generate_full_report_data(self, output_file='report_data.txt'):
        """Generate a text file with all data for the report."""
        output_path = os.path.join(self.results_dir, output_file)
        
        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("PROJECT REPORT DATA\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Results Directory: {self.results_dir}\n")
            f.write("=" * 80 + "\n\n")
            
            # Write all results
            for method_name, result in self.results.items():
                f.write(f"\n{method_name.upper().replace('_', ' ')}\n")
                f.write("-" * 80 + "\n")
                f.write(json.dumps(result, indent=2))
                f.write("\n\n")
        
        print(f"\nFull report data saved to: {output_path}")


def main():
    """Main function."""
    print("=" * 80)
    print("PROJECT REPORT GENERATOR")
    print("=" * 80)
    
    # Find most recent results directory
    results_dirs = glob.glob("results_*")
    
    if not results_dirs:
        print("\nError: No results directories found!")
        print("Please run main.py first to generate results.")
        return
    
    # Use most recent directory
    latest_dir = max(results_dirs, key=os.path.getctime)
    
    print(f"\nUsing results from: {latest_dir}")
    
    # Create generator
    generator = ReportGenerator(latest_dir)
    
    # Load results
    if not generator.load_results():
        return
    
    # Generate all outputs
    generator.print_summary()
    generator.generate_markdown_table()
    generator.generate_latex_table()
    generator.list_figures()
    generator.generate_key_findings()
    generator.generate_full_report_data()
    
    print("\n" + "=" * 80)
    print("REPORT GENERATION COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Copy the tables and findings above into REPORT_TEMPLATE.md")
    print("2. Include the figures listed above in your report")
    print("3. Add your analysis and discussion")
    print("4. Convert to PDF for submission")
    print("=" * 80)


if __name__ == "__main__":
    main()
