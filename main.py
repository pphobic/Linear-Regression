"""
Linear Regression with Stochastic Gradient Descent (SGD) - Professional Version
Complete ML Pipeline Implementation with Professional Terminal Output (Dashboard Edition)

This file implements a complete ML pipeline following the standard sequence:
1. Problem Definition
2. Data Collection  
3. Data Preprocessing
4. Data Exploration & Visualization
5. Model Selection
6. Model Training
7. Model Evaluation
8. Model Optimization
9. Model Deployment
"""

import csv
import random
import math

# =============================================================================
# FIX FOR WINDOWS: Set the matplotlib backend to ensure plots are displayed
# =============================================================================
import matplotlib
matplotlib.use('TkAgg')  # Use the TkAgg backend for interactive plots

import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# CONFIGURATION SETTINGS - Edit these values as needed
# =============================================================================
LEARNING_RATE = 0.01          # SGD learning rate
N_ITERATIONS = 1000           # Number of training epochs
TEST_SIZE = 0.2               # Train/test split ratio (0.2 = 20% test)
RANDOM_SEED = 42              # For reproducible results
SHOW_PLOTS = True             # Set to True to enable visualizations
PRINT_PROGRESS_EVERY = 100    # Print training progress every N iterations

# =============================================================================
# 1. PROBLEM DEFINITION
# =============================================================================
"""
Problem: Predict a target variable using 3 input features
Dataset: MultipleLR-Dataset.csv with 25 samples, 4 columns (3 features + 1 target)
Goal: Implement Linear Regression using Stochastic Gradient Descent from scratch
Requirements: No external ML libraries, professional interface, comprehensive analysis
"""

# ANSI Color Codes for terminal formatting
class Colors:
    RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = '\033[91m', '\033[92m', '\033[93m', '\033[94m', '\033[95m', '\033[96m', '\033[97m'
    BOLD, RESET = '\033[1m', '\033[0m'
    HEADER, SUCCESS, WARNING, ERROR, INFO, HIGHLIGHT = BOLD + CYAN, BOLD + GREEN, BOLD + YELLOW, BOLD + RED, BOLD + BLUE, BOLD + MAGENTA

# =============================================================================
# 2. DATA COLLECTION
# =============================================================================
def load_csv(filename):
    """Load data from CSV file"""
    X, y = [], []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) >= 4:
                X.append([float(val) for val in row[:3]])
                y.append(float(row[3]))
    return X, y

# =============================================================================
# 3. DATA PREPROCESSING
# =============================================================================
def normalize_features(X):
    """Normalize input features to [0, 1] range"""
    n_features = len(X[0])
    X_norm = []
    mins = [min(col) for col in zip(*X)]
    maxs = [max(col) for col in zip(*X)]
    
    for row in X:
        normalized_row = [(row[j] - mins[j]) / (maxs[j] - mins[j]) if (maxs[j] - mins[j]) != 0 else 0 for j in range(n_features)]
        X_norm.append(normalized_row)
    
    return X_norm, mins, maxs

def train_test_split(X, y, test_size=TEST_SIZE, random_seed=RANDOM_SEED):
    """Split data into training and testing sets"""
    random.seed(random_seed)
    n_samples = len(X)
    indices = list(range(n_samples))
    random.shuffle(indices)
    
    split_idx = int(n_samples * (1 - test_size))
    train_indices, test_indices = indices[:split_idx], indices[split_idx:]
    
    X_train, X_test = [X[i] for i in train_indices], [X[i] for i in test_indices]
    y_train, y_test = [y[i] for i in train_indices], [y[i] for i in test_indices]
    
    return X_train, X_test, y_train, y_test

# =============================================================================
# 5. MODEL SELECTION
# =============================================================================
class LinearRegressionSGD:
    def __init__(self, learning_rate=LEARNING_RATE, n_iterations=N_ITERATIONS, random_seed=RANDOM_SEED):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.random_seed = random_seed
        self.weights = None
        self.bias = None
        self.losses = []
    
    def fit(self, X, y):
        random.seed(self.random_seed)
        n_samples, n_features = len(X), len(X[0])
        self.weights = [random.uniform(-0.1, 0.1) for _ in range(n_features)]
        self.bias = 0.0
        
        for iteration in range(self.n_iterations):
            total_loss = 0
            indices = list(range(n_samples))
            random.shuffle(indices)
            
            for idx in indices:
                y_pred = self._predict_single(X[idx])
                error = y_pred - y[idx]
                total_loss += error ** 2
                
                for j in range(n_features):
                    self.weights[j] -= self.learning_rate * error * X[idx][j]
                self.bias -= self.learning_rate * error
            
            avg_loss = total_loss / n_samples
            self.losses.append(avg_loss)
            
            if (iteration + 1) % PRINT_PROGRESS_EVERY == 0:
                print(f"{Colors.INFO}Iteration {iteration + 1}/{self.n_iterations}{Colors.RESET}, {Colors.SUCCESS}Loss: {avg_loss:.4f}{Colors.RESET}")
    
    def _predict_single(self, x):
        return self.bias + sum(self.weights[i] * x[i] for i in range(len(x)))
    
    def predict(self, X):
        return [self._predict_single(x) for x in X]
    
    def score(self, X, y):
        y_pred = self.predict(X)
        y_mean = sum(y) / len(y)
        ss_tot = sum((y_i - y_mean) ** 2 for y_i in y)
        ss_res = sum((y_i - y_pred_i) ** 2 for y_i, y_pred_i in zip(y, y_pred))
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    def mean_squared_error(self, X, y):
        y_pred = self.predict(X)
        return sum((y_i - y_pred_i) ** 2 for y_i, y_pred_i in zip(y, y_pred)) / len(y)
    
    def mean_absolute_error(self, X, y):
        y_pred = self.predict(X)
        return sum(abs(y_i - y_pred_i) for y_i, y_pred_i in zip(y, y_pred)) / len(y)

# =============================================================================
# 6. MODEL TRAINING & EVALUATION
# =============================================================================
def train_model(X_train, y_train):
    print(f"\n{Colors.INFO}[4]{Colors.RESET} {Colors.BOLD}Training Linear Regression model...{Colors.RESET}")
    model = LinearRegressionSGD()
    model.fit(X_train, y_train)
    print(f"    {Colors.SUCCESS}✓ Model training completed!{Colors.RESET}")
    return model

def evaluate_model(model, X_train_norm, y_train, X_test_norm, y_test):
    print(f"\n{Colors.HEADER}{'=' * 60}\n{Colors.HEADER}{Colors.BOLD}RESULTS\n{Colors.HEADER}{'=' * 60}{Colors.RESET}")
    
    print(f"\n{Colors.HIGHLIGHT}Model Parameters{Colors.RESET}")
    print(f"  {Colors.BOLD}Weights:{Colors.RESET} {Colors.CYAN}{[f'{w:.4f}' for w in model.weights]}{Colors.RESET}")
    print(f"  {Colors.BOLD}Bias:{Colors.RESET} {Colors.CYAN}{model.bias:.4f}{Colors.RESET}")
    
    train_r2, train_mse, train_mae = model.score(X_train_norm, y_train), model.mean_squared_error(X_train_norm, y_train), model.mean_absolute_error(X_train_norm, y_train)
    test_r2, test_mse, test_mae = model.score(X_test_norm, y_test), model.mean_squared_error(X_test_norm, y_test), model.mean_absolute_error(X_test_norm, y_test)
    
    print(f"\n{Colors.HIGHLIGHT}Training Performance{Colors.RESET}")
    print(f"  {Colors.BOLD}R² Score:{Colors.RESET} {Colors.SUCCESS}{train_r2:.4f}{Colors.RESET}, {Colors.BOLD}MSE:{Colors.RESET} {Colors.YELLOW}{train_mse:.4f}{Colors.RESET}, {Colors.BOLD}MAE:{Colors.RESET} {Colors.YELLOW}{train_mae:.4f}{Colors.RESET}")
    
    print(f"\n{Colors.HIGHLIGHT}Testing Performance{Colors.RESET}")
    print(f"  {Colors.BOLD}R² Score:{Colors.RESET} {Colors.SUCCESS}{test_r2:.4f}{Colors.RESET}, {Colors.BOLD}MSE:{Colors.RESET} {Colors.YELLOW}{test_mse:.4f}{Colors.RESET}, {Colors.BOLD}MAE:{Colors.RESET} {Colors.YELLOW}{test_mae:.4f}{Colors.RESET}")
    
    print(f"\n{Colors.HIGHLIGHT}Sample Predictions on Test Set{Colors.RESET}")
    y_pred_test = model.predict(X_test_norm)
    for i in range(min(5, len(X_test_norm))):
        error = abs(y_test[i] - y_pred_test[i])
        error_color = Colors.SUCCESS if error < 1.0 else Colors.WARNING if error < 2.0 else Colors.ERROR
        print(f"  {Colors.BOLD}Sample {i+1}:{Colors.RESET} {Colors.CYAN}Actual = {y_test[i]:.2f}{Colors.RESET}, {Colors.MAGENTA}Predicted = {y_pred_test[i]:.2f}{Colors.RESET}, {error_color}Error = {error:.2f}{Colors.RESET}")

# =============================================================================
# 8. MODEL OPTIMIZATION
# =============================================================================
def optimize_model(X_train_norm, y_train, X_test_norm, y_test):
    print(f"\n{Colors.HIGHLIGHT}Model Optimization{Colors.RESET}")
    best_r2, best_lr = -float('inf'), None
    
    for lr in [0.001, 0.01, 0.1]:
        model = LinearRegressionSGD(learning_rate=lr, n_iterations=500, random_seed=42)
        model.fit(X_train_norm, y_train)
        r2 = model.score(X_test_norm, y_test)
        print(f"  {Colors.BOLD}Learning Rate: {lr}{Colors.RESET}, {Colors.CYAN}Test R²: {r2:.4f}{Colors.RESET}")
        if r2 > best_r2: best_r2, best_lr = r2, lr
            
    print(f"  {Colors.SUCCESS}✓ Best Learning Rate: {best_lr} (R²: {best_r2:.4f}){Colors.RESET}")
    return best_lr

# =============================================================================
# 9. MODEL DEPLOYMENT (DASHBOARD VISUALIZATION)
# =============================================================================
def create_dashboard(model, X_train_norm, y_train, X_test_norm, y_test):
    """Create a single, comprehensive visualization dashboard."""
    if not SHOW_PLOTS: return
        
    print(f"\n{Colors.HIGHLIGHT}Creating Full Analysis Dashboard...{Colors.RESET}")
    
    try:
        y_train_pred = model.predict(X_train_norm)
        y_test_pred = model.predict(X_test_norm)
        
        # Create a 3x3 grid for the plots
        fig, axs = plt.subplots(3, 3, figsize=(22, 18))
        fig.suptitle('Linear Regression with SGD - Complete Analysis Dashboard', fontsize=20, fontweight='bold')

        # 1. Training Loss Convergence
        axs[0, 0].plot(model.losses, color='#2E86AB', linewidth=2)
        axs[0, 0].set_title('1. Training Loss Convergence', fontsize=14, fontweight='bold')
        axs[0, 0].set_xlabel('Iteration', fontweight='bold')
        axs[0, 0].set_ylabel('Loss (MSE)', fontweight='bold')
        axs[0, 0].grid(True, alpha=0.3)

        # 2. Training Set: Predictions vs Actual
        axs[0, 1].scatter(y_train, y_train_pred, alpha=0.7, color='#2E86AB')
        min_val_train = min(min(y_train), min(y_train_pred))
        max_val_train = max(max(y_train), max(y_train_pred))
        axs[0, 1].plot([min_val_train, max_val_train], [min_val_train, max_val_train], 'r--', lw=2)
        axs[0, 1].set_title('2. Training Set: Predictions vs Actual', fontsize=14, fontweight='bold')
        axs[0, 1].set_xlabel('Actual Values', fontweight='bold')
        axs[0, 1].set_ylabel('Predicted Values', fontweight='bold')
        axs[0, 1].grid(True, alpha=0.3)

        # 3. Test Set: Predictions vs Actual
        axs[0, 2].scatter(y_test, y_test_pred, alpha=0.8, color='#E63946')
        min_val_test = min(min(y_test), min(y_test_pred))
        max_val_test = max(max(y_test), max(y_test_pred))
        axs[0, 2].plot([min_val_test, max_val_test], [min_val_test, max_val_test], 'r--', lw=2)
        axs[0, 2].set_title('3. Test Set: Predictions vs Actual', fontsize=14, fontweight='bold')
        axs[0, 2].set_xlabel('Actual Values', fontweight='bold')
        axs[0, 2].set_ylabel('Predicted Values', fontweight='bold')
        axs[0, 2].grid(True, alpha=0.3)
        
        # 4. Residual Analysis
        residuals = [actual - pred for actual, pred in zip(y_test, y_test_pred)]
        axs[1, 0].scatter(y_test_pred, residuals, alpha=0.7, color='#F18F01')
        axs[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=2)
        axs[1, 0].set_title('4. Residual Analysis', fontsize=14, fontweight='bold')
        axs[1, 0].set_xlabel('Predicted Values', fontweight='bold')
        axs[1, 0].set_ylabel('Residuals', fontweight='bold')
        axs[1, 0].grid(True, alpha=0.3)
        
        # 5. Feature Importance
        feature_names = ['Feature 1', 'Feature 2', 'Feature 3']
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        axs[1, 1].bar(feature_names, model.weights, color=colors, alpha=0.8)
        axs[1, 1].set_title('5. Feature Importance (Weights)', fontsize=14, fontweight='bold')
        axs[1, 1].set_ylabel('Weight Value', fontweight='bold')
        axs[1, 1].grid(True, alpha=0.3, axis='y')
        
        # 6. Error Distribution
        errors = [abs(actual - pred) for actual, pred in zip(y_test, y_test_pred)]
        axs[1, 2].hist(errors, bins=8, alpha=0.7, color='#A23B72', edgecolor='white')
        axs[1, 2].axvline(np.mean(errors), color='red', linestyle='--', lw=2, label=f'Mean Error: {np.mean(errors):.2f}')
        axs[1, 2].set_title('6. Error Distribution (Test Set)', fontsize=14, fontweight='bold')
        axs[1, 2].set_xlabel('Absolute Error', fontweight='bold')
        axs[1, 2].set_ylabel('Frequency', fontweight='bold')
        axs[1, 2].legend()
        axs[1, 2].grid(True, alpha=0.3, axis='y')
        
        # 7. Performance Metrics Comparison
        metrics = ['R² Score', 'MSE', 'MAE']
        train_r2 = model.score(X_train_norm, y_train)
        test_r2 = model.score(X_test_norm, y_test)
        train_mse = model.mean_squared_error(X_train_norm, y_train)
        test_mse = model.mean_squared_error(X_test_norm, y_test)
        train_mae = model.mean_absolute_error(X_train_norm, y_train)
        test_mae = model.mean_absolute_error(X_test_norm, y_test)
        train_values = [train_r2, train_mse, train_mae]
        test_values = [test_r2, test_mse, test_mae]
        x = np.arange(len(metrics))
        width = 0.35
        axs[2, 0].bar(x - width/2, train_values, width, label='Training Set', color='#2E86AB')
        axs[2, 0].bar(x + width/2, test_values, width, label='Test Set', color='#E63946')
        axs[2, 0].set_title('7. Performance Metrics: Train vs Test', fontsize=14, fontweight='bold')
        axs[2, 0].set_ylabel('Values', fontweight='bold')
        axs[2, 0].set_xticks(x)
        axs[2, 0].set_xticklabels(metrics)
        axs[2, 0].legend()
        axs[2, 0].grid(True, alpha=0.3, axis='y')

        # Hide unused subplots for a cleaner look
        fig.delaxes(axs[2, 1])
        fig.delaxes(axs[2, 2])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
        plt.show()
        
        print(f"  {Colors.SUCCESS}✓ Dashboard displayed successfully!{Colors.RESET}")
        
    except Exception as e:
        print(f"  {Colors.ERROR}✗ Visualization error: {e}{Colors.RESET}")

def main():
    """Main function executing the complete ML pipeline"""
    print(f"{Colors.HEADER}{'=' * 60}\n{Colors.HEADER}{Colors.BOLD}Linear Regression with SGD - Complete ML Pipeline\n{Colors.HEADER}{'=' * 60}{Colors.RESET}")
    
    # Data Collection
    print(f"\n{Colors.INFO}[1]{Colors.RESET} {Colors.BOLD}Data Collection...{Colors.RESET}")
    try:
        X, y = load_csv("MultipleLR-Dataset.csv")
        print(f"    {Colors.SUCCESS}[OK] Loaded {len(X)} samples with {len(X[0])} features (4 columns total){Colors.RESET}")
    except FileNotFoundError:
        print(f"    {Colors.ERROR}✗ ERROR: 'MultipleLR-Dataset.csv' not found!{Colors.RESET}")
        return
    
    # Data Preprocessing
    print(f"\n{Colors.INFO}[2]{Colors.RESET} {Colors.BOLD}Data Preprocessing...{Colors.RESET}")
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    print(f"    {Colors.SUCCESS}✓ Data split: {len(X_train)} training samples, {len(X_test)} testing samples.{Colors.RESET}")
    
    X_train_norm, mins, maxs = normalize_features(X_train)
    X_test_norm = [ [(row[j] - mins[j]) / (maxs[j] - mins[j]) if (maxs[j] - mins[j]) != 0 else 0 for j in range(len(row))] for row in X_test]
    print(f"    {Colors.SUCCESS}✓ Features normalized.{Colors.RESET}")
    
    # Model Training, Evaluation, and Optimization
    model = train_model(X_train_norm, y_train)
    evaluate_model(model, X_train_norm, y_train, X_test_norm, y_test)
    best_lr = optimize_model(X_train_norm, y_train, X_test_norm, y_test)
    
    # Final Model Deployment (Visualization Dashboard)
    print(f"\n{Colors.INFO}[5]{Colors.RESET} {Colors.BOLD}Generating Final Dashboard with optimized learning rate ({best_lr})...{Colors.RESET}")
    final_model = LinearRegressionSGD(learning_rate=best_lr)
    final_model.fit(X_train_norm, y_train)
    create_dashboard(final_model, X_train_norm, y_train, X_test_norm, y_test)
    
    print(f"\n{Colors.HEADER}{'=' * 60}\n{Colors.SUCCESS}{Colors.BOLD}ML Pipeline completed successfully!\n{Colors.HEADER}{'=' * 60}{Colors.RESET}")

if __name__ == "__main__":
    main()