"""
Simple Linear Regression with SGD - Core Implementation Only
Just the essential SGD algorithm from scratch - no fancy stuff here
"""

import csv
import random

# Load data
def load_data(filename):
    X, y = [], []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) >= 4:
                X.append([float(val) for val in row[:3]])  # 3 features
                y.append(float(row[3]))                     # 1 target
    return X, y

# Normalize features
def normalize(X):
    n_features = len(X[0])
    mins = [min(X[i][j] for i in range(len(X))) for j in range(n_features)]
    maxs = [max(X[i][j] for i in range(len(X))) for j in range(n_features)]
    
    X_norm = []
    for row in X:
        normalized_row = []
        for j in range(n_features):
            if maxs[j] - mins[j] != 0:
                normalized_row.append((row[j] - mins[j]) / (maxs[j] - mins[j]))
            else:
                normalized_row.append(0)
        X_norm.append(normalized_row)
    return X_norm, mins, maxs

# SGD Linear Regression Class
class LinearRegressionSGD:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        """Train using Stochastic Gradient Descent"""
        random.seed(42)
        n_samples, n_features = len(X), len(X[0])
        
        # Initialize weights and bias
        self.weights = [random.uniform(-0.1, 0.1) for _ in range(n_features)]
        self.bias = 0.0
        
        # SGD training
        for iteration in range(self.n_iterations):
            # Shuffle data
            indices = list(range(n_samples))
            random.shuffle(indices)
            
            # Process each sample
            for idx in indices:
                # Forward pass
                y_pred = self.bias + sum(self.weights[i] * X[idx][i] for i in range(n_features))
                
                # Calculate error
                error = y_pred - y[idx]
                
                # Update weights and bias
                for j in range(n_features):
                    self.weights[j] -= self.learning_rate * error * X[idx][j]
                self.bias -= self.learning_rate * error
            
            # Print progress
            if (iteration + 1) % 200 == 0:
                print(f"Iteration {iteration + 1}/{self.n_iterations}")
    
    def predict(self, X):
        """Make predictions"""
        predictions = []
        for x in X:
            pred = self.bias + sum(self.weights[i] * x[i] for i in range(len(x)))
            predictions.append(pred)
        return predictions
    
    def score(self, X, y):
        """Calculate R-squared"""
        y_pred = self.predict(X)
        y_mean = sum(y) / len(y)
        
        ss_tot = sum((y_i - y_mean) ** 2 for y_i in y)
        ss_res = sum((y_i - y_pred_i) ** 2 for y_i, y_pred_i in zip(y, y_pred))
        
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

# Main execution
def main():
    print("Simple Linear Regression with SGD")
    print("=" * 40)
    
    # Load and prepare data
    X, y = load_data("MultipleLR-Dataset.csv")
    print(f"Loaded {len(X)} samples with {len(X[0])} features")
    
    # Normalize features
    X_norm, mins, maxs = normalize(X)
    print("Features normalized")
    
    # Train model
    print("\nTraining model...")
    model = LinearRegressionSGD(learning_rate=0.01, n_iterations=1000)
    model.fit(X_norm, y)
    
    # Evaluate
    r2 = model.score(X_norm, y)
    print(f"\nR² Score: {r2:.4f}")
    print(f"Weights: {[f'{w:.4f}' for w in model.weights]}")
    print(f"Bias: {model.bias:.4f}")
    
    # Sample predictions
    predictions = model.predict(X_norm)
    print(f"\nSample predictions:")
    for i in range(5):
        error = abs(y[i] - predictions[i])
        print(f"Sample {i+1}: Actual={y[i]:.2f}, Predicted={predictions[i]:.2f}, Error={error:.2f}")

if __name__ == "__main__":
    main()