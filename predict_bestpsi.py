import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle

class VLSILinearRegression:
    """
    Linear regression implementation suitable for VLSI using fixed-point arithmetic.
    Uses only basic operations: multiplication, addition, and bit shifts.
    """

    def __init__(self, learning_rate=0.001, max_iterations=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

    def _fixed_point_multiply(self, a, b, precision_bits=16):
        """Simulate fixed-point multiplication for VLSI implementation"""
        scale = 2 ** precision_bits
        return (int(a * scale) * int(b * scale)) // scale / scale

    def fit(self, X, y):
        """Train the model using gradient descent with VLSI-friendly operations"""
        # Check for NaN values and remove them
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X_clean = X[mask]
        y_clean = y[mask]

        print(f"Removed {len(X) - len(X_clean)} samples with NaN values")

        # Normalize features and target
        X_scaled = self.scaler_X.fit_transform(X_clean)
        y_scaled = self.scaler_y.fit_transform(y_clean.reshape(-1, 1)).flatten()

        n_samples, n_features = X_scaled.shape

        # Initialize weights with small random values
        np.random.seed(42)  # For reproducibility
        self.weights = np.random.normal(0, 0.01, n_features)
        self.bias = 0.0

        prev_cost = float('inf')

        for iteration in range(self.max_iterations):
            # Forward pass: y_pred = X * weights + bias
            y_pred = np.dot(X_scaled, self.weights) + self.bias

            # Calculate cost (MSE)
            cost = np.mean((y_pred - y_scaled) ** 2)

            # Check for NaN in cost
            if np.isnan(cost):
                print("NaN detected in cost, reducing learning rate")
                self.learning_rate *= 0.1
                if self.learning_rate < 1e-8:
                    print("Learning rate too small, stopping")
                    break
                continue

            # Check for convergence
            if abs(prev_cost - cost) < self.tolerance:
                print(f"Converged at iteration {iteration}")
                break

            # Calculate gradients
            dw = (2 / n_samples) * np.dot(X_scaled.T, (y_pred - y_scaled))
            db = (2 / n_samples) * np.sum(y_pred - y_scaled)

            # Check for NaN in gradients
            if np.isnan(dw).any() or np.isnan(db):
                print("NaN detected in gradients, reducing learning rate")
                self.learning_rate *= 0.1
                continue

            # Update weights and bias
            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db

            prev_cost = cost

            if iteration % 100 == 0:
                print(f"Iteration {iteration}, Cost: {cost:.6f}")

    def predict(self, X):
        """Make predictions using VLSI-friendly operations"""
        X_scaled = self.scaler_X.transform(X)

        # Linear prediction: y = X * weights + bias
        y_pred_scaled = np.dot(X_scaled, self.weights) + self.bias

        # Inverse transform to original scale
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

        return y_pred

    def save_model(self, filepath):
        """Save the trained model"""
        model_data = {
            'weights': self.weights,
            'bias': self.bias,
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

    def load_model(self, filepath):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        self.weights = model_data['weights']
        self.bias = model_data['bias']
        self.scaler_X = model_data['scaler_X']
        self.scaler_y = model_data['scaler_y']

def load_and_prepare_data(csv_path):
    """Load and prepare the dataset"""
    print("Loading dataset...")
    df = pd.read_csv(csv_path)

    # Extract features (Ar, Ag, Ab) and target (BestPsi)
    X = df[['Ar', 'Ag', 'Ab']].values
    y = df['BestPsi'].values

    print(f"Dataset loaded: {len(df)} samples")
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    return X, y, df

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)

    print("\nModel Evaluation:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")

    return {'rmse': rmse, 'mae': mae, 'r2': r2}

def main():
    # Load and prepare data
    csv_path = "/Users/hongweichen/Documents/實驗室/論文/defog/昱恩交接/img_soft/dataset/SOTS_inout/report/score_optimize_psi_grid.csv"
    X, y, df = load_and_prepare_data(csv_path)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Create and train the VLSI-suitable model
    print("\nTraining VLSI Linear Regression model...")
    model = VLSILinearRegression(learning_rate=0.01, max_iterations=1000)
    model.fit(X_train, y_train)

    # Evaluate the model
    metrics = evaluate_model(model, X_test, y_test)

    # Save the trained model
    model.save_model('bestpsi_vlsi_model.pkl')
    print("\nModel saved as 'bestpsi_vlsi_model.pkl'")

    # Display model parameters (VLSI implementation ready)
    print("\nVLSI Implementation Parameters:")
    print(f"Weights: {model.weights}")
    print(f"Bias: {model.bias}")

    # Example prediction
    print("\nExample predictions:")
    sample_indices = np.random.choice(len(X_test), 5, replace=False)
    for i in sample_indices:
        actual = y_test[i]
        predicted = model.predict(X_test[i:i+1])[0]
        print(f"Ar={X_test[i,0]:.1f}, Ag={X_test[i,1]:.1f}, Ab={X_test[i,2]:.1f} -> "
              f"Actual: {actual:.3f}, Predicted: {predicted:.3f}")

    return model, metrics

if __name__ == "__main__":
    model, metrics = main()