import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from matplotlib import cm
from PIL import Image
import os
import pandas as pd
import sys
from mpl_toolkits.mplot3d import Axes3D

# Load dataset
data = pd.read_csv(r"C:\Users\pasca\OneDrive\2024\2024_10\simple_variational_inference\winequality-red.csv")

# Select only 'citric acid' and 'alcohol' columns as inputs
X = data[['citric acid', 'alcohol']].values

# Add a column of ones for the bias term
ones_column = np.ones((X.shape[0], 1))       # Shape: (N, 1)
X = np.hstack((X, ones_column))              # Shape: (N, 3)

y = data["quality"].values

# Normalize target variable to be between 0 and 1
y = (y - y.min()) / (y.max() - y.min())
# After normalizing, set y to 0 if in [0, 0.5), and 1 otherwise
y = np.where(y < 0.5, 0, 1)

# Initialize variables/parameters
K = 3  # Dimension of x
mu = np.random.randn(K, 1) * 0.01  # Small random values
Sigma = np.eye(K)
eta = 0.1  # Learning rate

# Initialize lists to store mu and Sigma at each iteration
mu_history = []
Sigma_history = []

# Sum over all n (rows) to get the summed feature vector
s = np.sum(X, axis=0).reshape((K, 1))  # Shape: (K, 1)

# Define necessary functions
def sigma(alpha):
    return 1 / (1 + np.exp(-alpha))

def sech_squared(x):
    return 1 / np.cosh(x)**2

def lambda_func(alpha):
    # Handle division by zero
    alpha = np.array(alpha, dtype=np.float64)
    return np.where(alpha == 0, 1 / 8, np.tanh(alpha / 2) / (4 * alpha))

def lambda_prime(alpha):
    # Derivative of lambda(alpha)
    alpha = np.array(alpha, dtype=np.float64)
    result = np.zeros_like(alpha)  # Initialize result array

    # Create a mask for non-zero alpha values
    mask = alpha != 0 # Create a boolean mask where alpha is not zero

    # For alpha != 0, compute the derivative
    alpha_nonzero = alpha[mask] # use boolean indexing to extract elements from alpha
    sech2 = sech_squared(alpha_nonzero / 2)
    tanh_term = np.tanh(alpha_nonzero / 2)
    term1 = sech2 / (8 * alpha_nonzero)
    term2 = tanh_term / (4 * alpha_nonzero**2)
    result[mask] = term1 - term2

    # For alpha == 0, result is already zero
    return result

def compute_D(X, Sigma, mu, lambda_prime_alpha):
    # Ensure mu is a 1D array of shape (K,)
    mu = mu.flatten()

    # Compute q1 = x_n^T Sigma x_n for all n
    q1 = np.sum((X @ Sigma) * X, axis=1)  # Shape: (N,)

    # Compute q2 = (mu^T x_n)^2 for all n
    u = X @ mu  # Shape: (N,)
    q2 = u ** 2  # Shape: (N,)

    # Compute the terms
    term_n = (q1 - q2) * lambda_prime_alpha  # Shape: (N,)

    # Sum over n to get D
    D = np.sum(term_n)

    return D

# Compute alpha_hat_n
alpha_hat = X @ mu  # Shape: (N, 1)

# Compute lambda_prime_alpha as an array
lambda_prime_alpha = lambda_prime(alpha_hat.flatten())  # Shape: (N,)

# Compute D
D = compute_D(X, Sigma, mu, lambda_prime_alpha)

# Output the result
print("Computed D:", D)
print("D.shape:", D.shape)
#sys.exit()

num_iterations = 10  # Number of iterations

for iteration in range(num_iterations):
    # Compute alpha_hat_n
    alpha_hat_n = X @ mu  # Shape: (N, 1)

    # Compute lambda_alpha and lambda_prime_alpha
    lambda_alpha = lambda_func(alpha_hat_n.flatten())  # Shape: (N,)
    lambda_prime_alpha = lambda_prime(alpha_hat_n.flatten())  # Shape: (N,)

    # Update Sigma
    Lambda = np.diag(2 * lambda_alpha)  # Shape: (N, N)
    Sigma_inv = X.T @ Lambda @ X + np.eye(K)
    Sigma = np.linalg.inv(Sigma_inv)  # Shape: (K, K)

    # Update mu
    s = X.T @ (y - 0.5)  # Shape: (K, 1)
    mu = Sigma @ s  # Shape: (K, 1)

    # Save mu and Sigma
    mu_history.append(mu.copy())
    Sigma_history.append(Sigma.copy())

    # Optionally, compute D in each iteration
    D = compute_D(X, Sigma, mu, lambda_prime_alpha)
    print(f"Iteration {iteration+1}, D: {D}")


# Output the final values
print("Final mu:")
print(mu)
print("\nFinal Sigma:")
print(Sigma)
print("\nFinal alpha_hat:")
print(alpha_hat)

# Convert mu_history to array
mu_history_array = np.array(mu_history).reshape(-1, K)
mu_x = mu_history_array[:, 0]
mu_y = mu_history_array[:, 1]
mu_z = mu_history_array[:, 2]

# Convert Sigma_histroy to array
Sigma_history_array = np.array(Sigma_history)  # Shape: (num_iterations, K, K)
