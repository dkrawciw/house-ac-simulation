from house_ac import simulate_default
import numpy as np

from tqdm import tqdm
import pickle as pkl
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = Path(__file__).parent.parent / "paper"

# Plot Style Settings
sns.set_style("whitegrid")
sns.set_color_codes(palette="colorblind")
plt.rcParams.update(
        {
            "text.usetex": False,
            "mathtext.fontset": "cm",
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 12,
        }
    )

"""Simulate the house AC model"""
simulated_data = simulate_default()
model = simulated_data.model
y = simulated_data.measurements

t_vals = simulated_data.solution.t
t_range = (t_vals[0], t_vals[-1])
num_time_steps = y.shape[0]  # Number of time steps
filter_stride = 10000  # Stride for applying the Kalman filter (e.g., every 100 time steps)
num_kalman_steps = num_time_steps // filter_stride  # Number of Kalman filter steps
num_rooms = simulated_data.model.num_rooms
plotting_step_size = num_time_steps // 500
step_size = simulated_data.solution.t[1] - simulated_data.solution.t[0]

# Number of particles for the SMC algorithm
num_particles = 10000
weights = np.ones(num_particles) / num_particles  # Initialize weights uniformly
# particles = np.random.randn(num_particles, num_rooms)  # Initialize particles randomly
particles = np.random.normal(70, 5, (num_particles, num_rooms))  # Initialize particles around 70 degrees
Q = 0.1 * np.eye(num_rooms)  # Process noise covariance
R = 0.5 ** 2  # Measurement noise covariance

estimates = np.zeros((num_kalman_steps, num_rooms))  # To store state estimates at each time step
covariances = np.zeros((num_kalman_steps, num_rooms, num_rooms))  # To store covariance estimates at each time step


for k in tqdm(range(num_kalman_steps)):
    y_k = y[k * filter_stride]  # Measurement at time k
    t_k = t_vals[k * filter_stride]  # Time at step k
    dt_filter = t_vals[k * filter_stride] - t_k

    disturbance = np.random.multivariate_normal(np.zeros(num_rooms), Q, size=num_particles)

    for i in range(num_particles):
        x_i = particles[i]
        k1 = model.derivative(t_k, x_i)
        k2 = model.derivative(t_k + 0.5 * dt_filter, x_i + 0.5 * dt_filter * k1)
        particles[i] = x_i + dt_filter * k2 + disturbance[i]

    y_pred = simulated_data.measurement_matrix @ particles.T

    # Update weights based on measurement likelihood
    error = y_k - y_pred
    # weights *= np.exp(-0.5 * np.sum((error ** 2) / R))
    weights *= np.exp(-0.5 * (error ** 2) / R)
    weights += 1e-300  # Avoid zero weights
    weights /= np.sum(weights)  # Normalize weights

    # Estimate state
    x_hat = weights @ particles
    estimates[k] = x_hat

    # Estimate covariance
    # covariances[k] = weights @ ((particles - x_hat) ** 2)
    centered = particles - x_hat
    covariances[k] = (centered.T * weights) @ centered

# Plotting the results
kalman_time_points = np.linspace(t_range[0], t_range[1], num_kalman_steps)
measurement_time_points = np.linspace(t_range[0], t_range[1], num_time_steps)

plt.plot(kalman_time_points, simulated_data.measurement_matrix @ estimates.T, label="Estimated Temperature in Room 1", linewidth=2)
plt.plot(measurement_time_points[0:-1:plotting_step_size],y[0:-1:plotting_step_size], "--", label="True Temperature in Room 1", linewidth=2)

plt.xlabel("Time (s)")
plt.ylabel("Temperature (°F)")
plt.title("SMC State Estimation for Room 1")
plt.legend()

plt.tight_layout()

plt.savefig(OUTPUT_DIR / "smc_estimation_room1.svg")