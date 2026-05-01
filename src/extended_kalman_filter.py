from house_ac import simulate_default
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import numpy as np
from tqdm import tqdm
import pickle as pkl

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
y = simulated_data.measurements
N = y.shape[0]  # Number of state variables (temperatures in each room)
num_rooms = simulated_data.model.num_rooms
step_size = N // 500

# """Plot the measured temperature of a room in the house with AC"""
# plt.figure(figsize=(10, 6))
# plt.plot(simulated_data.solution.t[0:-1:step_size], y[0:-1:step_size], label="Measured Temperature", linestyle="--", linewidth=4)

# plt.title("Measured Temperature of a Room in a House with AC")
# plt.xlabel("Time (seconds)")
# plt.ylabel("Measured Temperature (°F)")

# plt.tight_layout()
# # plt.show()
# plt.savefig(OUTPUT_DIR / "simulated_measured_temperature.svg")

# """Do the Extended Kalman Filter on the Simulated data"""
# x = np.zeros_like(simulated_data.solution.y)  # Initial state guess (temperatures in each room)
# P_0 = np.eye(num_rooms) * 1000.0  # Initial covariance guess
# R = np.eye(1) * 4.0  # Measurement noise covariance (variance of measurement noise)
# Q = np.eye(num_rooms) * 0.1  # Process noise covariance (small values)

# P = P_0
# P_pred = P_0
# x[:, 0] = simulated_data.solution.y[:, 0] + np.ones_like(simulated_data.solution.y[:, 0]) * np.random.randn(num_rooms)  # Initial state with small random noise

# for k in tqdm(range(1, len(simulated_data.solution.t)), desc="Extended Kalman Filter"):    
#     x_k = x[:, k-1]  # Previous state estimate
#     t_k = simulated_data.solution.t[k-1]  # Previous time step
    
#     # Prediction Step
#     dt = simulated_data.solution.t[k] - simulated_data.solution.t[k-1]
#     x_pred = x_k + dt*simulated_data.model.derivative(t_k, x_k)
    
#     # Construct the Jacobian matrix A_k using finite differences
#     A_k = np.zeros((num_rooms, num_rooms))
#     eps = 1e-2
#     for i in range(num_rooms):
#         e_i = np.zeros(num_rooms)
#         e_i[i] = eps
#         # A_k[:, i] = (rk2_step(x_k + e_i, t_k, dt) - rk2_step(x_k - e_i, t_k, dt)) / (2 * eps)
#         A_k[:, i] = (x_k + e_i + dt*simulated_data.model.derivative(t_k, x_k + e_i) - (x_k - e_i + dt*simulated_data.model.derivative(t_k, x_k - e_i))) / (2 * eps)
    
#     P_pred = A_k @ P @ A_k.T + Q  # Predicted covariance

#     # Measurement Update Step
#     z_k = y[k]  # Actual measurement at time k
#     z_pred = simulated_data.measurement_matrix @ x_pred  # Predicted measurement
#     measurement_residual = z_k - z_pred

#     H = simulated_data.measurement_matrix.reshape(1, -1)
#     S = H @ P_pred @ H.T + R  # Residual covariance
#     K = P_pred @ H.T @ np.linalg.inv(S)  # Kalman gain

#     x[:, k] = x_pred + K[:, 0] * measurement_residual  # Updated state estimate
#     P = (np.eye(num_rooms) - K @ H) @ P_pred  # Updated covariance estimate

# with open(OUTPUT_DIR / "ekf_estimation.pkl", "wb") as f:
#     pkl_obj = {
#         "t": simulated_data.solution.t,
#         "estimated_temperature": simulated_data.measurement_matrix @ x,
#         "measured_temperature": y,
#         "x": x,
#     }
#     pkl.dump(pkl_obj, f)

with open(OUTPUT_DIR / "ekf_estimation.pkl", "rb") as f:
    pkl_obj = pkl.load(f)
    t = pkl_obj["t"]
    estimated_temperature = pkl_obj["estimated_temperature"]
    measured_temperature = pkl_obj["measured_temperature"]
    x = pkl_obj["x"]

plt.figure(figsize=(10, 6))

plt.plot(simulated_data.solution.t[0:-1:step_size], y[0:-1:step_size], label="Measured Temperature", linewidth=2)
plt.plot(simulated_data.solution.t[0:-1:step_size], (simulated_data.measurement_matrix@x)[0:-1:step_size], linestyle="--",label="Estimated Temperature", linewidth=2)

measured_room_idx = simulated_data.measurement_matrix.argmax()  # Index of the room being measured
plt.title(f"Extended Kalman Filter State Estimation over Measured Room {measured_room_idx + 1}")
plt.xlabel("Time (seconds)")
plt.ylabel("Temperature (°F)")

plt.legend()
plt.tight_layout()
# plt.show()
plt.savefig(OUTPUT_DIR / "ekf_estimation.svg")

"""Plotting the comparison of the true temperature and the estimated temperature for each state variable (temperature in each room)"""
for i in range(num_rooms):
    plt.figure(figsize=(10, 6))
    plt.plot(simulated_data.solution.t[0:-1:step_size], simulated_data.solution.y[i][0:-1:step_size], label=f"True Temperature (Room {i + 1})", linewidth=2)
    plt.plot(simulated_data.solution.t[0:-1:step_size], x[i][0:-1:step_size], label=f"Estimated Temperature (Room {i + 1})", linestyle="--", linewidth=2)
    plt.title(f"Extended Kalman Filter State Estimation for Room {i + 1}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Temperature (°F)")
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(OUTPUT_DIR / f"ekf_estimation_room_{i + 1}.svg")