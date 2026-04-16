import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import seaborn as sns
from temp_diffeq import dTdt, outside_temperature, SECONDS_PER_DAY
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "paper"
OUTPUT_DIR.mkdir(exist_ok=True)

"""Plot setup"""
sns.set_style("whitegrid")
sns.set_color_codes(palette="colorblind")

plt.rcParams.update({
	"text.usetex": False,  # keep False to avoid requiring a LaTeX installation
	"mathtext.fontset": "cm",  # Computer Modern (LaTeX-like)
	"font.family": "serif",
	"font.serif": ["Computer Modern Roman", "DejaVu Serif"],
    "axes.labelsize": 14,      # increase axis label size
    "axes.titlesize": 16,
    "xtick.labelsize": 14,     # increase tick / bin label size
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
})

"""Parameters for the simulation"""
alpha = 0.8                              # AC cooling power per unit time (°F/s)
cool_air_temp = 60.0                         # Temperature of the cool air from the AC (°F)
num_rooms = 3                               # Number of Rooms in this house
# k_w = 1e-3 * np.ones(num_rooms)             # Heat Loss outside for each room (°F/s per °F difference)
k_w = np.array([0.5, 0.01, 0.01])             # Heat Loss outside for each room (°F/s per °F difference)
B = np.array([1, 0, 1])                     # AC activation for each room (1 if on, 0 if off)
K = np.random.rand(num_rooms, num_rooms)/2    # Thermal coupling between rooms (°F/s per °F difference)
K = abs((K + K.T) / 2)                      # Make it symmetric an positive


T_0 = np.random.normal(70, 5, num_rooms)

t_eval = np.linspace(0, SECONDS_PER_DAY, 500)
dTdt_wrapper = lambda t, T: dTdt(t, T, num_rooms, k_w, alpha, B, cool_air_temp, K)
sol = solve_ivp(dTdt_wrapper, (0, SECONDS_PER_DAY), T_0, t_eval=t_eval, method='BDF')

"""Plotting the Room Input"""
plt.figure(figsize=(10, 6))
plt.plot(t_eval, B[0] * np.ones_like(t_eval), label="AC in Room 1", linewidth=5)
plt.plot(t_eval, B[1] * np.ones_like(t_eval), label="AC in Room 2", linewidth=5)
plt.plot(t_eval, B[2] * np.ones_like(t_eval),"--", label="AC in Room 3", linewidth=7)
plt.xlabel("Time (s)")
plt.ylabel("AC Activation (1=On, 0=Off)")
plt.title("AC Activation in Each Room (Input)")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ac_activation.svg")

"""Plotting the Results"""
plt.figure(figsize=(10, 6))
for i in range(num_rooms):
    plt.plot(sol.t, sol.y[i], label=f"Room {i+1}", linewidth=2)

plt.plot(sol.t, outside_temperature(sol.t), label="Outside Temperature", linestyle='--', color='black', linewidth=2)

plt.xlabel("Time (s)")
plt.ylabel("Temperature (°F)")
plt.title("Temperature Evolution in Connected Rooms")
plt.legend()
# plt.show()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ac_simulation.svg")

"""Plotting Measured Flow"""
measurement_noise = 0.5 * np.random.normal(size=sol.t.shape)
C = np.array([1,0,0])
y = C @ sol.y + measurement_noise

plt.figure(figsize=(10, 6))
plt.plot(sol.t, y, label="Measured Temperature (Room 1)", linewidth=3)
plt.plot(sol.t, sol.y[0], label="True Temperature (Room 1)", linestyle='--', linewidth=3)
plt.xlabel("Time (s)")
plt.ylabel("Temperature (°F)")
plt.title("Measured vs True Temperature in Room 1")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ac_measurement.svg")