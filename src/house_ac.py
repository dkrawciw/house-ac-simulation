from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.integrate import solve_ivp

SECONDS_PER_DAY = 24 * 60 * 60
TemperatureFunction = Callable[[float | np.ndarray], float | np.ndarray]


def base_outside_temperature(t: float | np.ndarray) -> float | np.ndarray:
    """Return a smooth day/night outside temperature cycle in degrees F."""
    mean_temp = 90.0
    amplitude = 20.0
    phase_shift = 10 * 60 * 60

    return mean_temp + amplitude * np.sin(
        2 * np.pi * (t - phase_shift) / SECONDS_PER_DAY
    )


def build_noisy_outside_temperature(
    noise_std: float = 0.0,
    num_samples: int = 2000,
    rng: np.random.Generator | None = None,
) -> TemperatureFunction:
    """Build an interpolated outside-temperature function with fixed noise."""
    if rng is None:
        rng = np.random.default_rng()

    sample_times = np.linspace(0.0, SECONDS_PER_DAY, num_samples)
    sample_temperatures = base_outside_temperature(sample_times)

    if noise_std > 0.0:
        sample_temperatures = sample_temperatures + rng.normal(
            loc=0.0,
            scale=noise_std,
            size=sample_times.shape,
        )

    def outside_temperature(t: float | np.ndarray) -> float | np.ndarray:
        t_array = np.asarray(t, dtype=float)
        temperatures = np.interp(t_array, sample_times, sample_temperatures)

        if np.isscalar(t) or t_array.ndim == 0:
            return float(temperatures)
        return temperatures

    return outside_temperature


@dataclass(frozen=True)
class HouseACModel:
    """Continuous-time temperature model for connected rooms with AC input."""

    wall_loss: np.ndarray
    ac_power: float
    ac_activation: np.ndarray
    cool_air_temp: float
    room_coupling: np.ndarray
    outside_temperature: TemperatureFunction = base_outside_temperature

    @property
    def num_rooms(self) -> int:
        return len(self.wall_loss)

    def derivative(self, t: float, temperatures: np.ndarray) -> np.ndarray:
        outside_temp = self.outside_temperature(t)
        temperatures = np.asarray(temperatures, dtype=float)

        heat_loss = self.wall_loss * (outside_temp - temperatures) * temperatures
        ac_effect = (
            self.ac_power
            * self.ac_activation
            * temperatures
            * (self.cool_air_temp - temperatures)
        )

        room_heat_transfer = np.zeros(self.num_rooms)
        for i in range(self.num_rooms):
            room_heat_transfer[i] = np.sum(
                self.room_coupling[i] * temperatures[i] * (temperatures - temperatures[i])
            )

        return room_heat_transfer + ac_effect + heat_loss

    def solve(
        self,
        initial_temperatures: np.ndarray,
        t_eval: np.ndarray,
        method: str = "BDF",
    ) -> Any:
        return solve_ivp(
            self.derivative,
            (float(t_eval[0]), float(t_eval[-1])),
            initial_temperatures,
            t_eval=t_eval,
            method=method,
        )


@dataclass(frozen=True)
class SimulationResult:
    model: HouseACModel
    solution: Any
    measurement_matrix: np.ndarray
    measurements: np.ndarray


def build_random_symmetric_coupling(
    num_rooms: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    coupling = rng.random((num_rooms, num_rooms)) / 2
    return np.abs((coupling + coupling.T) / 2)


def create_default_model(
    rng: np.random.Generator | None = None,
    outside_temperature: TemperatureFunction | None = None,
) -> HouseACModel:
    if rng is None:
        rng = np.random.default_rng()
    if outside_temperature is None:
        outside_temperature = build_noisy_outside_temperature(noise_std=1.0, rng=rng)

    num_rooms = 3
    return HouseACModel(
        wall_loss=np.array([0.5, 0.01, 0.01]),
        ac_power=0.8,
        ac_activation=np.array([1, 0, 1]),
        cool_air_temp=60.0,
        room_coupling=build_random_symmetric_coupling(num_rooms, rng=rng),
        outside_temperature=outside_temperature,
    )


def simulate_default(seed: int = 0, num_time_points: int = 5000000) -> SimulationResult:
    rng = np.random.default_rng(seed)
    model = create_default_model(rng=rng)
    initial_temperatures = rng.normal(70, 5, model.num_rooms)
    t_eval = np.linspace(0, SECONDS_PER_DAY, num_time_points)
    solution = model.solve(initial_temperatures, t_eval)

    measurement_matrix = np.array([1, 0, 0])
    measurements = make_measurements(
        solution.y,
        measurement_matrix,
        noise_std=0.5,
        rng=rng,
    )

    return SimulationResult(
        model=model,
        solution=solution,
        measurement_matrix=measurement_matrix,
        measurements=measurements,
    )


def make_measurements(
    states: np.ndarray,
    measurement_matrix: np.ndarray,
    noise_std: float = 0.5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    return measurement_matrix @ states + noise_std * rng.normal(size=states.shape[1])


def configure_plot_style() -> None:
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


def plot_ac_activation(model: HouseACModel, t_eval: np.ndarray, output_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    for i, activation in enumerate(model.ac_activation):
        linestyle = "--" if i == 2 else "-"
        linewidth = 7 if i == 2 else 5
        plt.plot(
            t_eval,
            activation * np.ones_like(t_eval),
            linestyle,
            label=f"AC in Room {i + 1}",
            linewidth=linewidth,
        )

    plt.xlabel("Time (s)")
    plt.ylabel("AC Activation (1=On, 0=Off)")
    plt.title("AC Activation in Each Room (Input)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_temperature_evolution(result: SimulationResult, output_path: Path) -> None:
    solution = result.solution

    plt.figure(figsize=(10, 6))
    for i in range(result.model.num_rooms):
        plt.plot(solution.t, solution.y[i], label=f"Room {i + 1}", linewidth=2)

    plt.plot(
        solution.t,
        result.model.outside_temperature(solution.t),
        label="Outside Temperature",
        linestyle="--",
        color="black",
        linewidth=2,
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (°F)")
    plt.title("Temperature Evolution in Connected Rooms")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_measurements(result: SimulationResult, output_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(
        result.solution.t,
        result.measurements,
        label="Measured Temperature (Room 1)",
        linewidth=3,
    )
    plt.plot(
        result.solution.t,
        result.solution.y[0],
        label="True Temperature (Room 1)",
        linestyle="--",
        linewidth=3,
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (°F)")
    plt.title("Measured vs True Temperature in Room 1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def write_default_plots(
    result: SimulationResult,
    output_dir: Path | str | None = None,
) -> None:
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "paper"
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    configure_plot_style()
    plot_ac_activation(
        result.model,
        result.solution.t,
        output_dir / "ac_activation.svg",
    )
    plot_temperature_evolution(result, output_dir / "ac_simulation.svg")
    plot_measurements(result, output_dir / "ac_measurement.svg")


def main() -> None:
    result = simulate_default()
    write_default_plots(result)


if __name__ == "__main__":
    main()
