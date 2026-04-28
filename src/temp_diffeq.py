import numpy as np
from collections.abc import Callable

SECONDS_PER_DAY = 24 * 60 * 60

def base_outside_temperature(t: float | np.ndarray) -> float | np.ndarray:
    # Simple day/night cycle: coolest around 4 AM, warmest around 4 PM.
    mean_temp = 90.0
    amplitude = 20.0
    phase_shift = 10 * 60 * 60

    return mean_temp + amplitude * np.sin(2 * np.pi * (t - phase_shift) / SECONDS_PER_DAY)

def build_noisy_outside_temperature(
    noise_std: float = 0.0,
    num_samples: int = 2000,
    rng: np.random.Generator | None = None,
) -> Callable[[float | np.ndarray], float | np.ndarray]:
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

def outside_temperature(t: float | np.ndarray) -> float | np.ndarray:
    return base_outside_temperature(t)

def dTdt(t: float, T: np.ndarray, num_rooms: int, k_w: np.ndarray, alpha: float, B: np.ndarray, cool_air_temp: float, K: np.ndarray, outside_temperature: Callable[[float], float]) -> np.ndarray:
    T_out = outside_temperature(t)

    dT = np.zeros(num_rooms)
    Q_loss = np.zeros(num_rooms)
    room_temp = np.zeros(num_rooms)

    Q_loss = k_w * (T_out - T) * T
    ac_effect = alpha * B * T * (cool_air_temp - T)    

    for i in range(num_rooms):
        this_room = T[i]

        for j in range(num_rooms):
            other_room = T[j]
            rooms_coef = K[i,j]

            room_temp[i] += rooms_coef * this_room * (other_room - this_room)
        
    dT = room_temp + ac_effect + Q_loss

    return dT
