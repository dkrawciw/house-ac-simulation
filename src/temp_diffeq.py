import numpy as np

SECONDS_PER_DAY = 24 * 60 * 60

def outside_temperature(t: float) -> float:
    # Simple day/night cycle: coolest around 4 AM, warmest around 4 PM.
    mean_temp = 90.0
    amplitude = 20.0
    phase_shift = 10 * 60 * 60
    return mean_temp + amplitude * np.sin(2 * np.pi * (t - phase_shift) / SECONDS_PER_DAY)

def dTdt(t: float, T: np.ndarray, num_rooms: int, k_w: np.ndarray, alpha: float, B: np.ndarray, cool_air_temp: float, K: np.ndarray) -> np.ndarray:
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