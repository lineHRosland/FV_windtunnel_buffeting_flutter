#%%
import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append(r"C:\Users\oiseth\Documents\GitHub\w3tp")
import w3t

Name = 'LN'

# Quasi-steady motion for force coefficient tests
dt = 1/1000  # Sampling rate 1000 Hz
t = np.arange(0, 140 + dt, dt)
t1 = 10  # Starting time
T = 60  # Period of the motion
amp = 8  # Amplitude of the motion
R = amp * np.sin(2 * np.pi * (t - t1) / T)
TS = np.column_stack((np.zeros_like(R), np.zeros_like(R), R * 2 * np.pi / 360))
w3t.matrix2acs(t, TS, f"{Name}_QuasiSteady{round(amp)}.txt")

#%%

# Alternative static test
k = 5  # Sharpness of the rectangle
T = 120  # Length of motion
Ts = 10  # Start point of motion
a = 1 / T * 8  # Linear motion slope
amp = 8
R1 = w3t.rectsqueeze(t, Ts + 0 * T / 8, Ts + 1 * T / 8, k)
R2 = w3t.rectsqueeze(t, Ts + 1 * T / 8, Ts + 3 * T / 8, k)
R3 = w3t.rectsqueeze(t, Ts + 3 * T / 8, Ts + 5 * T / 8, k)
R4 = w3t.rectsqueeze(t, Ts + 5 * T / 8, Ts + 7 * T / 8, k)
R5 = w3t.rectsqueeze(t, Ts + 7 * T / 8, Ts + 8 * T / 8, k)
R = (R1 * (a * t + (1 - a * (Ts + 1 * T / 8))) +
     R2 * (-a * t + (-1 + a * (Ts + 3 * T / 8))) +
     R3 * (a * t + (1 - a * (Ts + 5 * T / 8))) +
     R4 * (-a * t + (-1 + a * (Ts + 7 * T / 8))) +
     R5 * (a * t + (0 - a * (Ts + 8 * T / 8))))
TS = np.column_stack((np.zeros_like(R), np.zeros_like(R), R * 2 * np.pi / 360 * 8))

w3t.matrix2acs(t, TS, f"{Name}_LinearQuasiSteady{round(amp)}.txt")

#save_motion_data(t, TS, f"{Name}_LinearQuasiSteady{round(amp)}.txt")
#%%
# Standard forced vibration tests
t = np.arange(0, 384, dt)
f_values = [0.5, 0.8, 1.1, 1.4, 1.7, 2.0]
NT = 20  # Number of periods of each test
T_ZERO = 1  # Time with zero motion between tests
A = np.zeros(int(1 / dt * T_ZERO))  # Initial standstill

for f in f_values:
    ts = np.arange(0, NT * 1 / f + 2, dt)
    R1 = w3t.rectsqueeze(ts, ts[0] + 1, ts[-1] - 1, 5)
    R2 = w3t.rectsqueeze(ts, ts[0] + 0.25, ts[-1] - 0.25, 20)
    A = np.concatenate([A, R1 * R2 * np.sin(2 * np.pi * f * ts), np.zeros(int(1 / dt * T_ZERO))])

A = np.concatenate([np.zeros(int(1 / dt * 5)), A, np.zeros(int(1 / dt * 5))])
t = t[:len(A)]

motions = {
    "FVHorizontal": 20,
    "FVVertical_20": 20,
    "FVVertical_10": 10,
    "FVTorsion_2": 2,
    "FVTorsion_1": 1
}

for key, amp in motions.items():
    if "Horizontal" in key:
        Motion = np.column_stack((A * amp, np.zeros_like(A), np.zeros_like(A)))
        w3t.matrix2acs(t, Motion, f"{Name}_FV_Horizontal{round(amp)}.txt")
    elif "Vertical" in key:
        Motion = np.column_stack((np.zeros_like(A), A * amp, np.zeros_like(A)))
        w3t.matrix2acs(t, Motion, f"{Name}_FV_Vertical{round(amp)}.txt")
    elif "Torsion" in key:
        Motion = np.column_stack((np.zeros_like(A), np.zeros_like(A), A * amp / 360 * 2 * np.pi))
        w3t.matrix2acs(t, Motion, f"{Name}_FV_Horizontal{round(amp)}.txt")

plt.show()