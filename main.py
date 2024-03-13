# Optimal Estimation - HW3 - Battery Equivalent Circuit Model Design

import pandas as pd
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt

def kalman_filter(F, G, H, Q, R, x_0, P_0, u, y):

    x_hist = np.zeros((len(u), 2))
    x_hist[0, :] = np.atleast_2d(x_0).T
    x_km1_km1 = x_0
    P_km1_km1 = P_0

    for i, (u_k, y_k) in enumerate(zip(u, y)):

        if i == 0:
            u_km1 = u_k
        else:
            # Prediction Step
            x_k_km1 = F @ x_km1_km1 + G * u_km1
            P_k_km1 = F @ P_km1_km1 @ F.T + Q
            y_k_est = H @ x_k_km1

            # Correction Step
            y_diff = y_k - y_k_est
            S_k = H @ P_k_km1 @ H.T + R
            K_k = P_k_km1 @ H.T * S_k**-1
            x_k_k = x_k_km1 + K_k * y_diff
            P_k_k = P_k_km1 - K_k * S_k @ K_k.T

            # Saving and reseting values
            x_hist[i, :] = np.atleast_2d(x_k_k).T
            u_km1 = u_k
            x_km1_km1 = x_k_k
            P_km1_km1 = P_k_k

    return x_hist

def main():

    # Load Data
    data = np.loadtxt('fuel_data.csv', delimiter=',')
    time = data[:, 0]
    u = data[:, 1]
    y = data[:, 2]
    f_k_act = data[:, 3]
    b_k_act = data[:, 4]
    # Constants Definition
    dt = 0.5                                            # s
    A_line = 1                                          # cm^2
    A_tank = 150                                        # cm^2
    Q = np.diag((A_line**2 * dt**2 * 0.1**2, 0.1**2))   # [cm^6, cm^2/s^2] 
    R = 1                                               # cm
    # Initial Conditions
    x_0 = np.array([[3000], [0]])                       # [[cm^3], [cm^3]]
    P_0 = np.diag((10**2, 0.1**2))                      # [cm^6, cm^6] 
    # Matrix Definitions
    F = np.array([[1, A_line * dt], [0, 1]])
    G = np.array([[-A_line * dt], [0]])
    H = np.atleast_2d(np.array([A_tank**-1, 0]))

    S_KF_hist = kalman_filter(F, G, H, Q, R, x_0, P_0, u, y)

    plt.plot(time, S_KF_hist[:, 0])
    plt.plot(time, f_k_act)
    plt.show()

    return 1

if __name__=='__main__':
    main()