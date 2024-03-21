# Optimal Estimation - HW3 - Battery Equivalent Circuit Model Design

import numpy as np
import numpy.linalg as la
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt

def ss_kalman_filter(F, G, H, Q, R, x_0, P_0, u, y):
    '''
    Temp
    '''
    x_hist = np.zeros((len(u), 2))
    x_hist[0, :] = np.atleast_2d(x_0).T
    x_km1_km1 = x_0

    # NEED TO ACTUALLY SOLVE FOR P_INF INSTEAD OF TAKING VALUE FROM OTHER RUN

    # P_inf = scipy.linalg.solve_discrete_are(F, G, Q, R)
    P_inf = np.array([[573.56642085, 14.80757623], [14.80757623, 0.77469338]])
    S_inf = H @ P_inf @ H.T + R
    K_inf = P_inf @ H.T * S_inf**-1

    for i, (u_k, y_k) in enumerate(zip(u, y)):

        if i == 0:
            u_km1 = u_k
        else:
            # Prediction Step
            x_k_km1 = F @ x_km1_km1 + G * u_km1
            y_k_est = H @ x_k_km1

            # Correction Step
            y_diff = y_k - y_k_est
            x_k_k = x_k_km1 + K_inf * y_diff

            # Saving and reseting values
            x_hist[i, :] = np.atleast_2d(x_k_k).T
            u_km1 = u_k
            x_km1_km1 = x_k_k

    return x_hist

def s_kalman_filter(F, G, H, Q, R, x_0, P_0, u, y):
    '''
    Temp
    '''
    x_hist = np.zeros((len(u), 2))
    x_hist[0, :] = np.atleast_2d(x_0).T
    P_hist = [0] * len(u)
    x_km1_km1 = x_0
    P_km1_km1 = P_0

    for i, (u_k, y_k) in enumerate(zip(u, y)):

        if i == 0:
            u_km1 = u_k
            P_hist[i] = P_0
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
            P_k_k = P_k_km1 - K_k @ S_k * K_k.T

            # Saving and reseting values
            x_hist[i, :] = np.atleast_2d(x_k_k).T
            P_hist[i] = P_k_k
            u_km1 = u_k
            x_km1_km1 = x_k_k
            P_km1_km1 = P_k_k

    return x_hist, P_hist

def covariance_intersection_kalman_filter(F, G, H, Q, R, x_0, P_0, u, y):
    '''
    Temp
    '''
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
            find_omega_opt = lambda omega: np.trace(la.inv(omega * la.inv(P_k_km1) \
                                                            + (1 - omega) * H.T * R**-1 @ H))
            omega_opt_res = sp.optimize.minimize_scalar(find_omega_opt, 0.5, bounds=(0, 1))
            omega_opt = omega_opt_res.x
            # omega_opt = 0.5
            P_k_k = np.linalg.inv(omega_opt * la.inv(P_k_km1) \
                                  + (1 - omega_opt) * H.T * R**-1 @ H)
            K_k = (1 - omega_opt) * P_k_k @ H.T * R**-1
            x_k_k = x_k_km1 + K_k * y_diff

            # Saving and reseting values
            x_hist[i, :] = np.atleast_2d(x_k_k).T
            u_km1 = u_k
            x_km1_km1 = x_k_k
            P_km1_km1 = P_k_k

    return x_hist

def s_kalman_smoother(F, G, Q, u, x_hist, P):
    '''
    Temp
    '''
    x_hist_smoothed = np.zeros((len(u), 2))

    for i, (u_k, x_k, P_k) in enumerate(zip(u[::-1], x_hist[::-1], P[::-1])):
        if i == 0:
            x_kp1_N = np.atleast_2d(x_k).T
        else:
            x_k = np.atleast_2d(x_k)
            x_k_diff = x_kp1_N - F @ x_k.T - G * u_k
            K_S_k = P_k @ F.T @ np.linalg.inv(F @ P_k @ F.T + Q)
            x_k_k = x_k.T + K_S_k @ x_k_diff
            x_hist_smoothed[i, :] = np.atleast_2d(x_k_k).T
            # P update

            x_kp1_N = x_k.T
    
    return x_hist_smoothed

def main():
    '''
    Temp
    '''
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

    # SS_KF_hist = ss_kalman_filter(F, G, H, Q, R, x_0, P_0, u, y)
    S_KF_hist = s_kalman_filter(F, G, H, Q, R, x_0, P_0, u, y)
    # CI_KF_hist = covariance_intersection_kalman_filter(F, G, H, Q, R, x_0, P_0, u, y)
    S_KS_hist = s_kalman_smoother(F, G, Q, u, S_KF_hist[0], S_KF_hist[1])

    plt.plot(time, S_KF_hist[0][:, 0])
    plt.plot(time, S_KS_hist[:, 0][::-1])
    plt.plot(time, f_k_act)
    # plt.plot(time, CI_KF_hist[:, 1])
    # plt.plot(time, b_k_act)
    plt.show()

    return 1

if __name__=='__main__':
    main()