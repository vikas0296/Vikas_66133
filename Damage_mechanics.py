import numpy as np
from scipy import linalg
import math
import operator
import matplotlib.pyplot as plt


def mechanics_damage(sigma_F1, sigma_F2, nu, alpha):

    beta = 1
    alpha = 1
    M = 1
    print(len(sigma_F1), len(sigma_F2))

    for i, j in zip(sigma_F1, sigma_F2):

        sigma1_1 = 0.5*(i[0] + i[1]) + np.sqrt(0.25*(i[0]-i[1])**2 + i[2]**2)
        sigma2_1 = 0.5*(i[0] + i[1]) - np.sqrt(0.25*(i[0]-i[1])**2 + i[2]**2)
        sigma12_1 = 0.5*(i[0]-i[1])

        Sigma_equivalent_1 = np.sqrt(sigma1_1**2 - sigma1_1*sigma2_1 + sigma2_1**2 + 3 * sigma12_1**2)

        Hydrostatic_stress_1 = (sigma1_1 + sigma2_1) * 0.5

        sigma1_2 = 0.5*(j[0] + j[1]) + np.sqrt(0.25*(j[0]-j[1])**2 + j[2]**2)
        sigma2_2 = 0.5*(j[0] + j[1]) - np.sqrt(0.25*(j[0]-j[1])**2 + j[2]**2)
        sigma12_2 = 0.5*(j[0]-j[1])

        Sigma_equivalent_2 = np.sqrt(sigma1_2**2 - sigma1_2*sigma2_2 + sigma2_2**2 + 3 * sigma12_2**2)

        Hydrostatic_stress_2 = (sigma1_2 + sigma2_2) * 0.5

    T = max_hydrostatic / max_equivalent

    Rv = (2/3)*(1+nu) + 3(1-2*nu)* T**2
    del_sigma = (max_equivalent - min_equivalent)
    U = (alpha*W)
    dWdN_N = M * (1 - (1e(-alpha))) * (del_sigma**beta) * (1eU) * (Rv**(0.5*beta))

    delta_W_N = delta_N * (dWdN_N)
    W_N_dn = W_N + delta_W_N


        # print(np.round(sigma1,3), np.round(sigma2,3), np.round(sigma12,3))
        # print(np.round(sigmaV,3))