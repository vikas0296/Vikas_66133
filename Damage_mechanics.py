import numpy as np
from scipy import linalg
import math
import operator
import matplotlib.pyplot as plt


def mechanics_damage(sigma_F1, sigma_F2, G, c, l_c):

    a_init = 9
    a_final = 31.5
    beta = 3.68
    alpha = 6.1
    M = 3.3884e-15
    nu = 0.33
    delta_N = 10
    D_cpc = 0.15
    W = np.zeros([10000])
    N = 0
    W[N] = 0
    echo = np.zeros([len(sigma_F1)])
    fox = np.zeros([len(sigma_F1)])
    while N < 1000:
        print("NNNN", N)
        for i, j, l, k in zip(sigma_F1, sigma_F2, G, range(len(sigma_F1))):

            sigma_i = 1/3 * (i[0] + i[1])
            sigma_1i = (1/np.sqrt(2)) * np.sqrt((i[0]-i[1])**2 + i[1]**2 + i[0]**2 + 6*i[2]**2)
            T_no_normal_stress_i = sigma_i/sigma_1i
            max_equivalent_1 = np.sqrt(i[0]**2 - i[0]*i[1] + i[1]**2 + 3 * i[2]**2)

            sigma_j = 1/3 * (j[0] + j[1])
            sigma_jj = (1/np.sqrt(2)) * np.sqrt((j[0]-j[1])**2 + j[1]**2 + j[0]**2 + 6*j[2]**2)
            T_no_normal_stress_jj = sigma_j/sigma_jj
            max_equivalent_2 = np.sqrt(j[0]**2 - j[0]*j[1] + j[1]**2 + 3 * j[2]**2)


            sigma1_1 = 0.5*(i[0] + i[1]) + np.sqrt(0.25*(i[0]-i[1])**2 + i[2]**2)
            sigma2_1 = 0.5*(i[0] + i[1]) - np.sqrt(0.25*(i[0]-i[1])**2 + i[2]**2)

            sigma1_2 = 0.5*(j[0] + j[1]) + np.sqrt(0.25*(j[0]-j[1])**2 + j[2]**2)
            sigma2_2 = 0.5*(j[0] + j[1]) - np.sqrt(0.25*(j[0]-j[1])**2 + j[2]**2)

            max_eq = max(max_equivalent_1, max_equivalent_2)
            max_pincip = max(sigma1_1, sigma2_1, sigma1_2, sigma2_2)
            min_eq = min(max_equivalent_1, max_equivalent_2)
            T_max = max(T_no_normal_stress_i, T_no_normal_stress_jj)

            Rv = (2/3) * (1+nu) + 3*(1-2*nu)* T_max**2
            del_sigma = (max_eq - min_eq)
            # print("WN", W[N])
            omega = ((M * (1 - math.exp(-alpha))) * (del_sigma**beta) * (Rv**(0.5*beta))* math.exp(alpha*W[N]))/alpha
            # print(omega)
            delta_omega = delta_N * omega
            W[N+delta_N] = W[N] + delta_omega
            echo[k] = W[N+delta_N]
            # print(W[N+delta_N])
            mod = abs(np.sqrt((l[0]-c[0])**2 + (l[1]-c[1])**2))
            lamda = (1/(2*np.pi)**1.5 * (l_c**3)) * (math.exp(-(mod)**2 / l_c**2))
            fox[k] = lamda

        # print(len(fox), len(echo))

        B = 0
        C = 0
        for i,j in zip(fox, echo):
            B += i*j
            C += i

        D_cp = B/C
        N = N+delta_N


        W2 = np.zeros([5000000])
        N = 0
        W2[N] = 0
        echo2 = np.zeros([len(sigma_F1)])
        fox2 = np.zeros([len(sigma_F1)])
        D_cp_c = 4e-3
        while D_cp < D_cp_c:
            for i, j, l, k in zip(sigma_F1, sigma_F2, G, range(len(sigma_F1))):

                sigma_i = 1/3 * (i[0] + i[1])
                sigma_1i = (1/np.sqrt(2)) * np.sqrt((i[0]-i[1])**2 + i[1]**2 + i[0]**2 + 6*i[2]**2)
                max_equivalent_1 = np.sqrt(i[0]**2 - i[0]*i[1] + i[1]**2 + 3 * i[2]**2)

                sigma_j = 1/3 * (j[0] + j[1])
                sigma_jj = (1/np.sqrt(2)) * np.sqrt((j[0]-j[1])**2 + j[1]**2 + j[0]**2 + 6*j[2]**2)
                max_equivalent_2 = np.sqrt(j[0]**2 - j[0]*j[1] + j[1]**2 + 3 * j[2]**2)

                sigma1_1 = 0.5*(i[0] + i[1]) + np.sqrt(0.25*(i[0]-i[1])**2 + i[2]**2)
                sigma2_1 = 0.5*(i[0] + i[1]) - np.sqrt(0.25*(i[0]-i[1])**2 + i[2]**2)

                sigma1_2 = 0.5*(j[0] + j[1]) + np.sqrt(0.25*(j[0]-j[1])**2 + j[2]**2)
                sigma2_2 = 0.5*(j[0] + j[1]) - np.sqrt(0.25*(j[0]-j[1])**2 + j[2]**2)

                max_eq = max(max_equivalent_1, max_equivalent_2)
                max_pincip = max(sigma1_1, sigma2_1, sigma1_2, sigma2_2)
                min_eq = min(max_equivalent_1, max_equivalent_2)
                T_max = max_pincip / max_eq

                Rv = (2/3) * (1+nu) + 3*(1-2*nu)* T_max**2
                del_sigma = (max_eq - min_eq)
                omega2 = ((M * (1 - math.exp(-alpha))) * (del_sigma**beta) * (Rv**(0.5*beta))* math.exp(alpha*W2[N]))/alpha
                delta_omega2 = delta_N * omega2
                W2[N+delta_N] = W2[N] + delta_omega2
                echo2[k] = W2[N+delta_N]
                mod2 = abs(np.sqrt((l[0]-c[0])**2 + (l[1]-c[1])**2))
                lamda2 = (1/(2*np.pi)**1.5 * (l_c**3)) * (math.exp(-(mod2)**2 / l_c**2))
                fox2[k] = lamda2

            E = 0
            F = 0
            for i,j in zip(fox2, echo2):
                E += i*j
                F += i

            D_cp = E/F

            N = N+delta_N
            print(D_cp)
            print("N: ", N)
































































            # for i, j, l in zip(sigma_F1, sigma_F2, G):
            #     # print(W)
            #     sigma1_1 = 0.5*(i[0] + i[1]) + np.sqrt(0.25*(i[0]-i[1])**2 + i[2]**2)
            #     sigma2_1 = 0.5*(i[0] + i[1]) - np.sqrt(0.25*(i[0]-i[1])**2 + i[2]**2)
            #     sigma12_1 = abs(0.5*(sigma1_1-sigma2_1))

            #     sigma1_2 = 0.5*(j[0] + j[1]) + np.sqrt(0.25*(j[0]-j[1])**2 + j[2]**2)
            #     sigma2_2 = 0.5*(j[0] + j[1]) - np.sqrt(0.25*(j[0]-j[1])**2 + j[2]**2)
            #     sigma12_2 = abs(0.5*(sigma1_2-sigma2_2))

            #     Hydrostatic_stress_1 = (sigma1_1 + sigma2_1) * 0.5
            #     Hydrostatic_stress_2 = (sigma1_2 + sigma2_2) * 0.5

            #     max_sigma_1 = [sigma1_1, sigma1_2, sigma2_1, sigma2_2]
            #     Hydrostatic_stress = [Hydrostatic_stress_1, Hydrostatic_stress_2]

            #     max_equivalent = np.sqrt(sigma1_1**2 - sigma1_1*sigma2_1 + sigma2_1**2 + 3 * sigma12_1**2)
            #     min_equivalent = np.sqrt(sigma1_2**2 - sigma1_2*sigma2_2 + sigma2_2**2 + 3 * sigma12_2**2)

            #     T = max_sigma_1 / max_equivalent
            #     Rv = (2/3)*(1+nu) + 3*(1-2*nu)* T**2
            #     # # print(round(Rv,3))
            #     del_sigma = (max_equivalent - min_equivalent)
            #     # print("sigma",round(del_sigma,3))
            #     omega = ((M * (1 - math.exp(-alpha))) * (del_sigma**beta) * (Rv**(0.5*beta))* math.exp(alpha*W[N]))/alpha
            #     delta_omega2 = delta_N * omega
            #     W[N+delta_N] = W[N] + delta_omega
            #     echo2.append(W[N + delta_N])
            #     # print(delta)
            # mod = np.sqrt((l[0]-c[0])**2 + (l[1]-c[1])**2)
            # lamda = (1/(2*np.pi)**1.5 * (l_c**3)) * (math.exp(-(mod)**2 / l_c**2))
            # fox2.append(lamda)
            # B = 0
            # C = 0
            # for i,j in zip(fox2, echo2):
            #     B += i*j
            #     C += i

            # D_cp = B/C