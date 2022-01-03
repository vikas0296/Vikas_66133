import numpy as np
import heaviside_enrichment
import class_crack
import math as m
import uniform_mesh
import enrichment_functions
import plots
from scipy import linalg

def mixed_enrichment(Nodes_U, Nodes_L, r, theta, alpha, GaussPoint_1to4, D_plane_stress):

    F1, F2, F3, F4, dF = enrichment_functions.asymptotic_functions(r, theta, alpha)

    storeU=[]
    storeL=[]
    for i in Nodes_U:
        for points in GaussPoint_1to4:
            xi_1 = points[0]
            xi_2 = points[1]
            B_std_U, B_heavy_U, N, dN, jacobi = enrichment_functions.classic_B_matric(i, D_plane_stress,
                                                                                            xi_1, xi_2, H2 = 1)

            heavy_1 = enrichment_functions.heaviside_function1(B_heavy_U)
            tip_2 = enrichment_functions.tip_enrichment_func_N2(F1, F2, F3, F4, dN, dF, N)
            tip_3 = enrichment_functions.tip_enrichment_func_N3(F1, F2, F3, F4, dN, dF, N)
            heavy_4 = enrichment_functions.heaviside_function4(B_heavy_U)

            B_all_U = np.concatenate((B_std_U, heavy_1, heavy_4, tip_2, tip_3), axis=1)

            Bt_D_U = np.matmul(B_all_U.T, D_plane_stress)
            Bt_D_B_U = (np.matmul(Bt_D_U, B_all_U)) * linalg.det(jacobi)
            K_mixed_U = Bt_D_B_U
            Q = Bt_D_B_U.shape
            ZeroU = np.zeros([Q[1],Q[1]])
            storeU.append(K_mixed_U)

    for k in storeU:
        ZeroU += k
    # print(ZeroU[0:5,0:5])

    for j in Nodes_L:
        for points in GaussPoint_1to4:
            xi_1 = points[0]
            xi_2 = points[1]
            B_std_L, B_heavy_L, N, dN, jacobi = enrichment_functions.classic_B_matric(j, D_plane_stress,
                                                                                            xi_1, xi_2, H2 = 0)

            heavy_1L = enrichment_functions.heaviside_function1(B_heavy_U)
            tip_2L = enrichment_functions.tip_enrichment_func_N2(F1, F2, F3, F4, dN, dF, N)
            tip_3L = enrichment_functions.tip_enrichment_func_N3(F1, F2, F3, F4, dN, dF, N)
            heavy_4L = enrichment_functions.heaviside_function4(B_heavy_U)

            B_all_L = np.concatenate((B_std_L, heavy_1L, heavy_4L, tip_2L, tip_3L), axis=1)

            Bt_D_L = np.matmul(B_all_L.T, D_plane_stress)
            Bt_D_B_L = (np.matmul(Bt_D_L, B_all_L)) * linalg.det(jacobi)
            K_mixed_L = Bt_D_B_L
            R = Bt_D_B_L.shape
            ZeroL = np.zeros([R[1],R[1]])
            storeL.append(K_mixed_L)

    for l in storeL:
        ZeroL += l
    # print(ZeroL[0:5,0:5])

    UL = np.add(ZeroL, ZeroU)
    return UL
