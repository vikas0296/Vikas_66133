import numpy as np
from scipy import linalg
import new_crack
import sys
import uniform_mesh
import enrichment_functions
from scipy import linalg

def Heaviside_enrichment(Nodes_L, Nodes_U, GaussPoint_1to4, D_plane_stress):
    storeU=[]
    storeL=[]
    for i in Nodes_U:
        # print("upper", i)
        for points in GaussPoint_1to4:
            # print(points)
            xi_1 = points[0]
            xi_2 = points[1]

            B_std_U, dN_en, N, dN, jacobi = enrichment_functions.classic_B_matric(i, D_plane_stress,
                                                                                            xi_1, xi_2, H2=1)

            B_enriched_U = np.array([[dN_en[0, 0], 0, dN_en[0, 1], 0, dN_en[0, 2], 0, dN_en[0, 3], 0],
                                     [0, dN_en[1, 0], 0, dN_en[1, 1], 0, dN_en[1, 2], 0, dN_en[1, 3]],
                                     [dN_en[1, 0], dN_en[0, 0], dN_en[1, 1], dN_en[0, 1], dN_en[1, 2], dN_en[0, 2],
                                      dN_en[1, 3], dN_en[0, 3]]])

            # print("upper", B_enriched_U.shape)
            B_all_U = np.concatenate((B_std_U, B_enriched_U), axis=1)
            Bt_D_U = np.matmul(B_all_U.T, D_plane_stress)
            Bt_D_B_U = (np.matmul(Bt_D_U, B_all_U)) * linalg.det(jacobi)
            K_mixed_U = Bt_D_B_U
            # print("upper", K_mixed_U[0:5,0:5])
            storeU.append(K_mixed_U)

    ZeroU = np.zeros([16,16])
    for z in storeU:
        ZeroU += z
    # print(ZeroU[0:10,0:10])

    for k in Nodes_L:
        # print("lower", k)
        for points in GaussPoint_1to4:
            # print(points)
            xi_1 = points[0]
            xi_2 = points[1]

            B_std_L, dN_en, N, dN, jacobi = enrichment_functions.classic_B_matric(k, D_plane_stress,
                                                                                            xi_1, xi_2, H2=-1)

            B_enriched_L = np.array([[dN_en[0, 0], 0, dN_en[0, 1], 0, dN_en[0, 2], 0, dN_en[0, 3], 0],
                                     [0, dN_en[1, 0], 0, dN_en[1, 1], 0, dN_en[1, 2], 0, dN_en[1, 3]],
                                     [dN_en[1, 0], dN_en[0, 0], dN_en[1, 1], dN_en[0, 1], dN_en[1, 2], dN_en[0, 2],
                                      dN_en[1, 3], dN_en[0, 3]]])

            B_all_L = np.concatenate((B_std_L, B_enriched_L), axis=1)
            Bt_D_L = np.matmul(B_all_L.T, D_plane_stress)
            Bt_D_B_L = (np.matmul(Bt_D_L, B_all_L)) * linalg.det(jacobi)
            K_mixed_L = Bt_D_B_L
            # print("upper", K_mixed_L[0:5,0:5])
            storeL.append(K_mixed_L)

    ZeroL = np.zeros([16,16])
    for j in storeL:
        ZeroL += j
    # print(ZeroL[10:20,10:16])

    UL = np.add(ZeroU, ZeroL)
    return UL

