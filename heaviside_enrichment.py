import numpy as np
from scipy import linalg
import uniform_mesh
import enrichment_functions
from scipy import linalg

def Heaviside_enrichment(Nodes, GaussPoint_1to4, D_plane_stress, cracks):
    storeU=[]
    #looping through sub_elements/ sub-domains
    for i in Nodes:
        A,B,C,D = i[0], i[1], i[2], i[3]
        # calling step_function to calculate the value of enrichment function (1 or 0 or -1)
        H1 = enrichment_functions.step_function(A, cracks)
        H2 = enrichment_functions.step_function(B, cracks)
        H3 = enrichment_functions.step_function(C, cracks)
        H4 = enrichment_functions.step_function(D, cracks)

        for points in GaussPoint_1to4:
            #Defining GaussPoints
            xi_1 = points[0]
            xi_2 = points[1]
            #extracting standard B-matrices, jacobi and enriched  B-matrix and values of the shape function
            B_std_U, dN_en, N, dN, jacobi, NaN = enrichment_functions.classic_B_matric(i, D_plane_stress,xi_1, xi_2, H1, H2, H3, H4)

            #forming a enriched B_matrix for the heaviside enriched elements
            B_enriched_U = np.array([[dN_en[0, 0], 0, dN_en[0, 1], 0, dN_en[0, 2], 0, dN_en[0, 3], 0],
                                     [0, dN_en[1, 0], 0, dN_en[1, 1], 0, dN_en[1, 2], 0, dN_en[1, 3]],
                                     [dN_en[1, 0], dN_en[0, 0], dN_en[1, 1], dN_en[0, 1], dN_en[1, 2], dN_en[0, 2],
                                      dN_en[1, 3], dN_en[0, 3]]])

            '''
            Steps to calculate K-element
            '''
            # integration(B.T * D * B * ||Jacobian|| * dV)
            # Gauss_Quadrature weights = W1, W2 = 1
            # K_element = W1 * W2 * ΣΣ(B.T * D * B * ||Jacobian||)

            B_all_U = np.concatenate((B_std_U, B_enriched_U), axis=1)
            Bt_D_U = np.matmul(B_all_U.T, D_plane_stress)
            Bt_D_B_U = (np.matmul(Bt_D_U, B_all_U)) * linalg.det(jacobi)
            K_mixed_U = Bt_D_B_U
            KP = np.round(K_mixed_U,3)
            storeU.append(K_mixed_U)

    ZeroU = np.zeros([16,16])

    for z in storeU:
        ZeroU += z

    return np.round(ZeroU,3)



















