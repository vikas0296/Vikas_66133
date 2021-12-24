import numpy as np
import heaviside_enrichment
import class_crack
import math as m
import enrichment_functions
import plots
from scipy import linalg

def set_matrix(Nodes, H2, GaussPoint_1to4, D_plane_stress):
    final_matrix = np.zeros([16,16])
    for i in Nodes:
        A,B,C,D = heaviside_enrichment.Heaviside_enrichment(i, H2, GaussPoint_1to4, D_plane_stress)
        Element_matrix = np.zeros([16,16])
        Element_matrix[0:8,0:8] = A
        Element_matrix[0:8, 8:16] = B
        Element_matrix[8:16, 0:8] = C
        Element_matrix[8:16, 8:16] = D
        final_matrix += Element_matrix

    return final_matrix


def to_polar_in_radians(x,y):
    '''
    Convertion of cartesian to polar coordinates in radiance
    Parameters
    ----------
    x : x-coordinate of crack tip
    y : y-coordinate of crack tip
    Returns
    -------
    r and, theta in radiance
    '''
    r = np.sqrt(x**2 + y**2)
    theta = m.atan(y/x)
    return np.round(r, 3), np.round(theta, 3)




# def connectivity(A_matrix, K_matrix):
#     At_K = np.matmul (A_matrix.T, K_matrix)
#     At_K_A = np.matmul (At_K, A_matrix)
#     return At_K_A

# def Assignment_matrix_enrich(k,shape, Elements):
#     P,Q,R,S = 0,1,2,3
#     Zero_main = np.zeros([16,16])
#     Zero_dummy = np.zeros([16,16])
#     A_matrix = np.zeros((10, Elements*2))
#     I = np.eye(8)
#     A_matrix[0:8, 0:8] = I
#     if shape == 10:
#         for i in range(len(k)):

#             if i==0:
#                 A_matrix[8, 8] = 1
#                 A_matrix[9, 9] = 1
#                 # print("00000000000", i)
#                 Zero_1 = connectivity(A_matrix, k[i])
#                 Zero_main = Zero_main + Zero_1

#             if i==1:
#                 A_matrix[8:10, 8:10] = 0
#                 A_matrix[8, 10] = 1
#                 A_matrix[9, 11] = 1
#                 # print("00000000000", i)
#                 Zero_2 = connectivity(A_matrix, k[i])
#                 Zero_main = Zero_main + Zero_2

#             if i==2:
#                 A_matrix[8:12, 8:12] = 0
#                 A_matrix[8, 12] = 1
#                 A_matrix[9, 13] = 1
#                 # print("00000000000", i)
#                 Zero_3 = connectivity(A_matrix, k[i])
#                 Zero_main = Zero_main + Zero_3

#             if i==3:
#                 A_matrix[8:14, 8:14] = 0
#                 A_matrix[8, 14] = 1
#                 A_matrix[9, 15] = 1
#                 # print("00000000000", i)
#                 Zero_4 = connectivity(A_matrix, k[i])
#                 Zero_main = Zero_main + Zero_4
#         # print(Zero_main.shape)
#         # Assign = Zero_main
#         return Zero_main

#     elif shape == 8:
#         add_matrix = np.zeros([10, 10])
#         for i in range(len(k)):
#             add_matrix[0:8, 0:8] = k[i]
#             Zero_no_enrich = connectivity(A_matrix, add_matrix)
#             Zero_dummy = Zero_dummy + Zero_no_enrich

#         return Zero_dummy



