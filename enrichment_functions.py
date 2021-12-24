import numpy as np
from scipy import linalg
import math as m
'''
All the functions necessary for the tip enrichment and heaviside_enrichment have been put together
'''
def classic_B_matric(coordinates, D_plane_stress, xi_1, xi_2, H2):

      #================for tip enrichment,shape functions are necessary=================================
      N1 = 0.25* (1-xi_1) * (1-xi_2)
      N2 = 0.25* (1+xi_1) * (1+xi_2)
      N3 = 0.25* (1+xi_1) * (1+xi_2)
      N4 = 0.25* (1-xi_1) * (1+xi_2)

      N = np.array([N1, N2, N3, N4])
      #==================================================================================================
      dNdxi = (1/4) * np.array([[-(1-xi_2), 1-xi_2, 1+xi_2, -(1+xi_2)],
                                [-(1-xi_1), -(1+xi_1), 1+xi_1, 1-xi_1]])

      # convertion from local coordinate system to global coordinate system
      # 1. Calculate jacobian
      # 2. Take inverse of the Jacobian
      # 3. multiply with differentiated shape functions

      jacobi = np.matmul(dNdxi, coordinates)
      inverse_jacobi = linalg.inv(jacobi)
      dN = np.round(np.matmul(inverse_jacobi, dNdxi), 3)

    # B-matrix for classical DOFs (2 per node)
      B_std = np.array([[dN[0, 0], 0, dN[0, 1], 0, dN[0, 2], 0, dN[0, 3], 0],
                        [0, dN[1, 0], 0, dN[1, 1], 0, dN[1,2], 0, dN[1,3]],
                        [dN[1, 0], dN[0, 0], dN[1, 1], dN[0, 1], dN[1, 2], dN[0, 2], dN[1,3], dN[0, 3]]])

      H1 = 0
      dN_en = (H1-H2) * (1/4) * np.array([[dN[0, 0], dN[0, 1], dN[0, 2], dN[0, 3]],
                                          [dN[1, 0], dN[1, 1], dN[1, 2], dN[1, 3]]])


      return B_std, dN_en, N, dN, jacobi

def asymptotic_functions(r, theta, alpha):
    '''
        #tip enrichment terms
        Fα (r, θ) = [F1, F2, F3, F4]
    '''
    F1 = np.sqrt(r) * np.sin(theta/2)
    F2 = np.sqrt(r) * np.sin(theta/2)
    F3 = np.sqrt(r) * np.sin(theta/2) * np.sin(theta)
    F4 = np.sqrt(r) * np.cos(theta/2) * np.sin(theta)

    '''
    #transformation of Fα (r, θ) between the polar and Cartesian
    coordinates in a local Cartesian coordinate system (x1, x2)
    α = 1,2,3,4
    ∂Fα/∂x1 = ∂Fα/∂r * ∂r/∂x1 + ∂Fα/∂θ * ∂θ/∂x1
    ∂Fα/∂x2 = ∂Fα/∂r * ∂r/∂x2 + ∂Fα/∂θ * ∂θ/∂x2
    F1x1 = ∂F1/∂x1
    F1y2 = ∂F1/∂y1
    '''
    F1x1 = -1/2 * np.sqrt(r) * np.sin(theta/2)
    F1y1 =  1/2 * np.sqrt(r) * np.cos(theta/2)

    F2x1 = 1/2 * np.sqrt(r) * np.cos(theta/2)
    F2y1 = 1/2 * np.sqrt(r) * np.sin(theta/2)

    F3x1 = -1/2 * np.sqrt(r) * np.sin(3 * theta/2) * np.sin(theta)
    F3y1 =  1/2 * np.sqrt(r) * (np.sin(theta/2) + np.sin(3 * theta/2) * np.cos(theta))

    F4x1 = -1/2 * np.sqrt(r) * np.cos(3 * theta/2) * np.sin(theta)
    F4y1 =  1/2 * np.sqrt(r) * (np.cos(theta/2) + np.cos(3 * theta/2) * np.cos(theta))
    '''
    the derivatives of crack tip asymptotic functions with respect to the global coordinate system (x, y)
    ∂Fα/∂x = ∂Fα/∂x1 * cosα − ∂Fα/∂x2 * sinα
    ∂Fα/∂x = ∂Fα/∂x1 * sinα + ∂Fα/∂x2 * cosα
    '''

    dF1X = F1x1 * np.cos(alpha) - F1y1 * np.sin(alpha)
    dF1Y = F1x1 * np.sin(alpha) + F1y1 * np.cos(alpha)

    dF2X = F2x1 * np.cos(alpha) - F2y1 * np.sin(alpha)
    dF2Y = F2x1 * np.sin(alpha) + F2y1 * np.cos(alpha)

    dF3X = F3x1 * np.cos(alpha) - F3y1 * np.sin(alpha)
    dF3Y = F3x1 * np.sin(alpha) + F3y1 * np.cos(alpha)

    dF4X = F4x1 * np.cos(alpha) - F4y1 * np.sin(alpha)
    dF4Y = F4x1 * np.sin(alpha) + F4y1 * np.cos(alpha)

    dF = np.array([dF1X, dF1Y, dF2X, dF2Y, dF3X, dF3Y, dF4X, dF4Y])

    return F1, F2, F3, F4, dF

def tip_enrichment_func_N1(F1, F2, F3, F4, dN, dF, N):

    B11 = np.array([[(F1 * dN[0, 0]) + (dF[0] * N[0]), 0],
                    [0, (F1 * dN[1, 0]) + (dF[1] * N[0])],
                    [(F1 * dN[1, 0]) + (dF[1] * N[0]), (F1 * dN[0, 0]) + (dF[0] * N[0])]])

    B21 = np.array([[(F2 * dN[0, 0]) + (dF[2] * N[0]), 0],
                    [0, (F2 * dN[1, 0]) + (dF[3] * N[0])],
                    [(F2 * dN[1, 0]) + (dF[3] * N[0]), (F2 * dN[0, 0]) + (dF[2] * N[0])]])

    B31 = np.array([[(F3 * dN[0, 0]) + (dF[4] * N[0]), 0],
                    [0, (F3 * dN[1, 0]) + (dF[5] * N[0])],
                    [(F3 * dN[1, 0]) + (dF[5] * N[0]), (F3 * dN[0, 0]) + (dF[4] * N[0])]])

    B41 = np.array([[(F4 * dN[0, 0]) + (dF[6] * N[0]), 0],
                    [0, (F4 * dN[1, 0]) + (dF[7] * N[0])],
                    [(F4 * dN[1, 0]) + (dF[7] * N[0]), (F4 * dN[0, 0]) + (dF[6] * N[0])]])

    B_tip1 = np.concatenate((B11, B21, B31, B41), axis=1)

    return B_tip1

def tip_enrichment_func_N2(F1, F2, F3, F4, dN, dF, N):

    B12 = np.array([[(F1 * dN[0, 1]) + (dF[0] * N[1]), 0],
                    [0, (F1 * dN[1, 1]) + (dF[1] * N[1])],
                    [(F1 * dN[1, 1]) + (dF[1] * N[1]), (F1 * dN[0, 1]) + (dF[0] * N[1])]])

    B22 = np.array([[(F2 * dN[0, 1]) + (dF[2] * N[1]), 0],
                    [0, (F2 * dN[1, 1]) + (dF[3] * N[1])],
                    [(F2 * dN[1, 1]) + (dF[3] * N[1]), (F2 * dN[0, 1]) + (dF[2] * N[1])]])

    B32 = np.array([[(F3 * dN[0, 1]) + (dF[4] * N[1]), 0],
                    [0, (F3 * dN[1, 1]) + (dF[5] * N[1])],
                    [(F3 * dN[1, 1]) + (dF[5] * N[1]), (F3 * dN[0, 1]) + (dF[4] * N[1])]])

    B42 = np.array([[(F4 * dN[0, 1]) + (dF[6] * N[1]), 0],
                    [0, (F4 * dN[1, 1]) + (dF[7] * N[1])],
                    [(F4 * dN[1, 1]) + (dF[7] * N[1]), (F4 * dN[0, 1]) + (dF[6] * N[1])]])

    B_tip2 = np.concatenate((B12, B22, B32, B42), axis=1)

    return B_tip2

def tip_enrichment_func_N3(F1, F2, F3, F4, dN, dF, N):

    B13 = np.array([[(F1 * dN[0, 2]) + (dF[0] * N[2]), 0],
                    [0, (F1 * dN[1, 2]) + (dF[1] * N[2])],
                    [(F1 * dN[1, 2]) + (dF[1] * N[2]), (F1 * dN[0, 2]) + (dF[0] * N[2])]])

    B23 = np.array([[(F2 * dN[0, 2]) + (dF[2] * N[2]), 0],
                    [0, (F2 * dN[1, 2]) + (dF[3] * N[2])],
                    [(F2 * dN[1, 2]) + (dF[3] * N[2]), (F2 * dN[0, 2]) + (dF[2] * N[2])]])

    B33 = np.array([[(F3 * dN[0, 2]) + (dF[4] * N[2]), 0],
                    [0, (F3 * dN[1, 2]) + (dF[5] * N[2])],
                    [(F3 * dN[1, 2]) + (dF[5] * N[2]), (F3 * dN[0, 2]) + (dF[4] * N[2])]])

    B43 = np.array([[(F4 * dN[0, 2]) + (dF[6] * N[2]), 0],
                    [0, (F4 * dN[1, 2]) + (dF[7] * N[2])],
                    [(F4 * dN[1, 2]) + (dF[7] * N[2]), (F4 * dN[0, 2]) + (dF[6] * N[2])]])

    B_tip3 = np.concatenate((B13, B23, B33, B43), axis=1)

    return B_tip3

def tip_enrichment_func_N4(F1, F2, F3, F4, dN, dF, N):

    B14 = np.array([[(F1 * dN[0, 3]) + (dF[0] * N[3]), 0],
                    [0, (F1 * dN[1, 3]) + (dF[1] * N[3])],
                    [(F1 * dN[1, 3]) + (dF[1] * N[3]), (F1 * dN[0, 3]) + (dF[0] * N[3])]])

    B24 = np.array([[(F2 * dN[0, 3]) + (dF[2] * N[3]), 0],
                    [0, (F2 * dN[1, 3]) + (dF[3] * N[3])],
                    [(F2 * dN[1, 3]) + (dF[3] * N[3]), (F2 * dN[0, 3]) + (dF[2] * N[3])]])

    B34 = np.array([[(F3 * dN[0, 3]) + (dF[4] * N[3]), 0],
                    [0, (F3 * dN[1, 3]) + (dF[5] * N[3])],
                    [(F3 * dN[1, 3]) + (dF[5] * N[3]), (F3 * dN[0, 3]) + (dF[4] * N[3])]])

    B44 = np.array([[(F4 * dN[0, 3]) + (dF[6] * N[3]), 0],
                    [0, (F4 * dN[1, 3]) + (dF[7] * N[3])],
                    [(F4 * dN[1, 3]) + (dF[7] * N[3]), (F4 * dN[0, 3]) + (dF[6] * N[3])]])

    B_tip4 = np.concatenate((B14, B24, B34, B44), axis=1)
    return B_tip4

def heaviside_function1(dN_en):
    B_enriched1 = np.array([[dN_en[0, 0], 0],
                            [0, dN_en[1, 0]],
                            [dN_en[1, 0], dN_en[0, 0]]])
    return B_enriched1

def heaviside_function2(dN_en):
    B_enriched2 = np.array([[dN_en[0, 1], 0],
                            [0, dN_en[1, 1]],
                            [dN_en[1, 1], dN_en[0, 1]]])
    return B_enriched2

def heaviside_function3(dN_en):
    B_enriched3 = np.array([[dN_en[0, 2], 0],
                            [0, dN_en[1, 2]],
                            [dN_en[1, 2], dN_en[0, 2]]])
    return B_enriched3

def heaviside_function4(dN_en):
    B_enriched4 = np.array([[dN_en[0, 3], 0],
                            [0, dN_en[1, 3]],
                            [dN_en[1, 3], dN_en[0, 3]]])
    return B_enriched4

