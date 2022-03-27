'''
AUTHOR : VIKAS MELKOT BALASUBRAMANYAM
MATRICULATION NUMBER : 66133
Personal Programming Project: IMPLEMENTATION OF XFEM AND FRACTURE MECHANICS
#=============================================================================
'''

import numpy as np
from scipy import linalg
import math as m
import enrichment_functions

def r_theta(QP, c_2, alpha):
    '''
    The function calculates distance from the crack tip and a gauss point in polar coordinates (r,θ)
    Parameters
    ----------
    QP : gauss point under query/ node under query
    c_2 : crack tip
    alpha : angle made by the crack tip w.r.t to global x-axis
    Returns
    -------
    distance r, and angle theta
    '''
    #defining crack tip coordinate system
    alpha = alpha * 180 / np.pi
    CTCS = np.array([[m.cos(alpha), m.sin(alpha)],
                     [-m.sin(alpha), m.cos(alpha)]])

    #calculating the x and y distance from the crack tip
    Xn = (QP[0] - c_2[0])
    Yn = (QP[1] - c_2[1])
    #transforming the distance from global system to local crack tip coordinate system
    XYdist = np.matmul(CTCS, np.array([Xn,Yn]))
    #r is the distance from the cracktip to the point
    r = np.sqrt(XYdist[1]**2 + XYdist[0]**2)
    # theta is the angle made by the gauss point w.r.t to crack tip
    theta = m.atan2(XYdist[1], XYdist[0])
    return round(r,3), round(theta,3)

def tip_enrichment(coordinates, alpha, GaussPoint_1to4, D_plane_stress, c_2, r1, r2, r3, r4,
                               theta1, theta2, theta3, theta4):
    '''
    The function computes stiffness matrix for each sub element and integrates it
    Parameters
    ----------
    coordinates : list of sub element coordinates.
    alpha : angle made by the crack w.r.t x-axis
    GaussPoint_1to4 :  Gauss points
    D_plane_stress : plane_stress relation/ Plane_strain relation
    r1, r2, r3, r4 : calculated distance from nodes to the crack tip
    theta1, theta2, theta3, theta4 : calculated diangles from nodes to the crack tip

    Returns
    -------
    Matrix_tip : final element stiffness matrix
    '''

    Z=[]
    # calling asymptotic_functions to calculate the tip enrichment function values
    F11, F21, F31, F41, dF1 = enrichment_functions.asymptotic_functions(r1, theta1, alpha)
    F12, F22, F32, F42, dF2 = enrichment_functions.asymptotic_functions(r2, theta2, alpha)
    F13, F23, F33, F43, dF3 = enrichment_functions.asymptotic_functions(r3, theta3, alpha)
    F14, F24, F34, F44, dF4 = enrichment_functions.asymptotic_functions(r4, theta4, alpha)

    #pre-defining the size of the output matrix
    Matrix_tip = np.zeros([40,40])
    #looping through sub_elements/ domains
    for i in coordinates:
        # looping through 4 gauss points
        for points in GaussPoint_1to4:
            '''
            Defining GaussPoints
            '''
            xi_1 = points[0]
            xi_2 = points[1]

            '''
            4 standard quadrilateral shape functions in CCW direction
            '''
            N1 = 0.25* (1-xi_1) * (1-xi_2)
            N2 = 0.25* (1+xi_1) * (1+xi_2)
            N3 = 0.25* (1+xi_1) * (1+xi_2)
            N4 = 0.25* (1-xi_1) * (1+xi_2)

            N = np.array([N1, N2, N3, N4])

            '''
            Differentiation of shape functions wrt xi_1, xi_2
            '''
            dNdxi = (1/4) * np.array([[-(1-xi_2), 1-xi_2, 1+xi_2, -(1+xi_2)],
                                      [-(1-xi_1), -(1+xi_1), 1+xi_1, 1-xi_1]])

            '''
            convertion from local coordinate system to global coordinate system
            1. Calculate jacobian
            2. Take inverse of the Jacobian
            3. multiply with differentiated shape functions
            '''
            jacobi = np.matmul(dNdxi, i)
            inverse_jacobi = linalg.inv(jacobi)
            dN = np.round(np.matmul(inverse_jacobi, dNdxi), 3)


            # B-matrix for classical DOFs (2 per node)
            B_std = np.array([[dN[0, 0], 0, dN[0, 1], 0, dN[0, 2], 0, dN[0, 3], 0],
                              [0, dN[1, 0], 0, dN[1, 1], 0, dN[1,2], 0, dN[1,3]],
                              [dN[1, 0], dN[0, 0], dN[1, 1], dN[0, 1], dN[1, 2], dN[0, 2], dN[1,3], dN[0, 3]]])

            '''
            asymptotic_crack_tip function call
            #########################################################################
            '''
            #B-matrix for additional DOFs (8 for node 1)
            B_tip1 = enrichment_functions.tip_enrichment_func_N1(F11, F21, F31, F41, dN, dF1, N)
            # print(B_tip1)
            #B-matrix for additional DOFs (8 for node 2)
            B_tip2 = enrichment_functions.tip_enrichment_func_N2(F12, F22, F32, F42, dN, dF2, N)

            #B-matrix for additional DOFs (8 for node 3)
            B_tip3 = enrichment_functions.tip_enrichment_func_N3(F13, F23, F33, F43, dN, dF3, N)

            # B-matrix for additional DOFs (8 for node 4)
            B_tip4 = enrichment_functions.tip_enrichment_func_N4(F14, F24, F34, F44, dN, dF4, N)

            '''
            Steps to calculate K-element
            '''
            # integration(B.T * D * B * ||Jacobian|| * dV)
            # Gauss_Quadrature weights = W1, W2 = 1
            # K_element = W1 * W2 * ΣΣ(B.T * D * B * ||Jacobian||)

            B_tip = np.concatenate((B_std, B_tip1, B_tip2, B_tip3, B_tip4), axis = 1)
            #B.T * D
            Bt_D = np.matmul(B_tip.T, D_plane_stress)
            # B.T * D * B * ||Jacobian||
            Bt_D_B = (np.matmul(Bt_D, B_tip)) * linalg.det(jacobi)
            K5 = np.round(Bt_D_B, 3)
            Z.append(K5)


    for z in Z:
      Matrix_tip += z

    # MI = np.round(Matrix_tip, 3)
    # print(MI[-3:-1,-3:-1])
    return Matrix_tip
