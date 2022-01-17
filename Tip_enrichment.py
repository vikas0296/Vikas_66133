import numpy as np
from scipy import linalg
import math as m
import enrichment_functions

def r_theta(QP, c_2, alpha):

    CTCS = np.array([[m.cos(alpha), m.sin(alpha)],
                     [-m.sin(alpha), m.cos(alpha)]])

    Xn = QP[0] - c_2[0]
    Yn = QP[1] - c_2[1]
    r = np.sqrt(Xn**2 + Yn**2)
    XYdist = np.matmul(CTCS, np.array([Xn,Yn]))
    if Yn == 0 and Xn==0:
        theta = 0
        return r, theta

    elif Yn == 0 and Xn !=0:
        theta = np.pi
        return r, theta

    elif Xn == 0 and Yn != 0:
        theta = -(90 * np.pi /180)
        return r, theta

    else:
        theta = m.atan2(XYdist[1], XYdist[0])

        return r, theta

def tip_enrichment(coordinates,alpha, GaussPoint_1to4, D_plane_stress, c_2):

    Z=[]

    for i in coordinates:
        QP1 = i[0]
        QP2 = i[1]
        QP3 = i[2]
        QP4 = i[3]
        r1, theta1 = r_theta(QP1, c_2, alpha)
        r2, theta2 = r_theta(QP2, c_2, alpha)
        r3, theta3 = r_theta(QP3, c_2, alpha)
        r4, theta4 = r_theta(QP4, c_2, alpha)
        F11, F21, F31, F41, dF1 = enrichment_functions.asymptotic_functions(r1, theta1, alpha)
        F12, F22, F32, F42, dF2 = enrichment_functions.asymptotic_functions(r2, theta2, alpha)
        F13, F23, F33, F43, dF3 = enrichment_functions.asymptotic_functions(r3, theta3, alpha)
        F14, F24, F34, F44, dF4 = enrichment_functions.asymptotic_functions(r4, theta4, alpha)
        print("rtheta------", r1, theta1*180/np.pi, QP1)

        '''
        This function is called when the element has a crack tip. The following illustration is shown below:
        N_4                 (N_5)               (N_6)
        +-------------------+------------------+
        +                   +                  +
        +                   +                  +
        +$$$$$$$$$$$$$$$$$$$$($$$$$$)-> crack_tip
        +        1          +        2         +
        +                   +                  +
        +-------------------+------------------+
        N_1                 (N_2)               (N_3)

        Here, 4 nodes of element 2 will be tip enriched.
        Parameters
        ----------
        coordinates : list of 4 nodes
        r, theta : crack tip coordinates in radiance
        alpha : crack tip angle wrt x-axis
        Returns
        -------
        Matrix_tip : 40x40 matrix containing 8 classical DOFs and 32 Enriched DOFs
        '''
        Matrix_tip = np.zeros([40,40])
        np.set_printoptions(suppress=True)
        '''

        looping through 4 gauss points
        '''
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
            # K_element = W1 * W2 * Summation(B.T * D * B * ||Jacobian||)

            B_tip = np.concatenate((B_std, B_tip1, B_tip2, B_tip3, B_tip4), axis = 1)
            #B.T * D
            Bt_D = np.matmul(B_tip.T, D_plane_stress)
            # B.T * D * B * ||Jacobian||
            Bt_D_B = (np.matmul(Bt_D, B_tip)) * linalg.det(jacobi)
            K5 = np.round(Bt_D_B, 3)
            Z.append(K5)
    return Z
