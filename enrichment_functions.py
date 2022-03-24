import numpy as np
from scipy import linalg
import math as m
'''
All the functions necessary for the tip enrichment and heaviside_enrichment have been put together
'''
def step_function(A, cracks):
    '''
    N_4                 N_5               N_6
    +-------------------+------------------+
    +                   +                  +
    +                   +                  +
    (*******************************************)-> through crack
    +                   +                  +
    +                   +                  +
    +-------------------+------------------+
    N_1                 N_2                N_3

    for the above illustration the function outputs +1 or -1 based on the nodal positions
    w.r.t to the crack.
    Parameters
    ----------
    A : point under query
    cracks : all the crack segment coordinates

    Returns
    -------
    1 or 0 or -1
    '''
    delta = []
    #looping through all the crack segment coordinates
    for i in range(0,len(cracks)-1):
        crack1 = cracks[i]
        crack2 = cracks[i+1]
        #calculating the distances from left point of the crack to the point under query and from right point
        # of the crack to the point under query
        TL = crack1[0] - A[0]
        TR = crack1[1] - A[1]
        BL = crack2[0] - A[0]
        BR = crack2[1] - A[1]
        #forming a matrix
        matrix = np.array([[TL, TR], [BL, BR]])
        #to calculate determinant
        step = np.linalg.det(matrix)
        delta.append(step)

    #calculate minimum of list
    get = min(delta)

    #check the sign of the minimum value of the list
    if np.sign(get) > 0:
        return 1

    elif np.sign(get) == 0:
        return 0

    elif np.sign(get) < 0:
        return -1

def classical_FE(x, GaussPoint_1to4, D_plane_stress):
    '''
    This function generates K-matrix for the pure elements
    Parameters
    ----------
    x : Nodes list
    GaussPoint_1to4 : List of 4 Gauss points
    D_plane_stress : Plane stress equation
    Returns
    -------
    non_enriched : K-matrix for the pure elements
    '''
    store = []
    non_enriched = []
    

    for points in GaussPoint_1to4:
        '''
        Defining GaussPoints
        '''
        xi_1 = points[0]
        xi_2 = points[1]

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
        jacobi = np.matmul(dNdxi, x)

        inverse_jacobi = linalg.inv(jacobi)

        dN = np.matmul(inverse_jacobi, dNdxi)

        B_std = np.array([[dN[0, 0], 0, dN[0, 1], 0, dN[0, 2], 0, dN[0, 3], 0],
                          [0, dN[1, 0], 0, dN[1, 1], 0, dN[1, 2], 0, dN[1, 3]],
                          [dN[1, 0], dN[0, 0], dN[1, 1], dN[0, 1], dN[1, 2], dN[0, 2], dN[1, 3], dN[0, 3]]])

        '''
        Steps to calculate K-element
        '''
        # integration(B.T * D * B * ||Jacobian|| * dV)
        # Gauss_Quadrature weights = W1, W2 = 1
        # K_element = W1 * W2 * ΣΣ(B.T * D * B * ||Jacobian||)

        Bt_D_1 = np.matmul(B_std.T, D_plane_stress)
        Bt_D_B1 = (np.matmul(Bt_D_1, B_std)) * linalg.det(jacobi)
        non_enriched.append(Bt_D_B1)

    Zero = np.zeros([8,8])
    for z in non_enriched:
        Zero += z
    return Zero

def classic_B_matric(coordinates, D_plane_stress, xi_1, xi_2, H1, H2, H3, H4):
    '''
    The function generates B-matrix for the enriched elements
    Parameters
    ----------
    coordinates : List of nodal coordinates
    D_plane_stress : plane stress relation
    xi_1 : Gauss point coordinate in x-direction
    xi_2 : Gauss point coordinate in y-direction
    H1, H2, H3, H4 : signed step function values
    Returns
    -------
    B_std : Std B_matrix
    dN_en : Enrtiched B_matrix
    N : shape functions
    dN : transformed shape functions from local coordinate system to global system
    jacobi : Jacobi matrix
    dN_par : B-matrix for pre tip elements
    '''

    #================for tip enrichment,shape functions are necessary=================================
    N1 = 0.25* (1-xi_1) * (1-xi_2)
    N2 = 0.25* (1+xi_1) * (1-xi_2)
    N3 = 0.25* (1+xi_1) * (1+xi_2)
    N4 = 0.25* (1-xi_1) * (1+xi_2)

    N = np.array([N1, N2, N3, N4])
    #==================================================================================================
    dNdxi = (1/4) * np.array([[-(1-xi_2), 1-xi_2, 1+xi_2, -(1+xi_2)],
                              [-(1-xi_1), -(1+xi_1), 1+xi_1, 1-xi_1]])

    '''
     convertion from local coordinate system to global coordinate system
     1. Calculate jacobian
     2. Take inverse of the Jacobian
     3. multiply with differentiated shape functions
    '''
    jacobi = np.matmul(dNdxi, coordinates)
    inverse_jacobi = linalg.inv(jacobi)
    dN = np.matmul(inverse_jacobi, dNdxi)

    # B-matrix for classical DOFs (2 per node)
    B_std = np.array([[dN[0, 0], 0, dN[0, 1], 0, dN[0, 2], 0, dN[0, 3], 0],
                      [0, dN[1, 0], 0, dN[1, 1], 0, dN[1,2], 0, dN[1,3]],
                      [dN[1, 0], dN[0, 0], dN[1, 1], dN[0, 1], dN[1, 2], dN[0, 2], dN[1,3], dN[0, 3]]])

    # Enriched B-matrix
    dN_en = np.array([[dN[0, 0]*H1, dN[0, 1]*H2, dN[0, 2]*H3, dN[0, 3]*H4],
                      [dN[1, 0]*H1, dN[1, 1]*H2, dN[1, 2]*H3, dN[1, 3]*H4]])

    # Enriched B-matrix for individual nodes this will be used later
    dN_par = np.array([[dN[0, 0], dN[0, 1], dN[0, 2], dN[0, 3]],
                       [dN[1, 0], dN[1, 1], dN[1, 2], dN[1, 3]]])


    return B_std, dN_en, N, dN, jacobi, dN_par

def asymptotic_functions(r, theta, alpha):
    '''
    This function generates the necessary terms required for generating tip enriched B-matrix.
    Parameters
    ----------
    r : polar coordinate of the point under query, measured from the crack tip
    theta : angle of the point under query, measured from the crack tip (-pi to +pi)
    alpha : angle made by crack w.r.t global x-axis
    Returns
    -------
    F1 : asymptotic_function1
    F2 : asymptotic_function2
    F3 : asymptotic_function3
    F4 : asymptotic_function4
    dF : Differentiation of the enrichment functions associated with Shape functions
    '''
    '''
    tip enrichment terms
    Fα (r, θ) = [F1, F2, F3, F4]
    '''
    theta = theta * 180 / np.pi
    F1 = np.sqrt(r) * np.sin(theta/2)
    F2 = np.sqrt(r) * np.cos(theta/2)
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

    F1x1 = (-1/(2 * np.sqrt(r))) * np.sin(theta/2)
    F1y1 =  (1/(2 * np.sqrt(r))) * np.cos(theta/2)

    F2x1 = (1/(2 * np.sqrt(r))) * np.cos(theta/2)
    F2y1 = (1/(2 * np.sqrt(r))) * np.sin(theta/2)

    F3x1 = (-1/(2 * np.sqrt(r))) * np.sin(3 * theta/2) * np.sin(theta)
    F3y1 =  (1/(2 * np.sqrt(r))) * (np.sin(theta/2) + np.sin(3 * theta/2) * np.cos(theta))

    F4x1 = (-1/(2 * np.sqrt(r))) * np.cos(3 * theta/2) * np.sin(theta)
    F4y1 =  (1/(2 * np.sqrt(r))) * (np.cos(theta/2) + np.cos(3 * theta/2) * np.cos(theta))

    '''
    the derivatives of crack tip asymptotic functions with respect to the global coordinate system (x, y)
    ∂Fα/∂x = ∂Fα/∂x1 * cosα − ∂Fα/∂x2 * sinα
    ∂Fα/∂x = ∂Fα/∂x1 * sinα + ∂Fα/∂x2 * cosα
    α = 1,2,3,4
    '''

    dF1X = F1x1 * np.cos(alpha) - F1y1 * np.sin(alpha)
    dF1Y = F1x1 * np.sin(alpha) + F1y1 * np.cos(alpha)

    dF2X = F2x1 * np.cos(alpha) - F2y1 * np.sin(alpha)
    dF2Y = F2x1 * np.sin(alpha) + F2y1 * np.cos(alpha)

    dF3X = F3x1 * np.cos(alpha) - F3y1 * np.sin(alpha)
    dF3Y = F3x1 * np.sin(alpha) + F3y1 * np.cos(alpha)

    dF4X = F4x1 * np.cos(alpha) - F4y1 * np.sin(alpha)
    dF4Y = F4x1 * np.sin(alpha) + F4y1 * np.cos(alpha)

    # array used for defining the B-matrix for the tip enriched elements
    dF = np.array([dF1X, dF1Y, dF2X, dF2Y, dF3X, dF3Y, dF4X, dF4Y])

    return F1, F2, F3, F4, dF

def tip_enrichment_func_N1(F1, F2, F3, F4, dN, dF, N):
    '''
    The function generates a B-matrix for the "Lower left" node w.r.t crack tip

    Parameters
    ----------
    F1 : asymptotic_function1
    F2 : asymptotic_function2
    F3 : asymptotic_function3
    F4 : asymptotic_function4
    dF : Differentiation of the enrichment functions associated with Shape functions
    dN = Differentiation of Shape functions w.r.t x and y (2x4 shape)
    N : Shape function values
    Returns
    -------
    B_tip1 : B_matrix for "Lower left" node (3x8 shape)
    '''

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

    # 1 node will have 8 additional DOF, hence 4 individual B-matrices have been concatenated
    B_tip1 = np.concatenate((B11, B21, B31, B41), axis=1)

    return B_tip1

def tip_enrichment_func_N2(F1, F2, F3, F4, dN, dF, N):
    '''
    The function generates a B-matrix for the "Lower right" node w.r.t crack tip

    Parameters
    ----------
    F1 : asymptotic_function1
    F2 : asymptotic_function2
    F3 : asymptotic_function3
    F4 : asymptotic_function4
    dF : Differentiation of the enrichment functions associated with Shape functions
    dN = Differentiation of Shape functions w.r.t x and y (2x4 shape)
    N : Shape function values
    Returns
    -------
    B_tip2 : B_matrix for "Lower right" node (3x8 shape)
    '''
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

    # 1 node will have 8 additional DOF, hence 4 individual B-matrices have been concatenated
    B_tip2 = np.concatenate((B12, B22, B32, B42), axis=1)

    return B_tip2

def tip_enrichment_func_N3(F1, F2, F3, F4, dN, dF, N):
    '''
    The function generates a B-matrix for the "top right" node w.r.t crack tip
    Parameters
    ----------
    F1 : asymptotic_function1
    F2 : asymptotic_function2
    F3 : asymptotic_function3
    F4 : asymptotic_function4
    dF : Differentiation of the enrichment functions associated with Shape functions
    dN = Differentiation of Shape functions w.r.t x and y (2x4 shape)
    N : Shape function values
    Returns
    -------
    B_tip3 : B_matrix for "top right" node (3x8 shape)
    '''

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

    # 1 node will have 8 additional DOF, hence 4 individual B-matrices have been concatenated
    B_tip3 = np.concatenate((B13, B23, B33, B43), axis=1)

    return B_tip3

def tip_enrichment_func_N4(F1, F2, F3, F4, dN, dF, N):
    '''
    The function generates a B-matrix for the "top left" node w.r.t crack tip
    Parameters
    ----------
    F1 : asymptotic_function1
    F2 : asymptotic_function2
    F3 : asymptotic_function3
    F4 : asymptotic_function4
    dF : Differentiation of the enrichment functions associated with Shape functions
    dN = Differentiation of Shape functions w.r.t x and y (2x4 shape)
    N : Shape function values
    Returns
    -------
    B_tip4 : B_matrix for "top left" node (3x8 shape)
    '''
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

    # 1 node will have 8 additional DOF, hence 4 individual B-matrices have been concatenated
    B_tip4 = np.concatenate((B14, B24, B34, B44), axis=1)
    return B_tip4

def heaviside_function1(dN_en, H1):
    '''
    The function generates a B-matrix for the "lower left" node w.r.t crack tip
    Parameters
    ----------
    dN_en : differentiated shape function W.r.t x and y (2x4 shape)
    H1 : step function value for the "lower left" node
    Returns
    -------
    B_enriched1 : Enriched B-matrix for lower left node
    '''
    # 1 node will have 2 additional DOF, hence individual B-matrix has been computed
    B_enriched1 = np.array([[dN_en[0, 0]*H1, 0],
                            [0, dN_en[1, 0]*H1],
                            [dN_en[1, 0]*H1, dN_en[0, 0]*H1]])

    return B_enriched1

def heaviside_function2(dN_en, H2):
    '''
    The function generates a B-matrix for the "lower right" node w.r.t crack tip
    Parameters
    ----------
    dN_en : differentiated shape function W.r.t x and y (2x4 shape)
    H2 : step function value for the "lower right" node
    Returns
    -------
    B_enriched2 : Enriched B-matrix for lower right node
    '''
    # 1 node will have 2 additional DOF, hence individual B-matrix has been computed
    B_enriched2 = np.array([[dN_en[0, 1]*H2, 0],
                            [0, dN_en[1, 1]*H2],
                            [dN_en[1, 1]*H2, dN_en[0, 1]*H2]])

    return B_enriched2

def heaviside_function3(dN_en, H3):
    '''
    The function generates a B-matrix for the "top right" node w.r.t crack tip
    Parameters
    ----------
    dN_en : differentiated shape function W.r.t x and y (2x4 shape)
    H3 : step function value for the "top right" node
    Returns
    -------
    B_enriched3 : Enriched B-matrix for top right node
    '''
    # 1 node will have 2 additional DOF, hence individual B-matrix has been computed
    B_enriched3 = np.array([[dN_en[0, 2]*H3, 0],
                            [0, dN_en[1, 2]*H3],
                            [dN_en[1, 2]*H3, dN_en[0, 2]*H3]])
    return B_enriched3

def heaviside_function4(dN_en, H4):
    '''
    The function generates a B-matrix for the "top left" node w.r.t crack tip
    Parameters
    ----------
    dN_en : differentiated shape function W.r.t x and y (2x4 shape)
    H4 : step function value for the "top left" node
    Returns
    -------
    B_enriched3 : Enriched B-matrix for top right node

    '''
    # 1 node will have 2 additional DOF, hence individual B-matrix has been computed
    B_enriched4 = np.array([[dN_en[0, 3]*H4, 0],
                            [0, dN_en[1, 3]*H4],
                            [dN_en[1, 3]*H4, dN_en[0, 3]*H4]])
    return B_enriched4

