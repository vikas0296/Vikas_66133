import numpy as np
from scipy import linalg
import new_crack
import sys
import uniform_mesh

def Heaviside_enrichment(coordinates, H2, GaussPoint_1to4, D_plane_stress):

    '''
    ===========================================================================
    This function generates the K-matrix for the heaviside-enriched element
    Parameters
    ----------
    coordinates : list of 4 nodes of each element/sub_element
    H2 : Heavyside function 0/1 based on the location of nodes w.r.t crack
    Returns
    -------
    Sub_matrix_TL : top left sub-matrix of the K_main element (8x8)
    Sub_matrix_TR : top right sub-matrix of the K_main element (8x8)
    Sub_matrix_BL : bottom left sub-matrix of the K_main element (8x8)
    Sub_matrix_BR : bottom right sub-matrix of the K_main element (8x8)
    ===========================================================================
    '''
    np.set_printoptions(suppress=True)

    Sub_matrix_TR = np.zeros([8,8])
    Sub_matrix_TL = np.zeros([8,8])
    Sub_matrix_BL = np.zeros([8,8])
    Sub_matrix_BR = np.zeros([8,8])

    '''
    ============================================================================================================
    Heaviside function:
        if the gauss point is below the interface(crack), then the function value H2 will be 0
        otherwise 1

    Quadratic Shape-functions in CCW direction
    N1 = 0.25* (1+xi_1) * (1+xi_2)
    N2 = 0.25* (1-xi_1) * (1+xi_2)
    N3 = 0.25* (1+xi_1) * (1-xi_2)
    N4 = 0.25* (1-xi_1) * (1-xi_2)
    ============================================================================================================
    '''
    H1 = 0
    # print(coordinates)
    for points in GaussPoint_1to4:
        xi_1 = points[0]
        xi_2 = points[1]
        '''
        =======================================================================
        Defining the enrichment functions
        =======================================================================
        enrichment_0 = (1/4) * np.array([[-(1-xi_2)*(H1-H2)],
                                          [-(1-xi_1)*(H1-H2)]])
        enrichment_1 = (1/4) * np.array([[(1-xi_2)*(H1-H2)],
                                        [-(1+xi_1)*(H1-H2)]])
        enrichment_2 = (1/4) * np.array([[(1+xi_2)*(H1-H2)],
                                        [(1+xi_1)*(H1-H2)]])

        enrichment_3 = (1/4) * np.array([[-(1+xi_2)*(H1-H2)],
                                        [(1-xi_1)*(H1-H2)]])
        =======================================================================
        '''

        '''
        differentiating normal shape functions
        '''
        dNdxi = (1/4) * np.array([[-(1-xi_2), 1-xi_2, 1+xi_2, -(1+xi_2)],
                                  [-(1-xi_1), -(1+xi_1), 1+xi_1, 1-xi_1]])

        '''
        converting local coordinate system to global coordinates
        '''
        jacobi = np.matmul(dNdxi, coordinates)

        inverse_jacobi = linalg.inv(jacobi)

        dN = np.matmul(inverse_jacobi, dNdxi)

        '''
        differentiating enriched shape functions and heavyside function
        '''
        dN_en = (H1-H2) * (1/4) * np.array([[dN[0, 0], dN[0, 1], dN[0, 2], dN[0, 3]],
                                            [dN[1, 0], dN[1, 1], dN[1, 2], dN[1, 3]]])

        '''
        Standard B-matrix
        '''
        B_std = np.array([[dN[0, 0], 0, dN[0, 1], 0, dN[0, 2], 0, dN[0, 3], 0],
                          [0, dN[1, 0], 0, dN[1, 1], 0, dN[1, 2], 0, dN[1, 3]],
                          [dN[1, 0], dN[0, 0], dN[1, 1], dN[0, 1], dN[1, 2], dN[0, 2], dN[1, 3], dN[0, 3]]])

        '''
        Heaviside enriched B-matrix
        '''
        B_enriched = np.array([[dN_en[0, 0], 0, dN_en[0, 1], 0, dN_en[0, 2], 0, dN_en[0, 3], 0],
                               [0, dN_en[1, 0], 0, dN_en[1, 1], 0, dN_en[1, 2], 0, dN_en[1, 3]],
                               [dN_en[1, 0], dN_en[0, 0], dN_en[1, 1], dN_en[0, 1], dN_en[1, 2], dN_en[0, 2],
                                dN_en[1, 3], dN_en[0, 3]]])

        '''
        ============================================================================================================
        K_enriched element =|[integrating-std_sub_elements], [integrating-std_sub_elements and enriched_nodes]|
                            |[integrating-std_sub_elements and enriched_nodes].T,[integrating-enriched_sub_elements]|

        K_enriched element = [shape(1)=8x8][shape(2)=8x8]
                             [shape(3)=8x8][shape(4)=8x8]

        K_enriched element = 16x16
        ============================================================================================================
        '''

        '''
        ============================================================================================================
        # integrating all the std_sub_elements
        K_element =  Sigma[(B_std.T * D * B_std) * det(Jacobian)*Weights of gauss quadrature]
        This is the top left element of the K_main element.
        It has a size of 8x8
        ============================================================================================================
        '''
        Bt_D_1 = np.matmul(B_std.T, D_plane_stress)
        Bt_D_B1 = (np.matmul(Bt_D_1, B_std)) * linalg.det(jacobi)
        K51 = np.round(Bt_D_B1, 3)
        Sub_matrix_TL += K51

        '''
        ============================================================================================================
        #integrating all the std_sub_elements with enriched_nodes
        K_element =  Sigma[(B_std.T * D * B_enriched) * det(Jacobian)*Weights of gauss quadrature]
        This is the top right element of the K_main element.
        It has a size of 8x8
        ============================================================================================================
        '''
        Bt_D_2 = np.matmul(B_std.T, D_plane_stress)
        Bt_D_B2 = (np.matmul(Bt_D_2, B_enriched)) * linalg.det(jacobi)
        K52 = np.round(Bt_D_B2, 3)
        Sub_matrix_TR += K52

        '''
        ============================================================================================================
        #integrating all the std_sub_elements with enriched_nodes
        K_element =  Sigma[(B_enriched.T * D * B_std) * det(Jacobian)*Weights of gauss quadrature]
        This is the bottom left element of the K_main element.
        It has a size of 8x8
        ============================================================================================================
        '''

        Bt_D_B3 = Bt_D_B2.T
        K53 = np.round(Bt_D_B3, 3)
        Sub_matrix_BL += K53

        '''
        ============================================================================================================
        #integrating all the enriched_sub_elements
        K_element =  Sigma[(B_enriched.T * D * B_enriched) * det(Jacobian)*Weights of gauss quadrature]
        This is the bottom right element of the K_main element.
        It has a size of 8x8
        ============================================================================================================
        '''
        Bt_D_4 = np.matmul(B_enriched.T, D_plane_stress)
        Bt_D_B4 = (np.matmul(Bt_D_4, B_enriched)) * linalg.det(jacobi)
        K54 = np.round(Bt_D_B4, 3)
        Sub_matrix_BR += K54

    return Sub_matrix_TL, Sub_matrix_TR, Sub_matrix_BL, Sub_matrix_BR





