import numpy as np  
from scipy import linalg 
import math as m
import enrichment_functions

def tip_enrichment(coordinates, r, theta, alpha):
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
    Material Properties
    '''
    D = 1 
    nu = 0.33
    D_plane_stress = (D / (1 - nu**2)) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]])
    
    '''
    4 - Gauss_points
    '''
    GaussPoint_1to4 = np.array([[ 1/np.sqrt(3),  1/np.sqrt(3)], [ 1/np.sqrt(3), -1/np.sqrt(3)],
                                [-1/np.sqrt(3),  1/np.sqrt(3)], [-1/np.sqrt(3), -1/np.sqrt(3)]])
    
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
        jacobi = np.matmul(dNdxi, coordinates)
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
        B_tip1 = enrichment_functions.enrichment_func_N1(F1, F2, F3, F4, dN, dF, N)
        
        #B-matrix for additional DOFs (8 for node 2)           
        B_tip2 = enrichment_functions.enrichment_func_N2(F1, F2, F3, F4, dN, dF, N)
        
        #B-matrix for additional DOFs (8 for node 3)              
        B_tip3 = enrichment_functions.enrichment_func_N3(F1, F2, F3, F4, dN, dF, N)
        
        # B-matrix for additional DOFs (8 for node 4)                     
        B_tip4 = enrichment_functions.enrichment_func_N4(F1, F2, F3, F4, dN, dF, N)

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
        Matrix_tip += K5
    return  Matrix_tip                           

# x  =  [[2,  1],
#       [2.5,  1],
#       [2.5,  1.25],
#       [2,  1.25]]

# crack_tip = [1, 2]
# alpha = 0
# r, theta = enrichment_functions.to_polar_in_radians(crack_tip[0], crack_tip[1])
# Matrix_tip = tip_enrichment(x, r, theta, alpha)
# print(Matrix_tip)
