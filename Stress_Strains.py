'''
AUTHOR : VIKAS MELKOT BALASUBRAMANYAM
MATRICULATION NUMBER : 66133
Personal Programming Project: IMPLEMENTATION OF XFEM AND FRACTURE MECHANICS
#=============================================================================
'''

import numpy as np
from scipy import linalg
import sys
import uniform_mesh

def strain_stress_enr(Uxy, Nodes, GaussPoint_1to4, D_plane_stress):
    '''
    ===================================================================================================
    The function calculates stresses and strains at each gauss point using Strain-displacement relation.
    The function calculates stresses and strains for the enriched elements and for the normal elements.
    ===================================================================================================
    Parameters
    ----------
    Uxy : Displacements in m
    Nodes : List of 4 nodal coordinates in ccw forming a single element
    GaussPoint_1to4 : 4 gauss points
    D_plane_stress : Plane stress relation

    Returns
    -------
    Sigmas : Stresses in N/m**2
    spannung : Strains
    '''
    #stresses and strains holder
    Sigmas=[]
    spannung=[]
    #looping through the elements (4nodes)
    for node in Nodes:
        #each element will be looped through 4 gauss points
        for points in GaussPoint_1to4:
            #definition of Gauss points
            xi_1 = points[0]
            xi_2 = points[1]

            # differentiating shape functions w.r.t xi_1 and xi_2 giving 2x4 matrix dNdξ1, dNdξ2


            dNdxi = (1/4) * np.array([[-(1-xi_2), 1-xi_2, 1+xi_2, -(1+xi_2)],
                                      [-(1-xi_1), -(1+xi_1), 1+xi_1, 1-xi_1]])

            #Jacobian converts local coordinate system to global coordinate system
            jacobi = np.matmul(dNdxi, node)

            inverse_jacobi = linalg.inv(jacobi)

            dN = np.matmul(inverse_jacobi, dNdxi)
            # Defining B-matrix
            B_std = np.array([[dN[0, 0], 0, dN[0, 1], 0, dN[0, 2], 0, dN[0, 3], 0],
                              [0, dN[1, 0], 0, dN[1, 1], 0, dN[1, 2], 0, dN[1, 3]],
                              [dN[1, 0], dN[0, 0], dN[1, 1], dN[0, 1], dN[1, 2], dN[0, 2], dN[1, 3], dN[0, 3]]])

            # calculating srains and strains using strain displacement relation
            #strains computed will have a shape of 3x1

            strains = np.matmul(B_std, Uxy[0])
            #stresses computed will have a shape of 3x1
            stresses  = (np.matmul(D_plane_stress, strains))
            spannung.append(strains)
            Sigmas.append(stresses)

    return Sigmas, spannung

