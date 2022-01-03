import numpy as np
from scipy import linalg
import new_crack
import sys
import uniform_mesh

def strain_displacement_stress(Uxy, Nodes, GaussPoint_1to4, D_plane_stress):

    for node, U in zip(Nodes, Uxy):
        print(node)
        for points in GaussPoint_1to4:
            xi_1 = points[0]
            xi_2 = points[1]


            dNdxi = (1/4) * np.array([[-(1-xi_2), 1-xi_2, 1+xi_2, -(1+xi_2)],
                                      [-(1-xi_1), -(1+xi_1), 1+xi_1, 1-xi_1]])

            jacobi = np.matmul(dNdxi, node)

            inverse_jacobi = linalg.inv(jacobi)

            dN = np.matmul(inverse_jacobi, dNdxi)

            B_std = np.array([[dN[0, 0], 0, dN[0, 1], 0, dN[0, 2], 0, dN[0, 3], 0],
                              [0, dN[1, 0], 0, dN[1, 1], 0, dN[1, 2], 0, dN[1, 3]],
                              [dN[1, 0], dN[0, 0], dN[1, 1], dN[0, 1], dN[1, 2], dN[0, 2], dN[1, 3], dN[0, 3]]])

            strains = np.matmul(B_std, U)
            # [Epsilon_x, Epsilon_y, Epsilon_xy] = np.matmul(B_std, U)
        #         STRAINS[pos] = [Epsilon_x, Epsilon_y, Epsilon_xy]
            print("-=-=-=-=-=-=-=-=")
            stresses  = np.matmul(D_plane_stress, strains)


















