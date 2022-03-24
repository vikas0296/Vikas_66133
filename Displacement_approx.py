import numpy as np
import math
import operator
import matplotlib.pyplot as plt
import Tip_enrichment
import enrichment_functions

def displacement_approximation(N, cracks, E, Tside, Hside, C_Dofs, disp, alpha):
    '''
    The function calculates the nodal displacements as per the corresponding enrichments

    Parameters
    ----------
    N : list of nodes
    cracks : list of all the crack segments and its coordinates
    E : list of elements
    Tside : List of nodes that are tip enriched
    Hside : List of nodes that are heaviside enriched
    C_Dofs : Total classical DOFs in thegiven geometry
    disp : computed XFEM displacements
    alpha : angle made by the crack w.r.t x-axis
    Returns
    -------
    DISPLACEMENTS : Nodal displacements (8 per element)
    '''

    DISPLACEMENTS = []

    # each heaviside enriched node will have occupied 2 rows in the main displacement array
    H_dof = len(Hside)*2
    # each Tip enriched node will have occupied 8 rows in the main displacement array
    T_dof = len(Tside)*8

    Tside = list(np.asarray(Tside))
    Hside = list(np.asarray(Hside))

    #looping through list of elements and nodes
    for i, node in zip(list(E), N):
        #predefining the list of size 8 for the nodal displacements
        Ux_Uy = [0,0,0,0,0,0,0,0]
        i = list(np.asarray(i))
        for j, n in zip(i,node):
            Ux_Uy[i.index(j)*2] = (disp[int(j)*2])
            Ux_Uy[i.index(j)*2+1] = (disp[int(j)*2+1])
            if j in Hside:
                for k in Hside:
                    if j == k:
                        # calling step_function to calculate the value of enrichment function (1 or 0 or -1)
                        H1 = enrichment_functions.step_function(n, cracks)
                        # summation of corresponding displacements
                        # to compute U_x
                        Ux_Uy[i.index(j)*2] = (disp[int(j)*2] + disp[(C_Dofs + Hside.index(k)*2)] * H1)
                        # to compute U_y
                        Ux_Uy[i.index(j)*2+1] = (disp[int(j)*2+1] + disp[(C_Dofs + Hside.index(k)*2+1)] * H1)


            elif j in Tside:
                for l in Tside:
                    if j == l:
                        #calculation of distance r and theta from nodes to the crack tip
                        r, theta = Tip_enrichment.r_theta(n, cracks[-1], alpha)
                        # calling asymptotic_functions to calculate the tip enrichment function values
                        F1 = np.sqrt(r) * np.sin(theta/2)
                        F2 = np.sqrt(r) * np.cos(theta/2)
                        F3 = np.sqrt(r) * np.sin(theta/2) * np.sin(theta)
                        F4 = np.sqrt(r) * np.cos(theta/2) * np.sin(theta)
                        # summation of corresponding displacements
                        # to compute U_x
                        Ux_Uy[i.index(j)*2] = (disp[int(j)*2] + (disp[(C_Dofs + H_dof + (Tside.index(l)*8))]) * F1 +
                                                (disp[(C_Dofs + H_dof + (Tside.index(l)*8))+2] )* F2 +
                                                (disp[(C_Dofs + H_dof + (Tside.index(l)*8))+4]) * F3 +
                                                (disp[(C_Dofs + H_dof + (Tside.index(l)*8))+6]) * F4)
                        # to compute U_y
                        Ux_Uy[i.index(j)*2+1] = (disp[int(j)*2+1] + disp[(C_Dofs + H_dof + Tside.index(l)*8+1)] * F1 +
                                                  (disp[(C_Dofs + H_dof + (Tside.index(l)*8))+3]) * F2 +
                                                  (disp[(C_Dofs + H_dof + (Tside.index(l)*8))+5]) * F3 +
                                                  (disp[(C_Dofs + H_dof + (Tside.index(l)*8))+7]) * F4)

        # to convert 'nan' to '0'
        Uxy = np.nan_to_num(Ux_Uy)
        DISPLACEMENTS.append(Uxy)

    return DISPLACEMENTS
