'''
AUTHOR : VIKAS MELKOT BALASUBRAMANYAM
MATRICULATION NUMBER : 66133
Personal Programming Project: IMPLEMENTATION OF XFEM AND FRACTURE MECHANICS
#=============================================================================
'''

import numpy as np
import plots
import matplotlib.pyplot as plt
import Displacement_approx
import Stress_Strains

def filtering(EN_list, GL_nodes):
    '''
    This function filters the nodes which do not require enrichment
    Parameters
    ----------
    EN_list : List of nodes which has been enriched
    GL_nodes : list containing all the nodes
    Returns
    -------
    result : Filtered nodes
    '''
    result = []
    for r in GL_nodes:
        if r not in EN_list:
            result.append(r)

    return result


def E_filter(EN_list, GL_elements):
    '''
    This function filters the elements which do not require enrichment
    Parameters
    ----------
    EN_list : Enriched element list
    GL_elements :  list containing all the elements

    Returns
    -------
    result : filtered elements
    '''

    c= []
    d= []
    result =[]
    for i in GL_elements:
        c.append(tuple(i))

    for j in EN_list:
        d.append(tuple(j))

    for i in c:
        if i not in d:
            result.append(i)
    return result

def G_points(NODES):
    '''
    The function plots the Gauss points in the global coordinate system

      N_4                 N_5               N_6
    +-------------------+------------------+
    +                   +  Gauss points    +
    +  x          x     +  x           x   +
    +                   +                  +
    +  x          x     +  x           x   +
    +                   +                  +
    +-------------------+------------------+
    N_1                 N_2                N_3

    In the above illustration, global coordinates of the 'x' positions are calculated using this function

    Parameters
    ----------
    NODES : list of nodes to calculate the coordinates of new Gauss points
    Returns
    -------
    G : List of 4 gauss points
    GPs : returns 1 gauss point
    '''
    GPs = []

    #looping through the nodal lists
    for i in NODES:
        #4 Gauss points for each element
        GP = np.zeros([4,2])
        Q1, Q2, Q3, Q4 = i[0], i[1], i[2], i[3]

        #approximated global coordinates of the Gauss points
        lengthX = abs(Q1[0] - Q2[0]) / 4
        lengthY = abs(Q1[1]- Q4[1]) / 4

        GP[0, 0] = Q1[0] + lengthX
        GP[0, 1] = Q1[1] + lengthY

        GP[1, 0] = abs(Q2[0] - lengthX)
        GP[1, 1] = abs(Q2[1] + lengthY)


        GP[2, 0] = abs(Q3[0] - lengthX)
        GP[2, 1] = abs(Q3[1] - lengthY)


        GP[3, 0] = abs(Q4[0] + lengthX)
        GP[3, 1] = abs(Q4[1] - lengthY)
        GPs.append(GP)

    G=[]
    for i in GPs:
        for k in i:
            G.append(k)

    return G, GPs

def DSS(NODES, ELEMENTS, cracks, Tside, Hside, CLASS_DOFs, Displacement_vector, alpha, D_plane_stress,
                                GaussPoint_1to4):
    '''
    The function calculates the stresses and displacements for all the nodal coordinates
    Parameters
    ----------
    NODES : Nodal List
    ELEMENTS : element list
    cracks : all the crack segments
    Tside : list of tip enriched element numbers
    Hside : list of heaviside enriched element numbers
    CLASS_DOFs : classical degrees of freedom
    Displacement_vector : displacement vector of the geometry
    alpha : crack angle w.r.t to x-axis
    D_plane_stress : plane stress relation
    GaussPoint_1to4 : list of 4 Gauss points
    Returns
    -------
    STS : Stress
    Ux_Uy : Nodal Displacements
    '''

    STS, Ux_Uy= [],[]
    #looping through all the nodes and elements
    for i,j in zip(NODES, ELEMENTS):
        #calling the displacement spproximation function
        DISPLACEMENTS_ql = Displacement_approx.displacement_approximation([i], cracks, [j], Tside, Hside,
                                                                          CLASS_DOFs, Displacement_vector, alpha)

        #calling the function to compute stresses
        STRESS_ql, strains = Stress_Strains.strain_stress_enr(DISPLACEMENTS_ql, [i], GaussPoint_1to4,D_plane_stress)

        STS.append(STRESS_ql)
        Ux_Uy.append(DISPLACEMENTS_ql)

    return STS, Ux_Uy
































