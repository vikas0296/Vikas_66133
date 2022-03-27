'''
AUTHOR : VIKAS MELKOT BALASUBRAMANYAM
MATRICULATION NUMBER : 66133
Personal Programming Project: IMPLEMENTATION OF XFEM AND FRACTURE MECHANICS
#=============================================================================
'''

import numpy as np
import operator; import plots
import math as m
import matplotlib.pyplot as plt


def dictionary(NL, EL):
    '''
    The function is used to store all the nodal coordinates and their corresponding element configuration together
    Parameters
    ----------
    NL : List of nodal coordinates[x,y]
    EL : list of elements
    Returns
    -------
    geom_ns : returns a list of 4 nodal coordinates corresponding to the element in ccw direction
        [[[x1,y1],[x2,y2],[x4,y4],[x3,y3]], [],[].....]
    Elems_nodes : [[1,2,4,3], [],[].....]
    '''
    '''
    ===========================================================================
    list of all the nodal coordinates in a ccw order of each element
      [x3,y3]                [x4,y4]
        +-------------------+
        +                   +
        +                   +
        +     Element_1     +
        +                   +
        +                   +
        +-------------------+
    [x1,y1]                [x2,y2]

    For the above illustration, the function return [[x1,y1],[x2,y2],[x4,y4],[x3,y3]; [1,2,4,3]]
    ===========================================================================
    '''

    y = []
    for j in range(len(EL)):
        for i in EL[j]:
            y.append(tuple(NL[int(i-1)]))
    '''
    ===========================================================================
    Formation of list of 4 nodes per element in ccw direction
    ===========================================================================
    '''
    #looping through all the elements to form a dictionary
    geom_ns = []
    Elems_nodes = []
    for k in range(len(EL)):
        l_dict = {tuple(EL[k]): y[k*4: k*4+4]}
        Elems_nodes.append(l_dict)
        z = operator.itemgetter(*l_dict.keys())(l_dict)
        geom_ns.append(z)

    return geom_ns, Elems_nodes


def elements(Elements, Nodes_elements, x):

    '''
    ===========================================================================
    This function generates the node numbers in ccw with respect to
    the elements
    list of all the nodes in a ccw order of each element
        4                   5                  6
        +-------------------+------------------+
        +                   +                  +
        +                   +                  +
        +     Element_1     +     Element_2    +
        +                   +                  +
        +                   +                  +
        +-------------------+------------------+
        1                   2                  3

    [(1,2,5,4),......
    ........(2,3,6,5)]
    ===========================================================================
    Parameters
    --------------------------
    Elements : total number of elements required
    Nodes_elements : 4, as each element will have 4 nodes
    x : number of elements per row
    Returns
    --------------------------
    ELS : list of elements
    '''
    ELS = np.zeros([Elements, Nodes_elements])
    for i in range(1, x+1):
        for j in range(1, x+1):
            if j == 1:

                ELS[(i-1)*x+j-1, 0] = (i-1)*(x+1) + j                 #EL[0,0]
                ELS[(i-1)*x+j-1, 1] = ELS[(i-1)*x+j-1,0] + 1          #EL[0,1]
                ELS[(i-1)*x+j-1, 3] = ELS[(i-1)*x+j-1,0] + x+1        #EL[0,3]
                ELS[(i-1)*x+j-1, 2] = ELS[(i-1)*x+j-1,3] + 1          #EL[0,2]

            else:

                ELS[(i-1)*x+j-1, 0] = ELS[(i-1)*x+j-2, 1]
                ELS[(i-1)*x+j-1, 3] = ELS[(i-1)*x+j-2, 2]
                ELS[(i-1)*x+j-1, 1] = ELS[(i-1)*x+j-1, 0] + 1
                ELS[(i-1)*x+j-1, 2] = ELS[(i-1)*x+j-1, 3] + 1

    return ELS


def nodes(Nodes, corners, x):

    '''
    This function generates the list of nodes
    Parameters
    --------------------------------
    Nodes : number of nodes(integers)
    corners : list of 4 corners of the mesh
    x : number of elements per row
    Returns
    --------------------------------
    NL : list of nodes
    '''
    #D_2, number of degrees of freedom
    D_2 = 2
    NS = np.zeros([Nodes, D_2])
    a = (corners[1, 0] - corners[0, 0]) / x     #divisions along X-axis
    b = (corners[2, 1] - corners[0, 1]) / x     #divisions along y-axis
    n = 0

    #looping through the divisions to generate nodal coordinates
    for i in range(1,x+2):
        for j in range(1, x+2):

            NS[n,0] = float("{:.2f}".format((corners[0, 0] + (j-1)*a)))  #x-values of nodes
            NS[n,1] = float("{:.2f}".format((corners[0, 1] + (i-1)*b)))  #y-values of nodes
            n += 1

    return NS


def mesh(A,B,x):

    '''
    ===========================================================================
    This function computes all the prerequisites for the
    nodes and elements generations
    ===========================================================================
    Parameters
    ----------------------------------
    A : length along x_axis
    B : length along b_axis
    x : number of elements per row
    Returns
    ---------------------------------
    NL : list of nodes
    EL : list of elements
    '''
    #defining 4 corners of the geometry
    corners = np.array([[0, 0],[A, 0],[0, B],[A, B]])
    Nodes = (x+1)**2
    Elements = x**2
    Nodes_elements = 4

    EL = elements(Elements, Nodes_elements, x)           # 4_nodes per_element list
    NL = nodes(Nodes, corners, x)                  # nodes_list
    round_NL = NL
    return round_NL, EL
