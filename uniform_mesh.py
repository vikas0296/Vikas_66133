import numpy as np
import operator
import new_crack
import math
import heaviside_enrichment
import matplotlib.pyplot as plt
import Assignment
import class_crack
import plots
import affected_enrichments
from scipy import linalg
import enrichment_functions

def dictionary(NL,EL):
    y =[]
    for j in range(len(EL)):
        for i in EL[j]:
            y.append(tuple(NL[int(i-1)]))
    '''
    ===========================================================================
    Formation of list of 4 nodes per element in ccw direction
    ===========================================================================
    '''
    all_ns = []
    Elems_nodes=[]
    for k in range(len(EL)):
        l_dict = {tuple(EL[k]): y[k*4: k*4+4]}
        Elems_nodes.append(l_dict)
        z = operator.itemgetter(*l_dict.keys())(l_dict)
        all_ns.append(z)
    return all_ns, Elems_nodes


def elements(Elements, Nodes_elements,x):
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
    Elements : total numbe rof elements required
    Nodes_elements : 4, as each element will have 4 nodes
    x : number of elements per row
    Returns
    --------------------------
    EL : list of elements
    '''
    EL = np.zeros([Elements, Nodes_elements])
    for i in range(1,x+1):
        for j in range(1,x+1):
            if j==1:

                EL[(i-1)*x+j-1, 0] = (i-1)*(x+1) + j                #EL[0,0]
                EL[(i-1)*x+j-1, 1] = EL[(i-1)*x+j-1,0] + 1          #EL[0,1]
                EL[(i-1)*x+j-1, 3] = EL[(i-1)*x+j-1,0] + x+1        #EL[0,3]
                EL[(i-1)*x+j-1, 2] = EL[(i-1)*x+j-1,3] + 1          #EL[0,2]

            else:

                EL[(i-1)*x+j-1, 0] = EL[(i-1)*x+j-2, 1]
                EL[(i-1)*x+j-1, 3] = EL[(i-1)*x+j-2, 2]
                EL[(i-1)*x+j-1, 1] = EL[(i-1)*x+j-1, 0] + 1
                EL[(i-1)*x+j-1, 2] = EL[(i-1)*x+j-1, 3] + 1

    return EL

def nodes(Nodes,corners, x):
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
    D_2 = 2
    NL = np.zeros([Nodes, D_2])
    a = (corners[1, 0] - corners[0, 0]) / x     #divisions along X-axis
    b = (corners[2, 1] - corners[0, 1]) / x     #divisions along y-axis
    n = 0

    for i in range(1,x+2):
        for j in range(1, x+2):
            NL[n,0] = corners[0, 0] + (j-1)*a  #x-values of nodes
            NL[n,1] = corners[0, 1] + (i-1)*b  #y-values of nodes
            n += 1

    return NL

def uniform_mesh(A,B,x):
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
    corners = np.array([[1, 1],[A, 1],[1, B],[A, B]])
    Nodes = (x+1)**2
    Elements = x**2
    Nodes_elements = 4

    EL = elements(Elements, Nodes_elements, x)           # 4_nodes per_element list
    NL = nodes(Nodes, corners, x)                  # nodes_list

    return NL, EL

if __name__ == "__main__":

    A = 4           #length of the geometry along x and y axis
    B = 4
    x = 3                                       #3x3 will be elements, 4x4 will be nodes
    NL, EL = uniform_mesh(A, B, x)
    x1 = NL[0]
    x2 = NL[1]
    '''
    length of a segment, given 2 points = sqrt((x2-x1)**2 + (y2-y1)**2)
    '''
    length_element = np.sqrt((x2[0]-x1[0])**2 + (x2[1]-x1[1])**2)
    print("the length of each element is ", length_element)
    all_ns, Elems_nodes = dictionary(NL,EL)
    elements = [ ]
    for j in Elems_nodes:
        elements.append(list(j.keys())[0])

    plots.plot_mesh(NL, A, B)


    #=============================Material_properties===========================================

    D = 200000
    nu = 0.33
    D_plane_stress = (D / (1 - nu**2)) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]])

    GaussPoint_1to4 = np.array([[ 1/np.sqrt(3),  1/np.sqrt(3)], [ 1/np.sqrt(3), -1/np.sqrt(3)],
                                [-1/np.sqrt(3),  1/np.sqrt(3)], [-1/np.sqrt(3), -1/np.sqrt(3)]])


    '''
    ===========================================================================
    list of all the nodes in a ccw order of each element
      N_4                    N_5                N_6
        +-------------------+------------------+
        +                   +                  +
        +                   +                  +
        +     Element_1     +     Element_2    +
        +                   +                  +
        +                   +                  +
        +-------------------+------------------+
    N_1                    N_2                N_3

    (N_1, N_2, N_5, N_4, N_2, N_3, N_6, N_5...............)
    ===========================================================================
    '''

    '''
    ===========================================================================
    Crack modelling
    ===========================================================================
    '''

    cracktip_1 = np.array([1, 3.5])
    cracktip_2 = np.array([2.5,2.5])

    plots.plot_crack(cracktip_1, cracktip_2)
    crack_length = np.sqrt((cracktip_1[0]-cracktip_1[1])**2 + (cracktip_2[0]-cracktip_2[1])**2)

    print("the crack length is", crack_length)
    c_1 = np.array([cracktip_1[0], cracktip_2[0]])
    c_0 = np.copy(c_1)
    c_2 = np.array([cracktip_1[1], cracktip_2[1]])

    r, theta = Assignment.to_polar_in_radians(c_2[0], c_2[1])
    alpha = 0
    gamma=[]
    Jarvis=[]
    # K_matrices holder
    KEs = []
    EEs = []
    Heavys = []
    # ################################
    if c_2[0] >= A:
        print(f"The crack_length {crack_length} is >= to the geometry length {A},"
                  "therefore the sample is broken into two halves")

    if crack_length <= length_element:
        for node, el_co in zip(all_ns, elements) :
            charlie, foxtrot, delta = class_crack.domain_splitter(node, el_co, c_1, c_2, length_element,
                                                                  GaussPoint_1to4, D_plane_stress)
            print(delta)
            print(foxtrot[0:8,0:8])
            Jarvis.append(charlie)
            KEs.append(foxtrot)
            EEs.append(delta)

    else:
        while c_1[0] < (c_2[0]-length_element):
              c_1[0] = c_1[0] + length_element
              XX = np.array([c_1[0],c_1[1]])
              gamma.append(XX)

        gamma.insert(0, c_0)
        gamma.insert(len(gamma), c_2)
        v_1 = gamma[-1]
        v_2 = gamma[-2]

        if abs(v_1[0]-v_2[0]) <= length_element:
            for i,j in zip(all_ns, elements):
                epsilon, T, U = class_crack.domain_splitter(i, j, v_2, v_1, length_element, GaussPoint_1to4, D_plane_stress)
                Jarvis.append(epsilon)
                # print(T[0:8,0:8])
                KEs.append(T)
                EEs.append(U)


    for i in range(len(gamma)-2):
        for node, el_co in zip(all_ns, elements):
          eta, K_mats, Ele_coords  = class_crack.Crack_length_PP_element_length(node,el_co, gamma[i], gamma[i+1], crack_length, length_element,
                                                            v_2, GaussPoint_1to4,  r, theta, alpha, D_plane_stress)
          Jarvis.append(eta)
          KEs.append(K_mats)
          if Ele_coords != None and len(Ele_coords) == 4:
              Heavys.append(Ele_coords)
              EEs.append(Ele_coords)
          else:
              EEs.append(Ele_coords)

    # #==========================================================================
    # #Block to extract the affected elements due to enrichment


    g=[]
    influenced_elements=[]
    classic_elements=[]

    filtered_nodes, b = new_crack.filtering(Jarvis, all_ns)

    iter_vals = new_crack.ele_filtering(Elems_nodes, filtered_nodes)

    elems =[]
    #this loop extracts the elements from the dictionary and stores in elems
    for i in iter_vals:
        elems.append(list(i)[0])


    for i in filtered_nodes:
        for j in range(len(i)):
            for k in b:
                for l in range(len(k)):
                    if k[l] == i[j]:
                        i.append(i.index(i[j]))
        g.append(i)

    for i in g:
        ZZ = list(dict.fromkeys(i))
        influenced_elements.append(ZZ)

    ELE=[]

    for E, N in zip(influenced_elements, elems):
        if len(E) > 4:
            Ks, bravo = affected_enrichments.conditions(E, N, c_1, c_2, length_element, r, theta, alpha, D_plane_stress,
                                            GaussPoint_1to4)
            EEs.append(bravo)
            KEs.append(np.round(Ks,3))
            Jarvis.append(E)

        else:
            Class = enrichment_functions.classical_FE(E, GaussPoint_1to4, D_plane_stress)
            Jarvis.append(E)
            EEs.append(N)
            Sub_matrix = np.zeros([8,8])
            for j in Class:
                Sub_matrix += j
            KEs.append(np.round(Sub_matrix,3))

    to_assignment = []
    Total_Elements = filter(None.__ne__, EEs)
    for i in list(Total_Elements):
        GGs = list(dict.fromkeys(i))
        to_assignment.append(GGs)

    Total_Nodes = filter(None.__ne__, Jarvis)
    Total_Ks = filter(None.__ne__, KEs)

    # print(len(list(Total_Ks)))
    # for i in to_assignment:
    #     print("oooooooo",i)
    #     print("--------------------------")

    K_global = new_crack.connectivity_matrix(list(Total_Nodes), to_assignment, list(Total_Ks), len(NL), Heavys)

    new_crack.Boundary_conds(NL, A, B, length_element, K_global)













