import numpy as np
import operator; import new_crack;
import math; import heaviside_enrichment
import matplotlib.pyplot as plt; import Assignment
import class_crack; import plots
import affected_enrichments; from scipy import linalg
import enrichment_functions; import Displacement_approx
import Stress_Strains; import Damage_mechanics


def dictionary(NL, EL):

    y = []
    for j in range(len(EL)):
        for i in EL[j]:
            y.append(tuple(NL[int(i-1)]))
    '''
    ===========================================================================
    Formation of list of 4 nodes per element in ccw direction
    ===========================================================================
    '''
    all_ns = []
    Elems_nodes = []
    for k in range(len(EL)):
        l_dict = {tuple(EL[k]): y[k*4: k*4+4]}
        Elems_nodes.append(l_dict)
        z = operator.itemgetter(*l_dict.keys())(l_dict)
        all_ns.append(z)
    return all_ns, Elems_nodes


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
    Elements : total numbe rof elements required
    Nodes_elements : 4, as each element will have 4 nodes
    x : number of elements per row
    Returns
    --------------------------
    EL : list of elements
    '''
    EL = np.zeros([Elements, Nodes_elements])
    for i in range(1, x+1):
        for j in range(1, x+1):
            if j == 1:

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
    round_NL = NL
    return round_NL, EL

if __name__ == "__main__":

    A = 10          #length of the geometry along x and y axis
    B = 10
    x = 6                                    #3x3 will be elements, 4x4 will be nodes
    NL, EL = uniform_mesh(A, B, x)
    x1 = NL[0]
    x2 = NL[1]

    '''
    length of a segment, given 2 points = sqrt((x2-x1)**2 + (y2-y1)**2)
    '''
    length_element = np.round(np.sqrt((x2[0]-x1[0])**2 + (x2[1]-x1[1])**2),4)
    print("the length of each element is ", length_element)
    all_ns, Elems_nodes = dictionary(NL,EL)
    elements = [ ]
    for j in Elems_nodes:
        elements.append(list(j.keys())[0])

    plots.plot_mesh(NL, A, B, x)

    #=============================Material_properties===========================================

    D = 205000
    nu = 0.3
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
    cracktip_1 = np.array([1, 6])
    cracktip_2 = np.array([4.8, 4.8])

    plots.plot_crack(cracktip_1, cracktip_2, length_element)
    crack_length = round(abs(cracktip_1[0]-cracktip_1[1]), 3)
    print("the length of the crack is: ", crack_length)

    c_1 = np.array([cracktip_1[0], cracktip_2[0]])
    c_0 = np.copy(c_1)
    c_2 = np.array([cracktip_1[1], cracktip_2[1]])


    r, theta = Assignment.to_polar_in_radians(c_2[0], c_2[1])
    alpha = 0
    gamma, Jarvis = [], []

    # K_matrices holder
    KEs, EEs = [], []

    #holder for collecting sub-domain elements and nodes
    Upper_D, Lower_D, Tip_elem, Tip_node = [], [], [], []
    Sub_U, Sub_L, Partial, Partial_node =  [], [], [], []
    Heavy_U, Heavy_L, Heavy, Heavy_node =  [], [], [], []

    #need to fix the else condition

    if c_1[1] in NL[:,1]:
        # print(NL, c_1)
        print("Please change the coordinates of the crack, it might be grazing the edge of the element")


    elif c_2[0] >= A:
        print(f"The crack_length {crack_length} is >= to the geometry length {A},"
                           "therefore the sample is broken into two halves")

    elif crack_length <= length_element:
         print("the crack exists in only one element hence it will be tip enriched")
         for node, el_co in zip(all_ns, elements) :

            Node_list, foxtrot, delta, Tip_U, Tip_L, Tip_Elem = class_crack.domain_splitter(node, el_co, c_1,
                                                                c_2, length_element,GaussPoint_1to4, D_plane_stress)

            Jarvis.append(Node_list)
            KEs.append(foxtrot)
            EEs.append(delta)
            Upper_D.append(Tip_U)
            Lower_D.append(Tip_L)
            Tip_elem.append(Tip_Elem)
            Tip_node.append(Node_list)

    else:
         splits = np.arange(c_1[0], c_2[0], length_element)
         idx = [0]
         W = np.delete(splits, idx)
         # print(W)
         new_points = [0,0]
         for i in W:
            new_points = [0,0]
            new_points[0] = round(i,2)
            new_points[1] = c_1[1]
            gamma.append(new_points)

         gamma.insert(0, list(c_0))
         gamma.insert(len(gamma), list(c_2))
         v_1 = gamma[-1]
         v_2 = gamma[-2]

         if abs(v_1[0] - v_2[0]) == 0:
             gamma.remove(v_1)

         p1 = gamma[-1]
         p2 = gamma[-2]
         if round(abs(p1[0] - p2[0]),2) <= length_element:
              # print(p1, p2)
              for i,j in zip(all_ns, elements):

                epsilon, K, E, Partial_UD, Partial_LD, Partial_Ele = class_crack.domain_splitter(i, j, p2, p1, length_element,
                                                                        GaussPoint_1to4, D_plane_stress)

                Jarvis.append(epsilon)
                KEs.append(K)
                EEs.append(E)
                Sub_U.append(Partial_UD)
                Sub_L.append(Partial_LD)
                Partial.append(Partial_Ele)
                Partial_node.append(epsilon)

         for i in range(len(gamma)-2):
            # print(gamma[i], gamma[i+1])
            N_List, K_mats, Ele_coords, Heavy_UD, Heavy_LD, Heavy_Ele =  class_crack.C_length_E_length(all_ns, elements,gamma[i], gamma[i+1],
                                                        crack_length, length_element, c_2, GaussPoint_1to4,
                                                                        r, theta, alpha, D_plane_stress)

            # print(N_List)
            Jarvis.append(N_List)
            KEs.append(K_mats)
            EEs.append(Ele_coords)
            Heavy_U.append(Heavy_UD)
            Heavy_L.append(Heavy_LD)
            Heavy.append(Heavy_Ele)
            Heavy_node.append(N_List)

    # #==========================================================================
    # # program to extract the affected elements due to enrichment

    g=[]
    influenced_elements=[]
    classic_elements=[]
    affect_node = [ ]

    filtered_nodes, b = Assignment.filtering(Jarvis, all_ns)

    iter_vals = Assignment.ele_filtering(Elems_nodes, filtered_nodes)

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


    for E, N in zip(influenced_elements, elems):
        if len(E) > 4:
            Ks, bravo = affected_enrichments.conditions(E, N, c_1, c_2, length_element, r, theta,
                                                        alpha, D_plane_stress,GaussPoint_1to4)

            EEs.append(bravo)
            KEs.append(Ks)
            Jarvis.append(E)
            affect_node.append(E)

        else:
            Class = enrichment_functions.classical_FE(E, GaussPoint_1to4, D_plane_stress)
            Sub_matrix = np.zeros([8,8])

            for i in Class:
                Sub_matrix += i
            KEs.append(Sub_matrix)
            Jarvis.append(E)
            EEs.append(N)
            affect_node.append(E)

    to_assignment = []
    Total_Elements = filter(None.__ne__, EEs)
    for i in list(Total_Elements):
        GGs = list(dict.fromkeys(i))
        to_assignment.append(GGs)

    Total_Nodes = filter(None.__ne__, Jarvis)
    Total_Ks = filter(None.__ne__, KEs)


    K_global, C_Dofs, Tside, Hside, Total_Dofs = new_crack.connectivity_matrix(list(Total_Nodes),
                                                                            to_assignment, list(Total_Ks), len(NL))

    #-----------------------for a fatigue problem a maximum and munimum load is applied---------------------------------

    #Fatigue stress_ratio = σ min /σ max

    stress_ratio = 0.1
    force1 = 1000
    force2 = stress_ratio * force1
    displacement_vector1, displacement_vector2, BCs = new_crack.Boundary_conds(NL, A, B, length_element, K_global, force1, force2)

#################################### for force1 ################################################################

    for j in BCs:
        displacement_vector1 = np.insert(displacement_vector1, j, 0)
        displacement_vector2 = np.insert(displacement_vector2, j, 0)

#-------------------------------single element and it is Tip enriched --------------------------------------------------
    # #stress_strain for partially cut or fully cut element if crack length < element length

    Upper_Lower = Upper_D + Lower_D
    U_L =  filter(None.__ne__, Upper_Lower)
    T_elem = filter(None.__ne__, Tip_elem)
    T_node = filter(None.__ne__, Tip_node)
    STRESS1 = []
    STRESS2 = []
    T_n = []
    T_e =[]
    upperlower = []

    for i in list(T_node):
        T_n.append(i)

    for j in list(U_L):
        for k in j:
            upperlower.append(k)

    for E in list(T_elem):
        T_e.append(E)

    Tip_displacements1 = Displacement_approx.displacement_approximation(T_n, c_1, c_2, T_e, r, theta,
                                              alpha, Tside, Hside, C_Dofs, displacement_vector1)

    Tip_displacements2 = Displacement_approx.displacement_approximation(T_n, c_1, c_2, T_e, r, theta,
                                              alpha, Tside, Hside, C_Dofs, displacement_vector2)

    Tip_stress1 = Stress_Strains.strain_stress_enr(Tip_displacements1, upperlower, GaussPoint_1to4, D_plane_stress)
    Tip_stress2 = Stress_Strains.strain_stress_enr(Tip_displacements2, upperlower, GaussPoint_1to4, D_plane_stress)
    STRESS1.append(Tip_stress1)
    STRESS2.append(Tip_stress2)


#-----------------------------If the crack length is more than element length-------------------------------------
    # #stress_strain for partially cut or fully cut element if crack length > element length
    P = []
    Q = []
    S_U = filter(None.__ne__, Sub_U)
    S_L = filter(None.__ne__, Sub_L)
    P_elm = filter(None.__ne__, Partial)
    P_node = filter(None.__ne__, Partial_node)
    P_n =[]
    P_e =[]

    for i in list(P_node):
        P_n.append(i)

    for k in list(P_elm):
        P_e.append(k)


    P_displacements1 = Displacement_approx.displacement_approximation(P_n, c_1, c_2, P_e, r, theta,
                                                        alpha, Tside, Hside, C_Dofs, displacement_vector1)

    P_displacements2 = Displacement_approx.displacement_approximation(P_n, c_1, c_2, P_e, r, theta,
                                                        alpha, Tside, Hside, C_Dofs, displacement_vector2)


    for i,j in zip(list(S_U), list(S_L)):
        Q.append(i)
        Q.append(j)
        Partial_stress_U1 = Stress_Strains.strain_stress_enr(P_displacements1, i, GaussPoint_1to4, D_plane_stress)
        Partial_stress_L1= Stress_Strains.strain_stress_enr(P_displacements1, j, GaussPoint_1to4, D_plane_stress)
        STRESS1.append(Partial_stress_U1)
        STRESS1.append(Partial_stress_L1)

        Partial_stress_U2 = Stress_Strains.strain_stress_enr(P_displacements2, i, GaussPoint_1to4, D_plane_stress)
        Partial_stress_L2 = Stress_Strains.strain_stress_enr(P_displacements2, j, GaussPoint_1to4, D_plane_stress)
        STRESS2.append(Partial_stress_U2)
        STRESS2.append(Partial_stress_L2)
#-------------------------------------------------------------------------------------------------------------------------
    # stress_strain for heaviside enriched elements, if crack length > element length

    H_U = filter(None.__ne__, Heavy_U)
    H_L = filter(None.__ne__, Heavy_L)
    H_elm = filter(None.__ne__, Heavy)
    H_node = filter(None.__ne__, Heavy_node)
    H_n, H_e = [], []

    for i in list(H_node):
        H_n.append(i)

    for j in list(H_elm):
        H_e.append(j)

    H_displacements1 = Displacement_approx.displacement_approximation(H_n, c_1, c_2, H_e, r, theta,
                                                          alpha, Tside, Hside, C_Dofs, displacement_vector1)

    H_displacements2 = Displacement_approx.displacement_approximation(H_n, c_1, c_2, H_e, r, theta,
                                                          alpha, Tside, Hside, C_Dofs, displacement_vector2)

    for i,j in zip(list(H_U), list(H_L)):
        P.append(i)
        P.append(j)
        Heavy_stress_U1 = Stress_Strains.strain_stress_enr(H_displacements1, i, GaussPoint_1to4, D_plane_stress)
        Heavy_stress_L1 = Stress_Strains.strain_stress_enr(H_displacements1, j, GaussPoint_1to4, D_plane_stress)
        STRESS1.append(Heavy_stress_U1)
        STRESS1.append(Heavy_stress_L1)

        Heavy_stress_U2 = Stress_Strains.strain_stress_enr(H_displacements2, i, GaussPoint_1to4, D_plane_stress)
        Heavy_stress_L2 = Stress_Strains.strain_stress_enr(H_displacements2, j, GaussPoint_1to4, D_plane_stress)
        STRESS2.append(Heavy_stress_U2)
        STRESS2.append(Heavy_stress_L2)


# #--------------------------------stress_strain for the unenriched elements-------------------------------------------------

    A_displacement1 = Displacement_approx.displacement_approximation(influenced_elements, c_1, c_2, elems, r, theta,
                                                            alpha, Tside, Hside, C_Dofs, displacement_vector1)

    A_displacement2 = Displacement_approx.displacement_approximation(influenced_elements, c_1, c_2, elems, r, theta,
                                                            alpha, Tside, Hside, C_Dofs, displacement_vector2)

    A_stress1 = Stress_Strains.strain_stress_tensors(A_displacement1,influenced_elements, GaussPoint_1to4, D_plane_stress)
    STRESS1.append(A_stress1)

    A_stress2 = Stress_Strains.strain_stress_tensors(A_displacement2,influenced_elements, GaussPoint_1to4, D_plane_stress)
    STRESS2.append(A_stress2)


    H_store1 =[]
    for i in P:
        H_store1 += i

    for k in Q:
        H_store1 += k

    for m in upperlower:
        H_store1 += m

    SE_ME = affect_node + H_store1
    Assignment.G_points(SE_ME)

    sigma1 = []
    for i in STRESS1:
        for j in i:
            sigma1.append(j)

    # Damage_mechanics.mechanics_damage(sigma1)
    # DBs = plots.deformation_plots(NL, displacement_vector1)

#-----------------------------If the crack length is more than element length-------------------------------------
    #stress_strain for partially cut or fully cut element if crack length > element length

#--------------------------------stress_strain for the unenriched elements-------------------------------------------------



# #-----------------------------------------------------------------------------------------------------------------------------

    # print(STRESS2)

    sigma2 = []
    for k in STRESS2:
        for j in k:
            sigma2.append(j)

    print(len(sigma2))
    print(len(sigma1))
    Damage_mechanics.mechanics_damage(sigma1, sigma2)


#plots of deformed shape and Gauss points==================


    # DBs = plots.deformation_plots(NL, displacement_vector2)


# #---------------------------------stress_strain for the whole geometry---------------------------------------------------


#     Uxy2 = Displacement_approx.displacement_approximation(list(Just_Nodes), c_1, c_2, to_assignment, r, theta, alpha,
#                                                           Tside, Hside, C_Dofs, displacement_vector2)

#---------------------------------stress_strain for the whole geometry---------------------------------------------------

    # Just_Nodes = filter(None.__ne__, Jarvis)
    # for k in Just_Nodes:
    #     while len(k) != 4:
    #         k.pop()

#     Uxy = Displacement_approx.displacement_approximation(list(Just_Nodes), c_1, c_2, to_assignment, r, theta, alpha,
#                                                           Tside, Hside, C_Dofs, displacement_vector1)
