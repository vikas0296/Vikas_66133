import numpy as np
import uniform_mesh; import matplotlib.pyplot as plt
import plots
import operator; import heaviside_enrichment
import Tip_enrichment; import math as m
import enrichment_functions; from scipy import linalg
from scipy import linalg

def updated_cracktip(NODES, ELEMENTS, c_2, GaussPoint_1to4, D_plane_stress, alpha):

    '''
    ==========================================================================
    This functions generates the mesh for the element which is partially cut
    by the crack.

      N_4                 N_5               N_6
    +-------------------+------------------+
    +                   +                  +
    +                   +                  +
    +$$$$$$$$$$$$$$$$$$$$(******)-> crack_tip
    +                   +                  +
    +                   +                  +
    +-------------------+------------------+
    N_1                 N_2                N_3

    Suppose, the element is partially cut by the crack(as in the above illustration)
    and therefore the ""length of the crack is << the element length"", hence,
    this function creates a subdomain same as the elements which are
    fully cut by the crack.
    This function separates the main element into 9 sub domains only in case of
    the length of the partial-crack << the element length

    ==========================================================================
    Parameters
    ---------------------------------------------------------------------------
    NODES :  list of 4 nodel coordinates.
    ELEMENTS : element list ex: [1,2,3,4]
    c_2 : crack_tip
    length_element : Element length
    GaussPoint_1to4 : Gauss points
    D_plane_stress : plane_stress relation
    alpha : angle made by the crack w.r.t x-axis

    Returns
    ---------------------------------------------------------------------------
    The output of the function is a nodal points which are used for Tip enrichment.
    Nodes_list : List of nodes, which is Tip_enriched
    Tip_matrix : element matrix for a tip containing element
    Tip_E : List of elements, which is Tip_enriched
    ===========================================================================
    '''
    r,theta =[],[]
    Nodes_list = np.array([NODES[0], NODES[1], NODES[2], NODES[3]])
    #calculation of distance r and theta from nodes to the crack tip
    r1, theta1 = Tip_enrichment.r_theta(Nodes_list[0], c_2, alpha)
    r2, theta2 = Tip_enrichment.r_theta(Nodes_list[1], c_2, alpha)
    r3, theta3 = Tip_enrichment.r_theta(Nodes_list[2], c_2, alpha)
    r4, theta4 = Tip_enrichment.r_theta(Nodes_list[3], c_2, alpha)
    ri = [r1,r2,r3,r4]
    ti = [theta1, theta2, theta3, theta4]
    # print(f"{NODES}------------ is full_tip_enriched")

    #division along x and y direction
    x=3
    #No of nodes, elements
    Nodes = (x+1)**2
    Elements = x**2
    # 4 nodes/element
    Nodes_elements = 4
    #calling uniform mesh.nodes to generate the coordinates
    domain_nodes = uniform_mesh.nodes(Nodes, Nodes_list, x)
    #calling uniform mesh.nodes to generate the element list
    domain_elements = uniform_mesh.elements(Elements, Nodes_elements, x)

    #calling uniform_mesh.dictionary to get a proper coordinates in ccw per element
    W_nodes, low = uniform_mesh.dictionary(domain_nodes,domain_elements)

    #to compute elemental stiffness matrix for 1 element, which consists of 9 sub elements
    Tip_matrix = Tip_enrichment.tip_enrichment(W_nodes, alpha, GaussPoint_1to4, D_plane_stress, c_2, r1, r2, r3, r4,
                                                                theta1, theta2, theta3, theta4)

    Tip_E = list(ELEMENTS)

    return Nodes_list, Tip_matrix, Tip_E, ri, ti


def updated_heaviside(NODES, ELEMENTS, GaussPoint_1to4, D_plane_stress, cracks):
    '''
    ==========================================================================
    This functions generates the mesh and element stiffness matrix
    for the element which is fully cut by the crack.

      N_4                 N_5               N_6
    +-------------------+------------------+
    +                   +                  +
    +                   +                  +
    (*******************************************)-> through crack
    +                   +                  +
    +                   +                  +
    +-------------------+------------------+
    N_1                 N_2                N_3

    Suppose, the element is fully cut by the crack(as in the above illustration)
    this function creates a subdomain same as the elements which are partially
    fully cut by the crack.
    This function separates the main element into 9 sub domains only in case of
    the length of the crack >> the element length
    ==========================================================================

    Parameters
    ----------
    NODES :  list of 4 nodel coordinates.
    ELEMENTS : element list ex: [1,2,3,4]
    cracks : list of all the crack coordinates
    length_element : Element length
    GaussPoint_1to4 : Gauss points
    D_plane_stress : plane_stress relation
    Returns
    -------
    Nodes_list : List of nodes, which is Tip_enriched
    LE : Element list which is heaviside enriched
    final_matrix_H : Final element stiffness matrix for an heaviside enriched element (s)
    '''

    Nodes_list = np.array([NODES[0], NODES[1], NODES[2], NODES[3]])
    #division along x and y direction
    x = 3
    #No of nodes, elements
    Nodes = (x+1)**2
    Elements = x**2
    # 4 nodes/element
    Nodes_elements = 4

    #calling uniform mesh.nodes to generate the coordinates
    domain_nodes = uniform_mesh.nodes(Nodes, Nodes_list, x)
    #calling uniform mesh.nodes to generate the element list
    domain_elements = uniform_mesh.elements(Elements, Nodes_elements, x)

    #calling uniform_mesh.dictionary to get a proper coordinates in ccw per element
    W_nodes, W_elements = uniform_mesh.dictionary(domain_nodes,domain_elements)

    #to compute elemental stiffness matrix for 1 element, which consists of 9 sub elements
    final_matrix_H = heaviside_enrichment.Heaviside_enrichment(W_nodes, GaussPoint_1to4, D_plane_stress, cracks)

    LE = list(ELEMENTS)
    return Nodes_list, LE, final_matrix_H

def updated_pretipenr(MIX_N, MIX_E, Hside, TS, GaussPoint_1to4, D_plane_stress, alpha, cracks):
    '''
    ==========================================================================
    This functions divides the main element to 9 sub domains for the element which is lying exactly
    behind the crack tip by the crack.

                         N_4                 N_5               N_6
                        +-------------------+------------------+
                        +                   +                  +
                        +                   +                  +
        pretip element<-($$$$$$$$$$$$$$$$$$$$)(******)-> crack_tip
                        +                   +                  +
                        +                   +                  +
                        +-------------------+------------------+
                        N_1                 N_2                N_3

    Suppose, the element is partially cut by the crack(as in the above illustration)
    and element which is located behind the crack tip element will be considered as the pre tip element
    These elements will have either tip enriched or heaviside enriched.
    For the above illustration, N_4 and N_1 will be heaviside enriched and N_5 and N_2 will be Tip enriched
    ==========================================================================

    Parameters
    ----------
    MIX_N : list of 4 nodel coordinates.
    MIX_E : list of 4 corner points corresponding to the element.
    Hside : Nodes that are heaviside enriched
    TS : Nodes that are tip enriched
    GaussPoint_1to4 : Gauss points
    D_plane_stress : plane_stress relation/ Plane_strain relation
    alpha : angle made by the crack w.r.t x-axis
    cracks : list of all the crack coordinates

    Returns
    -------
    HS : Updated heaviside enrichment nodes
    PT_mats : element stiffness matrix for the pre-tip elements
    '''

    if len(Hside) == 1:
        HS = Hside[0]

    else:
        H = []
        for i in Hside:
            H = H+i

        HS = []
        for i in H:
            if i not in HS:
                HS.append(i)


    PT_mats = []
    #looping through all the enriched nodal list and element points
    for NODES, ELEMENTS in zip(MIX_N, MIX_E):
        Nodes_list = np.array([NODES[0], NODES[1], NODES[2], NODES[3]])


        #calculation of distance r and theta from nodes to the crack tip
        r1, theta1 = Tip_enrichment.r_theta(Nodes_list[0], cracks[-1], alpha)
        r2, theta2 = Tip_enrichment.r_theta(Nodes_list[1], cracks[-1], alpha)
        r3, theta3 = Tip_enrichment.r_theta(Nodes_list[2], cracks[-1], alpha)
        r4, theta4 = Tip_enrichment.r_theta(Nodes_list[3], cracks[-1], alpha)

        #division along x and y direction
        x=3
        #No of nodes, elements
        Nodes = (x+1)**2
        Elements = x**2
        # 4 nodes/elemen
        Nodes_elements = 4

        #calling uniform mesh.nodes to generate the coordinates for the sub_elements
        domain_nodes = uniform_mesh.nodes(Nodes, Nodes_list, x)
        #calling uniform mesh.nodes to generate the element list for the sub_elements
        domain_elements = uniform_mesh.elements(Elements, Nodes_elements, x)
        #calling uniform_mesh.dictionary to get a proper coordinates in ccw per element
        S_nodes, low = uniform_mesh.dictionary(domain_nodes,domain_elements)

        E1, E2, E3, E4 = ELEMENTS[0], ELEMENTS[1], ELEMENTS[2], ELEMENTS[3]

        Hs =[]; Ts=[]
        matrices =[]
        #to compute elemental stiffness matrix for 1 element, which consists of 9 sub elements
        #looping through the sub_doman elements
        for coordinates in S_nodes:
            G,H,I,J = coordinates[0], coordinates[1], coordinates[2], coordinates[3]
            for GP in GaussPoint_1to4:
                #defining the Gauss points
                xi_1 = GP[0]
                xi_2 = GP[1]
                #initial values of heaviside functions(dummy values)
                H1, H2, H3, H4 = 0,0,0,0
                #extracting standard B-matrices, jacobi and enriched  B-matrix and values of the shape function
                B_std, Nan, N, dN, jacobi, dN_en = enrichment_functions.classic_B_matric(coordinates, D_plane_stress,
                                                                                          xi_1, xi_2, H1, H2, H3, H4)

                #conditions to check if the node is heaviside enriched or tip side enriched
                if E1 in HS:
                    # calling step_function to calculate the value of enrichment function (1 or 0 or -1)
                    H_1 =  enrichment_functions.step_function(G, cracks)
                    # calling individual b-matrix for the nodes to be heaviside enriched
                    BH1 = enrichment_functions.heaviside_function1(dN_en, H_1)
                    Hs.append(BH1)

                elif E1 in TS:
                    # calling asymptotic_functions to calculate the tip enrichment function values
                    F11, F21, F31, F41, dF1 = enrichment_functions.asymptotic_functions(r1, theta1, alpha)
                    # calling individual b-matrix for the nodes to be  tip enriched
                    T_1 = enrichment_functions.tip_enrichment_func_N1(F11, F21, F31, F41, dN, dF1, N)
                    Ts.append(T_1)

                else:
                    HS.append(E1)
                    # calling step_function to calculate the value of enrichment function (1 or 0 or -1)
                    H_1 =  enrichment_functions.step_function(G, cracks)
                    # calling individual b-matrix for the nodes to be heaviside enriched
                    BH1 = enrichment_functions.heaviside_function1(dN_en, H_1)
                    Hs.append(BH1)

                if E2 in HS:
                    # calling step_function to calculate the value of enrichment function (1 or 0 or -1)
                    H_2 =  enrichment_functions.step_function(H, cracks)
                    # calling individual b-matrix for the nodes to be heaviside enriched
                    BH2 = enrichment_functions.heaviside_function2(dN_en, H_2)
                    Hs.append(BH2)

                elif E2 in TS:
                    # calling asymptotic_functions to calculate the tip enrichment function values
                    F12, F22, F32, F42, dF2 = enrichment_functions.asymptotic_functions(r2, theta2, alpha)
                    # calling individual b-matrix for the nodes to be enriched
                    T_2 = enrichment_functions.tip_enrichment_func_N2(F12, F22, F32, F42, dN, dF2, N)
                    Ts.append(T_2)

                else:
                    HS.append(E2)
                    # calling step_function to calculate the value of enrichment function (1 or 0 or -1)
                    H_2 =  enrichment_functions.step_function(H, cracks)
                    # calling individual b-matrix for the nodes to be heaviside enriched
                    BH2 = enrichment_functions.heaviside_function2(dN_en, H_2)
                    Hs.append(BH2)

                if E3 in HS:
                    # calling step_function to calculate the value of enrichment function (1 or 0 or -1)
                    H_3 =  enrichment_functions.step_function(I, cracks)
                    # calling individual b-matrix for the nodes to be heaviside enriched
                    BH3 = enrichment_functions.heaviside_function3(dN_en, H_3)
                    Hs.append(BH3)

                elif E3 in TS:
                    # calling asymptotic_functions to calculate the tip enrichment function values
                    F13, F23, F33, F43, dF3 = enrichment_functions.asymptotic_functions(r3, theta3, alpha)
                    # calling individual b-matrix for the nodes to be enriched
                    T_3 = enrichment_functions.tip_enrichment_func_N3(F13, F23, F33, F43, dN, dF3, N)
                    Ts.append(T_3)

                else:
                    HS.append(E3)
                    # calling step_function to calculate the value of enrichment function (1 or 0 or -1)
                    H_3 =  enrichment_functions.step_function(I, cracks)
                    # calling individual b-matrix for the nodes to be heaviside enriched
                    BH3 = enrichment_functions.heaviside_function3(dN_en, H_3)
                    Hs.append(BH3)

                if E4 in HS:
                    # calling step_function to calculate the value of enrichment function (1 or 0 or -1)
                    H_4 =  enrichment_functions.step_function(J, cracks)
                    # calling individual b-matrix for the nodes to be heaviside enriched
                    BH4 = enrichment_functions.heaviside_function4(dN_en, H_4)
                    Hs.append(BH4)

                elif E4 in TS:
                    # calling asymptotic_functions to calculate the tip enrichment function values
                    F14, F24, F34, F44, dF4 = enrichment_functions.asymptotic_functions(r4, theta4, alpha)
                    # calling individual b-matrix for the nodes to be tip enriched
                    T_4 = enrichment_functions.tip_enrichment_func_N4(F14, F24, F34, F44, dN, dF4, N)
                    Ts.append(T_4)

                else:
                    HS.append(E4)
                    # calling step_function to calculate the value of enrichment function (1 or 0 or -1)
                    H_4 =  enrichment_functions.step_function(J, cracks)
                    # calling individual b-matrix for the nodes to be heaviside enriched
                    BH4 = enrichment_functions.heaviside_function4(dN_en, H_4)
                    Hs.append(BH4)

                #concatenating all the enriched B-matrix with std B-matrix
                for B in Hs:
                    B_std = np.concatenate((B_std, B), axis = 1)

                for T in Ts:
                    B_std = np.concatenate((B_std, T), axis = 1)

                # computation of element stiffness matrix
                Bt_D_TH = np.matmul(B_std.T, D_plane_stress)
                Bt_D_B_TH = (np.matmul(Bt_D_TH, B_std)) * linalg.det(jacobi)
                K_mixed_TH = Bt_D_B_TH
                F = K_mixed_TH.shape
                matrices.append(K_mixed_TH)
                Hs.clear()
                Ts.clear()

        PT_matrix = np.zeros([F[1],F[1]])
        for i in matrices:
            PT_matrix += i

        KP = np.round(PT_matrix,3)
        PT_mats.append(PT_matrix)

    return HS, PT_mats

def Normal_blended_en(NON_N, NON_E, HS, TS, GaussPoint_1to4, D_plane_stress, alpha, cracks):
    '''
    +++++++++++++++++++++++++++++++++++++
    +          +           +            +
    +****A********** B     +     C      +
    +          +           +            +
    +++++++++++++++++++++++++++++++++++++
    +          +           +            +
    +    D     +     E     +      F     +
    +          +           +            +
    +++++++++++++++++++++++++++++++++++++
    +          +           +            +
    +    G     +     H     +      I     +
    +          +           +            +
    +++++++++++++++++++++++++++++++++++++
    In the above illustraion, elements A and B contain the crack. Elements C,D,E,F will be partiallly enriched
    Elements G,H, and I will be considered as the unenriched elements
    Elements D,E,F, and C are called blended elements

    Parameters
    ----------
    NON_N : the list of nodes which contains both partially enriched nodes and unenriched nodes.
    NON_E : the list of elements which contains both partially enriched elements and unenriched elements.
    HS : The list of nodes that have been heaviside enriched
    TS : The list of nodes that have been tip enriched
    GaussPoint_1to4 : Usual Gauss points
    D_plane_stress : plane stress relation
    alpha : angle made by the crack w.r.t x-axis
    cracks : list of all the crack coordinates

    Returns
    -------
    Normal_K : unenriched elemental stiffness matrix list
    Blended_K : partially enriched elemental stiffness matrix list
    Normal_E : unenriched element list
    Blended_E : partially enriched element list
    Normal_N : unenriched nodal lists
    Blended_N : partially enriched nodal list
    The function outputs the element stiffness matrices for normal elements and partially enriched eements
    '''
    Normal_N =[]
    Normal_E =[]
    Normal_K=[]
    Blended_K =[]
    Blended_E =[]
    Blended_N =[]
    #looping through all the nodal list in which  maximum of 3 nodes could potentially
    #be heaviside or tip enriched or neither
    for NODES, ELEMENTS in zip(NON_N, NON_E):

        Nodes_list = np.array([NODES[0], NODES[1], NODES[2], NODES[3]])
        # calculation of distance r and theta from nodes to the crack tip
        r1, theta1 = Tip_enrichment.r_theta(Nodes_list[0], cracks[-1], alpha)
        r2, theta2 = Tip_enrichment.r_theta(Nodes_list[1], cracks[-1], alpha)
        r3, theta3 = Tip_enrichment.r_theta(Nodes_list[2], cracks[-1], alpha)
        r4, theta4 = Tip_enrichment.r_theta(Nodes_list[3], cracks[-1], alpha)

        E1, E2, E3, E4 = ELEMENTS[0], ELEMENTS[1], ELEMENTS[2], ELEMENTS[3]

        H_mat =[]; T_mat=[]
        matrices =[]
        G,H,I,J = NODES[0], NODES[1], NODES[2], NODES[3]
        #selecting the nodes which is not being either heaviside enriched or Tip_enriched
        # (Normal elements/unenriched elements)
        if ((E1 not in TS and E1 not in HS) and (E2 not in TS and E2 not in HS)
            and (E3 not in TS and E3 not in HS) and (E4 not in TS and E4 not in HS)) and NODES not in Normal_N:
                #calling std classical FE, B-matrix to calculate elemental stiffness matrix
                C_M = enrichment_functions.classical_FE(NODES, GaussPoint_1to4, D_plane_stress)
                KP = np.round(C_M,3)
                Normal_E.append(ELEMENTS)
                Normal_N.append(NODES)
                Normal_K.append(C_M)

        else:
            #to compute elemental stiffness matrix for 1 element, which could potentially contain an enriched node
            Blended_E.append(ELEMENTS)
            Blended_N.append(NODES)

            for GP in GaussPoint_1to4:
                # defining the Gauss points
                xi_1 = GP[0]
                xi_2 = GP[1]
                #initial values of heaviside functions(dummy values)
                H1, H2, H3, H4 = 0,0,0,0
                # extracting standard B-matrices, jacobi and enriched  B-matrix and values of the shape function
                B_std, Null, N, dN, jacobi, dN_en = enrichment_functions.classic_B_matric(NODES, D_plane_stress,
                                                                                          xi_1, xi_2, H1, H2, H3, H4)

                #conditions to check if the node is heaviside enriched or tip side enriched
                if E1 in HS:
                    # calling step_function to calculate the value of enrichment function (1 or 0 or -1)
                    H_1 =  enrichment_functions.step_function(G, cracks)
                    # calling individual b-matrix for the nodes to be heaviside enriched
                    BH1 = enrichment_functions.heaviside_function1(dN_en, H_1)
                    H_mat.append(BH1)

                elif E1 in TS:
                    # calling asymptotic_functions to calculate the tip enrichment function values
                    F11, F21, F31, F41, dF1 = enrichment_functions.asymptotic_functions(r1, theta1, alpha)
                    # calling individual b-matrix for the nodes to be tip enriched
                    T_1 = enrichment_functions.tip_enrichment_func_N1(F11, F21, F31, F41, dN, dF1, N)
                    T_mat.append(T_1)


                if E2 in HS:
                    # calling step_function to calculate the value of enrichment function (1 or 0 or -1)
                    H_2 =  enrichment_functions.step_function(H, cracks)
                    # calling individual b-matrix for the nodes to be heaviside enriched
                    BH2 = enrichment_functions.heaviside_function2(dN_en, H_2)
                    H_mat.append(BH2)

                elif E2 in TS:
                    # calling asymptotic_functions to calculate the tip enrichment function values
                    F12, F22, F32, F42, dF2 = enrichment_functions.asymptotic_functions(r2, theta2, alpha)
                    # calling individual b-matrix for the nodes to be tip enriched
                    T_2 = enrichment_functions.tip_enrichment_func_N2(F12, F22, F32, F42, dN, dF2, N)
                    T_mat.append(T_2)

                if E3 in HS:
                    # calling step_function to calculate the value of enrichment function (1 or 0 or -1)
                    H_3 =  enrichment_functions.step_function(I, cracks)
                    # calling individual b-matrix for the nodes to be heaviside enriched
                    BH3 = enrichment_functions.heaviside_function3(dN_en, H_3)
                    H_mat.append(BH3)

                elif E3 in TS:
                    # calling asymptotic_functions to calculate the tip enrichment function values
                    F13, F23, F33, F43, dF3 = enrichment_functions.asymptotic_functions(r3, theta3, alpha)
                    # calling individual b-matrix for the nodes to be tip enriched
                    T_3 = enrichment_functions.tip_enrichment_func_N3(F13, F23, F33, F43, dN, dF3, N)
                    T_mat.append(T_3)

                if E4 in HS:
                    # calling step_function to calculate the value of enrichment function (1 or 0 or -1)
                    H_4 =  enrichment_functions.step_function(J, cracks)
                    # calling individual b-matrix for the nodes to be heaviside enriched
                    BH4 = enrichment_functions.heaviside_function4(dN_en, H_4)
                    H_mat.append(BH4)

                elif E4 in TS:
                    # calling asymptotic_functions to calculate the tip enrichment function values
                    F14, F24, F34, F44, dF4 = enrichment_functions.asymptotic_functions(r4, theta4, alpha)
                    # calling individual b-matrix for the nodes to be tip enriched
                    T_4 = enrichment_functions.tip_enrichment_func_N4(F14, F24, F34, F44, dN, dF4, N)
                    T_mat.append(T_4)

                #concatenating all the enriched B-matrix with std B-matrix
                for B in H_mat:
                    B_std = np.concatenate((B_std, B), axis = 1)

                for T in T_mat:
                    B_std = np.concatenate((B_std, T), axis = 1)

                # computation of element stiffness matrix
                Bt_D_TH = np.matmul(B_std.T, D_plane_stress)
                Bt_D_B_TH = (np.matmul(Bt_D_TH, B_std)) * linalg.det(jacobi)
                K_mixed_TH = Bt_D_B_TH
                F = K_mixed_TH.shape
                matrices.append(K_mixed_TH)
                H_mat.clear()
                T_mat.clear()
            #each element will have 4 Gauss points, therefore 4 stiffness matrices are generated and
            #all the stiffness matrices will be summed up
            Enr_K = np.zeros([F[1],F[1]])
            for i in matrices:
                Enr_K += i
            Blended_K.append(Enr_K)

    return Normal_K, Blended_K, Normal_E, Blended_E, Normal_N, Blended_N
