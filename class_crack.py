import numpy as np
import uniform_mesh
import matplotlib.pyplot as plt
import plots
import new_crack
import operator
import heaviside_enrichment
import Assignment
import Tip_enrichment
import Mixed_enrichment
import enrichment_functions

def domain_splitter(nodes, elems, c_1, c_2, length_element, GaussPoint_1to4, D_plane_stress):
    '''
    ==========================================================================
    This functions generates the mesh for the element which is partially cut
    by the crack.
     N_4                 N_5               N_6
    +-------------------+------------------+
    +                   +                  +
    +                   +                  +
    +$$$$$$$$$$$$$$$$$$$$(******)-> partial_crack
    +                   +                  +
    +                   +                  +
    +-------------------+------------------+
    N_1                 N_2                N_3

    Suppose, the element is partially cut by the crack(as in the above illustration)
    and therefore the ""length of the crack is << the element length"", hence,
    this function creates a subdomain same as the elements which are
    fully cut by the crack.
    This function separates the main element into 2 sub domains only in case of
    the length of the partial-crack << the element length
    ==========================================================================
    Parameters
    ---------------------------------------------------------------------------
    nodes :  list of 4 nodel coordinates.
    c_1 : crack_tip 1
    c_2 : crack_tip 2
    length_element : Element length
    Returns
    ---------------------------------------------------------------------------
    The output of the function is a nodal points which are used for Tip enrichment.
    ===========================================================================
    '''
    NL_1 = nodes[0]
    NL_2 = nodes[1]
    NL_3 = nodes[2]
    NL_4 = nodes[3]

    Tip_matrix_L = np.zeros([40, 40])
    Tip_matrix_U = np.zeros([40, 40])
    domain_elements=[]
    if (c_2[0] > NL_1[0] and c_2[0] <= NL_2[0] and c_2[1] <= NL_3[1] and c_2[1]<=NL_4[1] and
        c_2[1]> NL_1[1]):
        Nodes_list = [(NL_1),(NL_2),(NL_3),(NL_4)]

        '''
        array: outer nodes of the element
        '''
        x=2
        Nodes = (x+1)**2
        Elements = x**2
        Nodes_elements = 4
        c_add = [0,0]
        c_add[0] = c_1[0] + length_element
        c_add[1] = c_1[1]


        Lower_domain = np.array([nodes[0], nodes[1], c_add, c_1]) #lower domain elements
        Upper_domain = np.array([c_1,c_add,nodes[2], nodes[3]])   #upper domain elements

        domain_elements.append(Lower_domain)
        domain_elements.append(Upper_domain)

        for d in domain_elements:
            '''fox: inner nodes of the upper and lower domain
            '''
            fox = uniform_mesh.nodes(Nodes,d,x)
            plots.plot_sub_elements(fox)

        uppernodes = uniform_mesh.nodes(Nodes,Upper_domain,x)
        upperelements = (uniform_mesh.elements(Elements, Nodes_elements,x))


        r, theta = Assignment.to_polar_in_radians(c_2[0], c_2[1])

        lowernodes = uniform_mesh.nodes(Nodes,Lower_domain,x)
        lowerelements = uniform_mesh.elements(Elements, Nodes_elements,x)

        Nodes_L, low = uniform_mesh.dictionary(lowernodes,lowerelements)

        Nodes_U, up = uniform_mesh.dictionary(uppernodes,upperelements)
        L_U = (Nodes_L + Nodes_U)
        alpha = 0

        final_matrix_L = Tip_enrichment.tip_enrichment(L_U, r, theta, alpha, GaussPoint_1to4, D_plane_stress)
        # print(f"{Nodes_list} is full_tip_enriched")
        for i in final_matrix_L:
            Tip_matrix_L += i

        # all the 4 nodes are tip enriched and its a default result
        Z = list(elems)
        A = ["0tip_enriched", "1tip_enriched", "2tip_enriched", "3tip_enriched"]   #used to define the assignment matrix

        return  Nodes_list, Tip_matrix_L, Z+A

    else:
        return None, None, None

def Crack_length_PP_element_length(nodes, elems, c_1, c_2, crack_length, length_element, v_3,
                                   GaussPoint_1to4, r, theta, alpha, D_plane_stress):

    '''
    ===========================================================================
    This function will evaluate if the element contains a
    crack and the crack length should be of the same length as that of
    the element and it returns the list of the nodes to be enriched.
    ===========================================================================
    Parameters
    ----------
    nodes : list of 4 nodel coordinates.
    c_1 : coordinates of crack_tip 1.
    c_2 : coordinates of crack_tip 2.
    counter : integer value to keep track of the element where crack exist.
    Returns
    -------
    nodes : list of 4 nodes which require enrichment
    '''
    for i, j in zip(nodes, elems):
        NL_1 = i[0]
        NL_2 = i[1]
        NL_3 = i[2]
        NL_4 = i[3]

        dist_1 = np.sqrt((NL_1[0]-c_1[0])**2 + (NL_1[1]-c_1[1])**2)
        dist_2 = np.sqrt((NL_2[0]-c_2[0])**2 + (NL_2[1]-c_2[1])**2)

        if  dist_1 < length_element and dist_2 < length_element and c_1[1] > NL_1[1] and c_2[1] > NL_2[1]:
            Nodes_list = [(NL_1),(NL_2),(NL_3),(NL_4)]

            sub_elements = []
            Lower_domain = np.array([Nodes_list[0], Nodes_list[1], c_2, c_1])
            Upper_domain = np.array([c_1, c_2, Nodes_list[2], Nodes_list[3]])

            sub_elements.append(Lower_domain)
            sub_elements.append(Upper_domain)

            x = 2
            Nodes = (x+1)**2
            Elements = x**2
            Nodes_elements = 4

            lowernodes = uniform_mesh.nodes(Nodes,Lower_domain,x)
            round_Lnodes = lowernodes
            lowerelements = uniform_mesh.elements(Elements, Nodes_elements,x)

            uppernodes = uniform_mesh.nodes(Nodes, Upper_domain, x)
            round_Unodes = uppernodes
            upperelements = uniform_mesh.elements(Elements, Nodes_elements,x)

            Nodes_U, up = uniform_mesh.dictionary(round_Unodes, upperelements)
            Nodes_L, low = uniform_mesh.dictionary(round_Lnodes, lowerelements)

            for m in sub_elements:
                '''
                nodes created after the sub_division
                '''
                zeta = uniform_mesh.nodes(Nodes,m,x)
                plots.plot_sub_elements(zeta)


            S1 = Nodes_list[2]
            S2 = Nodes_list[0]
            V = tuple(v_3)
            if abs(S1[0] - V[0]) < length_element:
                # print(f"{Nodes_list} is Mixed_enrichment")
                mixed_matrix = Mixed_enrichment.mixed_enrichment(Nodes_U, Nodes_L, r, theta, alpha,
                                                                      GaussPoint_1to4, D_plane_stress)

                Q = list(j)
                # print("mixed elem",Q)
                B = ["0heaviside_enriched", "1tip_enriched", "2tip_enriched", "3heaviside_enriched"]

                return Nodes_list, mixed_matrix, Q+B

            else:
                Heavy_matrix = heaviside_enrichment.Heaviside_enrichment(Nodes_L, Nodes_U, GaussPoint_1to4,
                                                                          D_plane_stress)
                R = list(j)
                # print("heavy elems", R)
                V = ["0heaviside_enriched", "1heaviside_enriched", "2heaviside_enriched", "3heaviside_enriched"]
                return Nodes_list, Heavy_matrix, R+V

#     =========================================================================
#     This function splits the length of the crack in equal proportion as
#     that of the length of the element.
#     This function is called only in case of crack_length >> element_length
#     =========================================================================

#     N_4                 N_5               N_6
#     +-------------------+------------------+
#     +                   +                  +
#     +                   +                  +
#     @-------------------@----@ ~>crack     +
#     +                   +                  +
#     +                   +                  +
#     +-------------------+------------------+
#     N_1                 N_2                N_3
#     For the above illustration, the function outputs "@"-coordinates


