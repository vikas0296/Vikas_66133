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
        print(elems)
        print(Tip_matrix_L[0:5,0:5])

        return  Nodes_list, Tip_matrix_L, Z+A

    else:
        return None, None, None


def sub_nodes_liste(a, elems, crack_length, length_element, c_1, c_2, v_3, r, theta, alpha, GaussPoint_1to4, D_plane_stress):

    '''
    ===========================================================================
    The output of this function is used to plot the sub domains of an enriched
    element. This function establishes the upper domain and lower domain.
    ===========================================================================
    Parameters
    ----------
    a : an array 4 nodes to be enriched
    '''

    sub_elements = []
    Lower_domain = np.array([a[0], a[1], c_2, c_1])
    Upper_domain = np.array([c_1, c_2, a[2], a[3]])

    sub_elements.append(Lower_domain)
    sub_elements.append(Upper_domain)

    x = 2
    Nodes = (x+1)**2
    Elements = x**2
    Nodes_elements = 4

    lowernodes = uniform_mesh.nodes(Nodes,Lower_domain,x)
    lowerelements = uniform_mesh.elements(Elements, Nodes_elements,x)

    uppernodes = uniform_mesh.nodes(Nodes,Upper_domain,x)
    upperelements = uniform_mesh.elements(Elements, Nodes_elements,x)

    Nodes_U, up = uniform_mesh.dictionary(uppernodes,upperelements)
    Nodes_L, low = uniform_mesh.dictionary(lowernodes,lowerelements)

    '''
    =======================================================================
    in Assignment file "Heaviside enrichment" function has been called
    '''
    for m in sub_elements:
        '''
        nodes created after the sub_division
        '''
        zeta = uniform_mesh.nodes(Nodes,m,x)
        plots.plot_sub_elements(zeta)

    S1 = a[2]
    S2 = a[1]
    V = tuple(v_3)
    if abs(S1[0] - V[0]) < 1e-3:
        # print(a, True)
        # print(f"{a} is Mixed_enrichment")
        mixed_matrix = Mixed_enrichment.mixed_enrichment(a, Nodes_U, Nodes_L, r, theta, alpha,
                                                             GaussPoint_1to4, D_plane_stress)

        # print("mixed", mixed_matrix[0:5,0:5])
        # all the 4 nodes are heaviside enriched and its a default possibility
        Q = list(elems)
        print(elems)
        print(mixed_matrix[0:10,0:10])
        B = ["0heaviside_enriched", "1tip_enriched", "2tip_enriched", "3heaviside_enriched"]

        return mixed_matrix, Q+B

    else:
        H2 = -1
        final_matrix_L = Assignment.set_matrix(np.round(Nodes_L, 3), H2, GaussPoint_1to4, D_plane_stress)
        H3 = 1
        final_matrix_U = Assignment.set_matrix(np.round(Nodes_U, 3), H3, GaussPoint_1to4, D_plane_stress)
        print(final_matrix_L.shape)
        Element_matrix_Full = np.add(final_matrix_L, final_matrix_U)
        # print(f"{a} is heaviside_enriched")

        # all the 4 nodes are heaviside enriched and its a default possibility
        Z = list(elems)
        print(elems)
        print(Element_matrix_Full[0:10,0:10])
        C = ["0heaviside_enriched", "1heaviside_enriched", "2heaviside_enriched", "3heaviside_enriched"]

        return Element_matrix_Full, Z

def Crack_length_PP_element_length(nodes, elems, c_1, c_2, crack_length, length_element, v_3, GaussPoint_1to4, r, theta,
                                     alpha, D_plane_stress):

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
    NL_1 = nodes[0]
    NL_2 = nodes[1]
    NL_3 = nodes[2]
    NL_4 = nodes[3]

    dist_1 = np.sqrt((NL_1[0]-c_1[0])**2 + (NL_1[1]-c_1[1])**2)
    dist_2 = np.sqrt((NL_2[0]-c_2[0])**2 + (NL_2[1]-c_2[1])**2)

    if  dist_1 < length_element and dist_2 < length_element and c_1[1] > NL_1[1] and c_2[1] > NL_2[1]:
        Nodes_list = [(NL_1),(NL_2),(NL_3),(NL_4)]

        Matrices, element = sub_nodes_liste(Nodes_list, elems, crack_length, length_element, c_1, c_2, v_3,
                                   r, theta, alpha, GaussPoint_1to4, D_plane_stress)

        # new_crack.list_of_Elements(sub_list)
        return Nodes_list, Matrices, element

    else:
        return None, None, None

# def crack_splitter(x_1, x_2, length_element,z):
#     '''
#     ===========================================================================
#     This function splits the length of the crack in equal proportion as
#     that of the length of the element.
#     This function is called only in case of crack_length >> element_length
#     ===========================================================================
#     Parameters
#     ----------
#     x_1 : tip_1
#     x_2 : tip_2
#     length_element : Element length
#     xx : points holder
#     z : integer value of the number of points to split
#     Returns
#     -------
#     list of intersection points.
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
#     '''

