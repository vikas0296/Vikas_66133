import numpy as np
import operator; import plots
import math as m; import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString, Point
from shapely.geometry.polygon import Polygon
import class_crack; import J_integral
import Assignment; import KU_F
import enrichment_functions; import Displacement_approx
import Stress_Strains

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

                ELS[(i-1)*x+j-1, 0] = (i-1)*(x+1) + j                #EL[0,0]
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
    D_2 = 2
    NS = np.zeros([Nodes, D_2])
    a = (corners[1, 0] - corners[0, 0]) / x     #divisions along X-axis
    b = (corners[2, 1] - corners[0, 1]) / x     #divisions along y-axis
    n = 0

    for i in range(1,x+2):
        for j in range(1, x+2):

            NS[n,0] = float("{:.2f}".format((corners[0, 0] + (j-1)*a)))  #x-values of nodes
            NS[n,1] = float("{:.2f}".format((corners[0, 1] + (i-1)*b)))  #y-values of nodes
            n += 1

    return NS


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
    corners = np.array([[0, 0],[A, 0],[0, B],[A, B]])
    Nodes = (x+1)**2
    Elements = x**2
    Nodes_elements = 4

    EL = elements(Elements, Nodes_elements, x)           # 4_nodes per_element list
    NL = nodes(Nodes, corners, x)                  # nodes_list
    round_NL = NL
    return round_NL, EL

if __name__ == "__main__":

    A = 2000     #length of the geometry along x and y axes
    B = 2000
    x = 20 # no of divisions along x and y directions

    scale = 4 #scaling factor for the circle around the cracktip
    NL, EL = uniform_mesh(A, B, x)
    #identifying the necessary points to calculate the element length
    x1 = NL[0]
    x2 = NL[1]
    x3 = NL[21]
    length_nodes = len(NL)
    #prescribing displacements
    DISP = 1.5

    # #=============================Material_properties===========================================

    #Young's modulus, Poission's Ratio, Plane stres srelation
    D = 200000
    nu = 0.25
    D_plane_stress = (D / (1 - nu**2)) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]])

    GaussPoint_1to4 = np.array([[ 1/np.sqrt(3),  1/np.sqrt(3)], [ 1/np.sqrt(3), -1/np.sqrt(3)],
                                [-1/np.sqrt(3),  1/np.sqrt(3)], [-1/np.sqrt(3), -1/np.sqrt(3)]])

    '''
    length of a segment, given 2 points = sqrt((x2-x1)**2 + (y2-y1)**2)
    '''
    #calculating the length of the element along x and y directions
    length_element_x = np.round(np.sqrt((x2[0]-x1[0])**2 + (x2[1]-x1[1])**2),4)
    print("the length of each element is in x ", length_element_x)
    length_element_y = np.round(np.sqrt((x1[0]-x3[0])**2 + (x1[1]-x3[1])**2),4)
    print("the length of each element is in y ", length_element_y)
    geom_ns, Elems_nodes = dictionary(NL,EL)
    elements = []

    for j in Elems_nodes:
        elements.append(list(j.keys())[0])

    #Subtracting 1 for the element list
    #before = [1,2,4,3]
    #after = [0,1,3,2]
    UNIT =[]
    for i in elements:
        ELES = np.asarray(i)
        UNIT.append(ELES-1)

    #calling the mesh function to plot the mesh
    plots.plot_mesh(NL, A, B, x, length_element_x)


    '''
    ===========================================================================
    Crack modelling
    ===========================================================================
    '''
    #defining  the crack tip locations
    cracktip_1 = np.array([0, (A/3-length_element_y/2)])
    cracktip_2 = np.array([B/2 + length_element_y/2 , B/2 + length_element_y/2])

    #plotting the crack segment
    plots.plot_crack(cracktip_1, cracktip_2, length_element_x)
    crack_length = round(abs(cracktip_1[0]-cracktip_1[1]), 3)
    print("the length of the crack is: ", crack_length)

    NODE = []
    ELE = []
    alpha = 0
    gamma = []
    cracks = []
    N_list, E_list, HE_list =[], [], []
    Klassic_mat, Enr_matrix =[], []
    Verschiebung = []
    Belastung = []
    NODES, ELEMENTS, K_MATS = [], [], []
    N_s, E_s = [], []

    if cracktip_1[1] in NL[:,1]:
        print("Please change the coordinates of the crack, it might be grazing the edge of the element")

    if cracktip_2[0]/1000 >= A or cracktip_2[1]/1000 >= B:
        print(f"The crack_length {crack_length} is >= to the geometry length {A},"
                            "therefore the sample is broken into two halves")

    else:
        c_1 = np.array([cracktip_1[0], cracktip_2[0]])
        c_0 = np.copy(c_1)
        c_2 = np.array([cracktip_1[1], cracktip_2[1]])

        #dividing the crack segments to n number of segments to the length equal to element length
        splits = np.arange(c_1[0], c_2[0], length_element_x)
        idx = [0]
        W = np.delete(splits, idx)
        new_points = [0,0]
        for i in W:
            new_points = [0,0]
            new_points[0] = round(i,2)
            new_points[1] = c_1[1]
            gamma.append(new_points)

        gamma.insert(0, list(c_0))
        gamma.insert(len(gamma), list(c_2))
        gamma=list(np.unique(gamma,axis=0))

        #selecting all the elements where the crack has passed through
        for j in range(0,len(gamma)-1):
            for i,k in zip(geom_ns, UNIT):
                TR = i[2]
                TR2 = i[0]
                polygon = Polygon(i)
                path = LineString([tuple(gamma[j]), tuple(gamma[j+1])])
                EXT = gamma[j+1]
                EXT2 = gamma[j]
                if path.intersects(polygon) and i not in NODE and EXT[0] <= TR[0] and EXT[0] > TR2[0]:
                    NODE.append(i)
                    ELE.append(k)

        cracks = cracks + gamma
        cracks, index = np.unique(cracks, return_index=True, axis = 0)
        cracks = list(cracks[index.argsort()])

        #selecting the element which require tip enrichment
        for i,j in zip(NODE, ELE):
            point = Point(tuple(cracks[-1]))
            polygon = Polygon(i)
            #using the polygon library to check where the crack tip is
            if point.touches(polygon) or polygon.contains(point) or point.within(polygon):
                T_Node_list, Tip_matrix, T_Elem, ri, thetai  = class_crack.updated_cracktip(i, j, cracks[-1],GaussPoint_1to4,
                                                                        D_plane_stress, alpha)

                N_list.append(i)
                E_list.append((T_Elem))
                Enr_matrix.append(Tip_matrix)

        #selecting the elements which require Heaviside enerichments
        #looping through all the nodes and elements
        for i, j in zip(NODE, ELE):
            Q,W,E,R = j[0], j[1], j[2], j[3]
            if Q not in T_Elem and W not in T_Elem and E not in T_Elem and R not in T_Elem:
                H_Node_list, H_Elem, H_matrix = class_crack.updated_heaviside(i, j, GaussPoint_1to4, D_plane_stress, cracks)
                N_list.append(i)
                E_list.append(j)
                HE_list.append(H_Elem)
                Enr_matrix.append(H_matrix)

        #calling the filtering function to filter out the nodes which require enrichments
        MIX_N = Assignment.filtering(N_list, NODE)
        MIX_E = Assignment.E_filter(E_list, ELE)
        #calling the function to compute the stiffness matrices for pretip elements
        H_Enrich_Elem, PT_matrix  =  class_crack.updated_pretipenr(MIX_N, MIX_E, HE_list, T_Elem, GaussPoint_1to4,
                                                                           D_plane_stress, alpha, cracks)

        #calling the filtering function to filter out the enenriched nod
        H_Enrich_Elem = list(np.unique(H_Enrich_Elem))
        NON_N = Assignment.filtering(NODE, geom_ns)
        NON_E = Assignment.E_filter(ELE, UNIT)

        #computing the B-matrices for normal and blended elements/nodes
        N_matrix, B_matrix, N_elements, B_elements, N_nodes, B_nodes = class_crack.Normal_blended_en(NON_N, NON_E, H_Enrich_Elem, T_Elem, GaussPoint_1to4,
                                                            D_plane_stress, alpha, cracks)

        #collecting all the enriched element number
        ENRICHMENTS = []
        for i in E_list:
            ENRICHMENTS.append(tuple(i))

        #combining all the nodal lists and element lists
        NODES = N_list + MIX_N + N_nodes + B_nodes
        ELEMENTS = ENRICHMENTS + MIX_E + N_elements + B_elements
        #gathering all the K-matrices
        K_MATS = Enr_matrix + PT_matrix + N_matrix + B_matrix

        Tside = []
        Hside = []

        #making a list of tip enriched elements
        for i in T_Elem:
            Tside.append(i.tolist())

        #making a list of Heaviside enriched elements
        for i in H_Enrich_Elem:
            Hside.append(i.tolist())

        #calling the connectivity_matrix function to compute the global stiffness matrix
        K_global, CLASS_DOFs, Total_Dofs = KU_F.connectivity_matrix(ELEMENTS, K_MATS, length_nodes, Hside, Tside)

        #calling the Boundary_conds function to solve the BVP
        Displacement_vector, fixed_BCs = KU_F.Boundary_conds(NL, A, B, K_global, DISP)

        #adding zeros in the displacement vector rows where the nodes are fixed
        for j in fixed_BCs:
            Displacement_vector = np.insert(Displacement_vector, j, 0)

        # Inputs for Interaction integral
        for i,j in zip(geom_ns, UNIT):
            for k in i:
                #to check if the nodes lie inside the prescribed circle
                Z = J_integral.inside_circ(cracks[-1], length_element_x, length_element_y, k, scale)
                if Z == True and i not in N_s:
                    N_s.append(i)
                    E_s.append(j)

        # area = length_element_x*length_element_y
        # radius = scale * np.sqrt(area)
        # plots.circle(cracks[-1], radius)

        #combining the lists of enriched nodal list
        ENR_NODES = N_list + MIX_N
        for i,j in zip(N_s, E_s):
            if i in ENR_NODES:
                #calling the function to compute the displacements for enriched nodes
                DISPLACEMENTS_En = Displacement_approx.displacement_approximation([i], cracks, [j], Tside, Hside,
                                                                                  CLASS_DOFs, Displacement_vector, alpha)
                #calling the function to compute the stresses for enriched nodes
                STRESS_En = Stress_Strains.strain_stress_enr(DISPLACEMENTS_En, [i], GaussPoint_1to4,
                                                                        D_plane_stress)

                Verschiebung.append(DISPLACEMENTS_En)
                Belastung.append(STRESS_En)


            else:
                #calling the function to compute the displacements for un_enriched nodes
                DISPLACEMENTS_N = Displacement_approx.displacement_approximation([i], cracks, [j], Tside, Hside,
                                                                                  CLASS_DOFs, Displacement_vector, alpha)
                #calling the function to compute the stresses for unenriched nodes
                STRESS_N = Stress_Strains.strain_stress_enr(DISPLACEMENTS_N, [i], GaussPoint_1to4, D_plane_stress)

                Verschiebung.append(DISPLACEMENTS_N)
                Belastung.append(STRESS_N)

        #calling the G_points function to computing the gauss points for the nodes inside the circle
        G, GPs = Assignment.G_points(N_s)

        #calling the Interaction integral to compute SIF
        J_integral.Interaction_integral(N_s, Verschiebung, Belastung, GPs, cracks[-1],GaussPoint_1to4,
                                        length_element_x, length_element_y, D_plane_stress, alpha, crack_length,
                                                                   A, DISP, D, scale)

        #combining all the nodal lists and element lists
        NODES = N_nodes + B_nodes + MIX_N +  N_list
        ELEMENTS = N_elements + B_elements + MIX_E + ENRICHMENTS

        #calling the function to compute the stresses and displacements for all the nodes
        STS, Ux_Uy = Assignment.DSS(NODES, ELEMENTS, cracks, Tside, Hside, CLASS_DOFs,
                                    Displacement_vector, alpha, D_plane_stress,GaussPoint_1to4)

        #plot function for stress distribution contour plot
        fig = plt.figure()
        fig.set_figwidth(10)
        fig.set_figheight(10)

        STR1, STR2, STR3 =[],[],[]
        #Gathering the Gauss points
        GAUSS, GAUSS_PTS = Assignment.G_points(NODES)

        #separating sigma1 ,sigma2, sigma12
        for i in STS:
            for j in i:
                STR1.append(j[0])
                STR2.append(j[1])
                STR3.append(j[2])

        #calling the contour plot function
        L = np.asarray(GAUSS)
        stress1 = np.asarray(STR1)
        stress2 = np.asarray(STR2)
        stress3 = np.asarray(STR3)
        plots.contour_plot(L, stress1, A, 1)
        plots.contour_plot(L, stress2, A, 2)
        plots.contour_plot(L, stress3, A, 3)
