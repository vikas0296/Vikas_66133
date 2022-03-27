'''
AUTHOR : VIKAS MELKOT BALASUBRAMANYAM
MATRICULATION NUMBER : 66133
Personal Programming Project: IMPLEMENTATION OF XFEM AND FRACTURE MECHANICS
#=============================================================================
'''

import uniform_mesh
import numpy as np
import plots
from scipy import linalg
import pandas as pd
import matplotlib.pyplot as plt

def addAtPos(dummy, matrix):

    """
    This function is created to balance the size of the matrices.
    This function is called when the size of the matrix is not  40X40, the highest possible size for a matrix
    in the sample
    The function adds two matrices of different sizes in place, offset by xy coordinates
    Usage:
      - matrix: base matrix
      - dummy: add this matrix to mat1
      - pos: tuple (x,y) containing the loaction where the matrix to be added
      This code has been taken from https://stackoverflow.com/
    """
    pos = (0,0)
    x, y =  pos
    ysize, xsize = dummy.shape
    xmax, ymax = (x + xsize), (y + ysize)
    matrix[y:ymax, x:xmax] += dummy
    return matrix

def connectivity_matrix(EL, KL, length_nodes, Hside, Tside):
    '''
    The function computes Assignment matrix for all the types of the elements

    Parameters
    ----------
    EL : Element list [1,2,3,4]
    KL : List of stiffness matrices
    length_nodes : Total nodes present in the geometry
    Hside : List of nodes that are heaviside enriched
    Tside : List of nodes that are tip enriched
    Returns
    -------
    K_global : assembled stiffness matrix
    Total_Normal_DOFs : Total classical DOFs
    Total_Dofs : Total Geometry DOFs
    Tside = sorted list of nodes that are tip enriched
    Hside = sorted list of nodes that are Heaviside enriched
    '''

    Tside.sort()
    #printing all the essential DOFs
    classic_element_DOFs = 8

    # print(f" A classical element has '{classic_element_DOFs}' degrees of freedom")

    Total_heavy_dofs = len(Hside)*2
    # print(f"Geometry consists of '{Total_heavy_dofs}' heaviside enriched degrees of freedom")

    Total_tip_dofs = len(Tside)*8
    # print(f"Geometry consists of '{Total_tip_dofs}' Tip enriched degrees of freedom")

    Total_Normal_DOFs = length_nodes*2
    # print(f"Geometry consists of '{Total_Normal_DOFs}' classical degrees of freedom")

    if len(Hside) == 1 or len(Hside) == 2 or len(Hside) == 3:
        rows = classic_element_DOFs + len(Hside)*2 + Total_tip_dofs
        h_rows = len(Hside)*2

    elif len(Hside) >= 4:
        rows = classic_element_DOFs + 8 + Total_tip_dofs
        h_rows = 8

    elif len(Hside) == 0:
        rows = classic_element_DOFs + Total_tip_dofs
        h_rows = 0

    #Columns are formed based on the total number of DOFs that is available
    columns = Total_Normal_DOFs + Total_heavy_dofs + Total_tip_dofs
    # print(f"Geometry consists of '{columns}' degrees of freedom it includes enrichments")
    print("===================================================================")

# =====================generating element connectivity matrix===========================================================
    #Formation of Assignment matrix
    A = []
    Ks = []
    AKs = []
    #looping through the element stiffness matrices to make the shape of all the matrices even (40x40)
    for i in KL:
        dummy = np.zeros([rows, rows])
        Ks.append(addAtPos(i, dummy))

    #looping through the element list [[1,2,3,4], [], []..]
    for i in EL:
        A_matrix = np.zeros([rows, columns])
        P = int(i[0])
        Q = int(i[1])
        R = int(i[2])
        S = int(i[3])
        A_matrix[0, P*2] = 1
        A_matrix[1, P*2+1] = 1
        A_matrix[2, Q*2] = 1
        A_matrix[3, Q*2+1] = 1
        A_matrix[4, R*2] = 1
        A_matrix[5, R*2+1] = 1
        A_matrix[6, S*2] = 1
        A_matrix[7, S*2+1] = 1
        for j in i:
            #to check if the node has been Heaviside enriched
            if j in Hside:
                for k in Hside:
                    if j == k:
                        A_matrix[(i.index(k))*2+8, (Hside.index(k))*2+Total_Normal_DOFs] = 1
                        A_matrix[(i.index(k))*2+9, (Hside.index(k))*2+(Total_Normal_DOFs+1)] = 1

            #to check if the node has been Tip enriched
            elif j in Tside:
                for T in Tside:
                    if j == T:
                        A_matrix[(i.index(j))*8+8+h_rows, (Tside.index(T))*8+Total_Normal_DOFs+
                                  Total_heavy_dofs] = 1

                        A_matrix[(i.index(j))*8+9+h_rows, (Tside.index(T))*8+Total_Normal_DOFs+
                                  Total_heavy_dofs+1] = 1

                        A_matrix[(i.index(j))*8+10+h_rows, (Tside.index(T))*8+Total_Normal_DOFs+
                                  Total_heavy_dofs+2] = 1

                        A_matrix[(i.index(j))*8+11+h_rows, (Tside.index(T))*8+Total_Normal_DOFs+
                                  Total_heavy_dofs+3] = 1

                        A_matrix[(i.index(j))*8+12+h_rows, (Tside.index(T))*8+Total_Normal_DOFs+
                                  Total_heavy_dofs+4] = 1

                        A_matrix[(i.index(j))*8+13+h_rows, (Tside.index(T))*8+Total_Normal_DOFs+
                                  Total_heavy_dofs+5] = 1

                        A_matrix[(i.index(j))*8+14+h_rows, (Tside.index(T))*8+Total_Normal_DOFs+
                                  Total_heavy_dofs+6] = 1

                        A_matrix[(i.index(j))*8+15+h_rows, (Tside.index(T))*8+Total_Normal_DOFs+
                                  Total_heavy_dofs+7] = 1

        A.append(A_matrix)
#==============================Assembling Global stiffness matrix===========================================
    #looping through the Assignment matrices and element stiffness matrices
    for i,j in zip(A, Ks):
        At_k = np.matmul(i.T, j)
        At_k_A = np.matmul(At_k,i)
        AKs.append(At_k_A)

    #formation of Global Stiffness matrix
    K_global = np.zeros([columns, columns])
    for i in AKs:
        K_global += i

    Total_Dofs = columns
    return K_global, Total_Normal_DOFs, Total_Dofs, A

#===========================================================================================================

###############################################################3
def Boundary_conds_I_II(Nodes, A, B, K_global, prescribe_disp, Mode):
    '''
    The function is used to solve the BVP for mode_I and mode_II cracks.
    Parameters
    ----------
    Nodes : List of Nodal coordinates
    A, B : length of geometry along x and y directions
    K_global : Global stiffness matrix
    prescribe_disp : Applied displacement
    Mode : Type of crack tip loading

    Returns
    -------
    D_vector : Computed displacement vector
    fixed_nodes : List of fixed nodes
    '''
    south = []
    south_points = []
    DOFs=[]
    #identifying the nodes to be fixed
    for nodes, numbers in zip(Nodes, enumerate(Nodes)):
        if nodes[0] <= A and nodes[1] <= 0:
            south.append(nodes)
            south_points.append(numbers[0])
            DOFs.append(numbers[0])
    plots.DOFs(south)

    # Application of Neumann and Derechlet boundary conditions
    loc = []
    loc_elements = []

    #identifying the the force applying nodes
    for N, E in zip(Nodes, enumerate(Nodes)):
        if N[0] >= 0 and N[1] == B:
            loc.append(N)
            loc_elements.append(E[0])


    plots.force(loc, 'Displacement')
    D_vector = np.zeros([len(K_global), 1])

    #collecting the points where the displacemnts are described
    locations =[]
    if Mode == "I":
        for i in loc_elements:
            Py = (i*2)+1
            locations.append(Py)

    elif Mode == "II":
        for i in loc_elements:
            Px = (i*2)
            locations.append(Px)


    #computing the location for fixing the nodes
    fixed_nodes = []
    for i in DOFs:
        L = i*2
        fixed_nodes.append(L)
        K = L+1
        fixed_nodes.append(K)

    #deleting rows for the fixed DOFs
    reduced_d = np.delete(D_vector, fixed_nodes, 0)
    K_global[fixed_nodes, fixed_nodes] = 0

    Ud = np.zeros([len(locations),1])
    Ud[:,0] = prescribe_disp

    widt = len(reduced_d) - len(locations)
    Uf = np.zeros([widt, 1])

    UnD = []
    #identifying the locations where the displacements are unknown
    for i in range(len(K_global)):
        if i not in locations and i not in fixed_nodes:
            UnD.append(i)

    Nan = 0
    '''
    #for kfd [rows x columns], rows where displacements are described and columns
    #where desplacements are unknown. if one deletes these rows and columns
    #one can get rows where displacements are not described and columns where displacements are known
    '''

    kfd = np.delete(np.delete(K_global, fixed_nodes + locations, 0), UnD+fixed_nodes, 1)
    '''
    #for kdd, we need rows and columns of the unwnown displacements, therefore we delete the
    #rows and columns of known displacements
    '''
    kdd = np.delete(np.delete(K_global, fixed_nodes + locations, 0), fixed_nodes + locations, 1)

    #multiplying kfd and Ud
    FF = kfd@Ud
    Uf = -np.linalg.pinv(kdd).dot(FF)

    #collecting all the displacements
    for i in locations:
        D_vector[i,0] = prescribe_disp

    for j, k in zip(UnD, Uf):
        D_vector[j, 0] = k

    return D_vector, fixed_nodes, Nan


def Boundary_conds_M1(Nodes, A, B, K_global, force, cracks, a):

    '''
    The function is used to solve the BVP for mixed mode crack. This example has been
    taken from "STRESS ANALYSIS OF CRACKS HANDBOOK"
    Parameters
    ----------
    Nodes : List of Nodal coordinates
    A, B : length of geometry along x and y directions
    K_global : Global stiffness matrix
    prescribe_disp : Applied displacement
    force: applied load
    cracks: list of crack segments
    a : length of the crack
    Returns
    -------
    D_vector : Computed displacement vector
    fixed_nodes : List of fixed nodes
    Y: ratio of Distance from the point of load application to the left crack tip and crack length

    '''
    #selecting the nodes to fix
    south = []
    south_points = []
    DOFs=[]
    for nodes, numbers in zip(Nodes, enumerate(Nodes)):
        if nodes[0] <= A and nodes[1] == 0:
            south.append(nodes)
            south_points.append(numbers[0])
            DOFs.append(numbers[0])
    plots.DOFs(south)

    #selecting nodes to apply boundary conditions
    loc = []
    loc_elements = []
    for N, E in zip(Nodes, enumerate(Nodes)):
        if N[0] == 0 and N[1] > B*0.8 and N[1] < B*0.9:
            loc.append(N)
            loc_elements.append(E[0])

    force_vector = np.zeros([len(K_global),1])
    #plotting the point
    plots.force(loc, 'load')
    #looping through the locations
    locations =[]
    for i in loc_elements:
        Py = (i*2)+1
        Px = (i*2)
        force_vector[Px, 0] = force
        force_vector[Py, 0] = force

    #calculating the ratio Y
    N_o = loc[0] #y coordinate of the node
    pt1 = cracks[0] #y coordinate of the crack tip
    Y0 = abs(pt1[1] - N_o[1])
    Y = Y0/a

    #nodes will be fixed here
    fixed_nodes = []
    for i in DOFs:
        L = i*2
        fixed_nodes.append(L)
        K = L+1
        fixed_nodes.append(K)

    #solving the actual BVP by reducing the rows and columns from global stiffness matrix
    reduced_k = np.delete(np.delete(K_global, fixed_nodes, 0),  fixed_nodes, 1)
    reduced_f = np.delete(force_vector, fixed_nodes, 0)

    displacement_vector = np.linalg.pinv(reduced_k).dot(reduced_f)

    return displacement_vector, fixed_nodes, Y
