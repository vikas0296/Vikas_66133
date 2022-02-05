import uniform_mesh
import numpy as np
import plots
import pandas as pd
from scipy import linalg
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
    """
    pos = (0,0)
    x, y =  pos
    ysize, xsize = dummy.shape
    xmax, ymax = (x + xsize), (y + ysize)
    matrix[y:ymax, x:xmax] += dummy
    return matrix

def connectivity_matrix(EL, KL, length_nodes, Hside, Tside):
    Tside.sort()
    # print("hside", Hside)
    # print("Tside", Tside)

    classic_element_DOFs = 8

    print(f" A classical element has '{classic_element_DOFs}' degrees of freedom")

    Total_heavy_dofs = len(Hside)*2
    print(f"Geometry consists of '{Total_heavy_dofs}' heaviside enriched degrees of freedom")

    Total_tip_dofs = len(Tside)*8
    print(f"Geometry consists of '{Total_tip_dofs}' Tip enriched degrees of freedom")

    Total_Normal_DOFs = length_nodes*2
    print(f"Geometry consists of '{Total_Normal_DOFs}' classical degrees of freedom")

    if len(Hside) == 1 or len(Hside) == 2 or len(Hside) == 3:
        rows = classic_element_DOFs + len(Hside)*2 + Total_tip_dofs
        h_rows = len(Hside)*2

    elif len(Hside) >= 4:
        rows = classic_element_DOFs + 8 + Total_tip_dofs
        h_rows = 8

    elif len(Hside) == 0:
        rows = classic_element_DOFs + Total_tip_dofs
        h_rows = 0

    columns = Total_Normal_DOFs + Total_heavy_dofs + Total_tip_dofs
    print(f"Geometry consists of '{columns}' degrees of freedom it includes enrichments")
    print("===================================================================")

# =====================generating element connectivity matrix===========================================================
    A = []
    Ks = []
    AKs = []
    for i in KL:
        dummy = np.zeros([rows, rows])
        # shape = dummy.shape
        Ks.append(addAtPos(i, dummy))

    # print("length", len(Ks))
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
            if j in Hside:
                for k in Hside:
                    if j == k:
                        A_matrix[(i.index(k))*2+8, (Hside.index(k))*2+Total_Normal_DOFs] = 1
                        A_matrix[(i.index(k))*2+9, (Hside.index(k))*2+(Total_Normal_DOFs+1)] = 1


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


        # # print(A_matrix[30:40,39:55])
        A.append(A_matrix)
#==============================Assembling Global stiffness matrix===========================================

    for i,j,l in zip(A, Ks, EL):
        At_k = np.matmul(i.T, j)
        At_k_A = np.matmul(At_k,i)
        AKs.append(At_k_A)

    K_global = np.zeros([columns, columns])
    for i in AKs:
        K_global += i

    Total_Dofs = columns
    return K_global, Total_Normal_DOFs, Tside, Hside, Total_Dofs

def Boundary_conds(Nodes, A, B, element_length, K_global, force1):
    south = []
    south_points = []
    west = []
    west_points =[]
    north = []
    north_points = []
    east=[]
    east_points=[]
    DOFs=[]
    side = 'east'
    if side == 'south':
        for nodes, numbers in zip(Nodes, enumerate(Nodes)):
            if nodes[0] <= A and nodes[1] <= 0:
                south.append(nodes)
                south_points.append(numbers[0])
                DOFs.append(numbers[0])
        plots.DOFs(south)
# if nodes[0] == A and nodes[1] < B*0.6 and nodes[1]>B*0.45:
    # if nodes[0] == A and nodes[1] < B  and nodes[1]> B*0.45:

    elif side == 'east':
        for nodes, numbers in zip(Nodes, enumerate(Nodes)):
            if nodes[0] == A and nodes[1] < B*0.6 and nodes[1]>B*0.45:
                east.append(nodes)
                east_points.append(numbers[0])
                DOFs.append(numbers[0])
        plots.DOFs(east)

    elif side == 'west':
        for nodes, numbers in zip(Nodes, enumerate(Nodes)):
            if nodes[0] == 1 and nodes[1] <= B:
                west.append(nodes)
                west_points.append(numbers[0])
                DOFs.append(numbers[0])
        plots.DOFs(west)

    # print(DOFs)
    # Application of Neumann and Derechlet boundary conditions

    location = "3"
    loc = []
    loc_elements = []
    if location == "3":
        for N, E in zip(Nodes, enumerate(Nodes)):
            # print(N)
            # if N[0] == 0 and N[1] <= A:
            #     # print("node", N)
            #     loc.append(N)
            #     loc_elements.append(E[0])

            if N[0] >= 0 and N[1] == B:
                loc.append(N)
                loc_elements.append(E[0])

            if N[0] <= A and N[1] == 0:
                loc.append(N)
                loc_elements.append(E[0])


    plots.force(loc)
    locations =[]
    for i in loc_elements:
        Py = (i*2)+1
        locations.append(Py)

    # print(locations)
    force_vector1 = np.zeros([len(K_global),1])
    down = locations[:len(locations)//2]
    up = locations[len(locations)//2:]

    # for i in locations:
    #     force_vector1[i, 0] = force1

    for i,j in zip(up,down):
        force_vector1[i, 0] = force1
        force_vector1[j, 0] = -force1

    # print(force_vector1)

    # Us = pd.DataFrame(force_vector1)
    # Us.to_excel(excel_writer = "f_vector.xlsx")

    # fixed_nodes = []
    # for i in DOFs:
    #     L = i*2
    #     fixed_nodes.append(L)
    #     K = L+1
    #     fixed_nodes.append(K)

    fixed_nodes = [0,0]
    # reduced_k = np.delete(np.delete(K_global, fixed_nodes, 0),  fixed_nodes, 1)
    # reduced_f1 = np.delete(force_vector1, fixed_nodes, 0)

    # Us = pd.DataFrame(reduced_k)
    # Us.to_excel(excel_writer = "k_vector.xlsx")

    displacement_vector1 = np.linalg.pinv(K_global).dot(force_vector1)
    # disps_vector = []
    # for i in displacement_vector1:
    #     disps_vector.append(i*1e-3)

    return displacement_vector1, fixed_nodes

