import uniform_mesh
import numpy as np
import plots
import pandas as pd
from scipy import linalg
from pandas import ExcelWriter

def addAtPos(matrix, dummy):
    """
    Add two matrices of different sizes in place, offset by xy coordinates
    Usage:
      - mat1: base matrix
      - mat2: add this matrix to mat1
      - xypos: tuple (x,y) containing coordinates
    """
    pos = (0,0)
    np.set_printoptions(suppress=True)

    x, y =  pos
    ysize, xsize = matrix.shape
    xmax, ymax = (x + xsize), (y + ysize)
    dummy[y:ymax, x:xmax] += matrix
    return dummy

def connectivity_matrix(NL, EL, KL, length_nodes):
    heavy0 = []
    heavy1 = []
    heavy2 = []
    heavy3 = []
    tip0 = []
    tip1 = []
    tip2 = []
    tip3 = []

    for j in EL:
        for i in range(len(j)):
            if j[i] == "0heaviside_enriched":
                heavy0.append(j[0])

            elif j[i] == "1heaviside_enriched":
                heavy1.append(j[1])

            elif j[i] == "2heaviside_enriched":
                heavy2.append(j[2])

            elif j[i] == "3heaviside_enriched":
                heavy3.append(j[3])

            elif j[i] == "0tip_enriched":
                  tip0.append(j[0])

            elif j[i] == "1tip_enriched":
                  tip1.append(j[1])

            elif j[i] == "2tip_enriched":
                  tip2.append(j[2])

            elif j[i] == "3tip_enriched":
                  tip3.append(j[3])

    TIP = [tip1,tip2,tip3]

    for i in TIP:
        tip0.extend(i)

    Tside = list(dict.fromkeys(tip0))

    HEAVY = [heavy1, heavy2, heavy3]

    for i in HEAVY:
        heavy0.extend(i)

    Hside = list(dict.fromkeys(heavy0))

    print("heavy_side", Hside)
    Tside.sort()
    # print("Tside", Tside)
    Tside[-1],Tside[-2] = Tside[-2], Tside[-1]
    print("Tside", Tside)
    # for i in EL:
    #     # print("el before", i)
    #     while len(i) > 4:
    #         i.pop()
    #         # print("el after i", i)


    classic_element_DOFs = 8
    print(f" A classical element has '{classic_element_DOFs}' degrees of freedom")

    Total_heavy_dofs = len(Hside)*2
    print(f"Geometry consists of '{Total_heavy_dofs}' heaviside enriched degrees of freedom")

    Total_tip_dofs = len(Tside)*8
    print(f"Geometry consists of '{Total_tip_dofs}' Tip enriched degrees of freedom")

    Total_geometry_DOFs = length_nodes*2
    print(f"Geometry consists of '{Total_geometry_DOFs}' classical degrees of freedom")

    if len(Hside) == 1 or len(Hside) == 2 or len(Hside) == 3:
        rows = classic_element_DOFs + len(Hside)*2 + Total_tip_dofs
        h_rows = len(Hside)*2

    elif len(Hside) >= 4:
        rows = classic_element_DOFs + 8 + Total_tip_dofs
        h_rows = 8

    elif len(Hside) == 0:
        rows = classic_element_DOFs + Total_tip_dofs
        h_rows = 0

    columns = Total_geometry_DOFs + Total_heavy_dofs + Total_tip_dofs
    print(f"Geometry consists of '{columns}' degrees of freedom it includes enrichments")
    print("===================================================================")
# =====================generating element connectivity matrix===========================================================
    A = []
    Ks = []
    AKs = []
    for i in KL:
        dummy = np.zeros([rows, rows])
        Ks.append(addAtPos(i, dummy))

    for i in EL:
        # print(i)
        # print("-------------------")
        A_matrix = np.zeros([rows, columns])
        P = int(i[0])-1
        Q = int(i[1])-1
        R = int(i[2])-1
        S = int(i[3])-1
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
                        # print(f"{j} is equal to {k} and heaviside {Hside.index(k)}")
                        # print(f"row is{(Hside.index(k))*2+8} and column{(Hside.index(k))*2+Total_geometry_DOFs}")
                        # print(f"row is{(Hside.index(k))*2+9} and column{(Hside.index(k))*2+Total_geometry_DOFs+1}")
                        A_matrix[(i.index(k))*2+8, (Hside.index(k))*2+Total_geometry_DOFs] = 1
                        A_matrix[(i.index(k))*2+9, (Hside.index(k))*2+(Total_geometry_DOFs+1)] = 1


            elif j in Tside:
                for T in Tside:
                    if j == T:
                            # print(f"{j} is equal to {T} and tip enriched the index is {Tside.index(T)}")
                            A_matrix[(i.index(j))*8+8+h_rows, (Tside.index(T))*8+Total_geometry_DOFs+
                                      Total_heavy_dofs] = 1

                            A_matrix[(i.index(j))*8+9+h_rows, (Tside.index(T))*8+Total_geometry_DOFs+
                                      Total_heavy_dofs+1] = 1

                            A_matrix[(i.index(j))*8+10+h_rows, (Tside.index(T))*8+Total_geometry_DOFs+
                                      Total_heavy_dofs+2] = 1

                            A_matrix[(i.index(j))*8+11+h_rows, (Tside.index(T))*8+Total_geometry_DOFs+
                                      Total_heavy_dofs+3] = 1

                            A_matrix[(i.index(j))*8+12+h_rows, (Tside.index(T))*8+Total_geometry_DOFs+
                                      Total_heavy_dofs+4] = 1

                            A_matrix[(i.index(j))*8+13+h_rows, (Tside.index(T))*8+Total_geometry_DOFs+
                                      Total_heavy_dofs+5] = 1

                            A_matrix[(i.index(j))*8+14+h_rows, (Tside.index(T))*8+Total_geometry_DOFs+
                                      Total_heavy_dofs+6] = 1

                            A_matrix[(i.index(j))*8+15+h_rows, (Tside.index(T))*8+Total_geometry_DOFs+
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
    return K_global, Total_geometry_DOFs, Tside, Hside, Total_Dofs

def Boundary_conds(Nodes, A, B, element_length, K_global, force1, force2):
    south = []
    south_points = []
    west = []
    west_points =[]
    north = []
    north_points = []
    east=[]
    east_points=[]
    DOFs=[]
    side = 'south'
    if side == 'south':
        for nodes, numbers in zip(Nodes, enumerate(Nodes)):
            if nodes[0] <= A and nodes[1] < 1+element_length:
                south.append(nodes)
                south_points.append(numbers[0])
                DOFs.append(numbers[0])
        plots.DOFs(south)

    elif side == 'east':
        for nodes, numbers in zip(Nodes, enumerate(Nodes)):
            if nodes[0] == A and nodes[1] <= B:
                east.append(nodes)
                east_points.append(numbers[0])
                DOFs.append(numbers[0])
        plots.DOFs(east)

    elif side == 'north':
        for nodes, numbers in zip(Nodes, enumerate(Nodes)):
            if nodes[0] <= A and nodes[1] == B:
                north.append(nodes)
                north_points.append(numbers[0])
                DOFs.append(numbers[0])
        plots.DOFs(north)

    elif side == 'west':
        for nodes, numbers in zip(Nodes, enumerate(Nodes)):
            if nodes[0] == 1 and nodes[1] <= B:
                west.append(nodes)
                west_points.append(numbers[0])
                DOFs.append(numbers[0])
        plots.DOFs(west)


    # Application of Neumann and Derechlet boundary conditions

    location = "3"
    loc = []
    loc_elements = []
    if location == "3":
        for N, E in zip(Nodes, enumerate(Nodes)):
            if N[0] == 1 and N[1] == B:
                loc.append(N)
                loc_elements.append(E[0])

    elif location == '2':
        for N, E in zip(Nodes, enumerate(Nodes)):
            if N[0] == A and N[1] == B:
                loc.append(N)
                loc_elements.append(E[0])

    elif location == '1':
        for N, E in zip(Nodes, enumerate(Nodes)):
            if N[0] == A and N[1] == 1:
                loc.append(N)
                loc_elements.append(E[0])

    elif location == '0':
        for N, E in zip(Nodes, enumerate(Nodes)):
            if N[0] == 0 and N[1] == 0:
                loc.append(N)
                loc_elements.append(E[0])

    plots.force(loc)

    if len(loc_elements) == 1:
        Py = loc_elements[0] * 2
        Px = 0

    force_vector1 = np.zeros([len(K_global),1])
    force_vector2 = np.zeros([len(K_global),1])
    print("in KN", Py)
    force_vector1[Py, 0] = force1
    force_vector2[Py, 0] = force2

    fixed_nodes = []
    for i in DOFs:
        L = i*2
        fixed_nodes.append(L)
        K = L+1
        fixed_nodes.append(K)

    reduced_k = np.delete(np.delete(K_global, fixed_nodes, 0),  fixed_nodes, 1)
    reduced_f1 = np.delete(force_vector1, fixed_nodes, 0)
    reduced_f2 = np.delete(force_vector2, fixed_nodes, 0)

    displacement_vector1 = linalg.solve(reduced_k, reduced_f1)
    displacement_vector2 = linalg.solve(reduced_k, reduced_f2)

    return displacement_vector1, displacement_vector2, fixed_nodes

