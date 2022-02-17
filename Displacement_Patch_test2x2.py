import numpy as np
import matplotlib.pyplot as plt
import operator
from scipy import linalg


fig = plt.figure()
fig.set_figwidth(10)
fig.set_figheight(10)
def plot_mesh(NL, A, B, X):
    plt.scatter(NL[:,0], NL[:,1], c='black', s=50, label = "Node")
    plt.legend(loc="upper right")

    for a, b, in zip(NL[0:,0], NL[0:,1]):
        plt.vlines(a, ymin=1, ymax=A, color="b", alpha=1)
        plt.hlines(b, xmin=1, xmax=B, color="c", alpha=1)
        # plt.savefig('saved_figure.png')
    plt.ylabel('Y-AXIS')
    plt.xlabel('X-AXIS')
    plt.title(f'2D_UNIFORM_MESH of size {X}x{X}')


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

    A = 2          #length of the geometry along x and y axis
    B = 2
    x = 2                                    #3x3 will be elements, 4x4 will be nodes
    NL, EL = uniform_mesh(A, B, x)
    x1 = NL[0]
    x2 = NL[1]

    '''
    length of a segment, given 2 points = sqrt((x2-x1)**2 + (y2-y1)**2)
    '''
    length_element = np.round(np.sqrt((x2[0]-x1[0])**2 + (x2[1]-x1[1])**2),4)
    print("the length of each element is ", length_element)
    all_ns, Elems_nodes = dictionary(NL, EL)
    elements = [ ]
    for j in Elems_nodes:
        elements.append(list(j.keys())[0])

    plot_mesh(NL, A, B, x)

    GP = np.zeros([4,2])
    for i in all_ns:
        Q1,Q2 = i[0], i[1]
        length = abs(Q1[0]-Q2[0]) / 4

        P = np.asarray(i)
        GP[0,0:2] = P[0] + length

        GP[2,0:2] = P[2] - length

        S = P[1]
        U,V = S[0],S[1]
        GP[1, 0] = abs(U - length)
        GP[1, 1] = abs(V + length)

        R = P[3]
        U, V = R[0],R[1]
        GP[3,0] = abs(U + length)
        GP[3,1] = abs(V - length)

        plt.scatter(GP[:,0], GP[:,1], marker = '*', color = 'r', s=100)

    #=============================Material_properties===========================================

    D = 200000
    nu = 0.3
    D_plane_stress = (D / (1 - nu**2)) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]])

    GaussPoint_1to4 = np.array([[ 1/np.sqrt(3),  1/np.sqrt(3)], [ 1/np.sqrt(3), -1/np.sqrt(3)],
                                [-1/np.sqrt(3),  1/np.sqrt(3)], [-1/np.sqrt(3), -1/np.sqrt(3)]])

    a = []
    Zero = []
    ZeroU = np.zeros([8,8])
    for i in all_ns:
        # print(i)
        for points in GaussPoint_1to4:
            xi_1 = points[0]
            xi_2 = points[1]


            dNdxi = (1/4) * np.array([[-(1-xi_2), 1-xi_2, 1+xi_2, -(1+xi_2)],
                                      [-(1-xi_1), -(1+xi_1), 1+xi_1, 1-xi_1]])

            jacobi = np.matmul(dNdxi, i)

            inverse_jacobi = linalg.inv(jacobi)

            dN = np.matmul(inverse_jacobi, dNdxi)

            B_std = np.array([[dN[0, 0], 0, dN[0, 1], 0, dN[0, 2], 0, dN[0, 3], 0],
                              [0, dN[1, 0], 0, dN[1, 1], 0, dN[1, 2], 0, dN[1, 3]],
                              [dN[1, 0], dN[0, 0], dN[1, 1], dN[0, 1], dN[1, 2], dN[0, 2], dN[1, 3], dN[0, 3]]])

            Bt_D_1 = np.matmul(B_std.T, D_plane_stress)
            Bt_D_B1 = (np.matmul(Bt_D_1, B_std)) * linalg.det(jacobi)
            # print(Bt_D_B1)
            a.append(Bt_D_B1)

        for j in a:
            ZeroU += j
        Zero.append(ZeroU)

    Assignment = []
    for i in elements:
        A_matrix = np.zeros([8,18])
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
        Assignment.append(A_matrix)

    AKs = []
    for i, j  in zip(Assignment, Zero):
        At_k = np.matmul(i.T, j)
        At_k_A = np.matmul(At_k,i)
        AKs.append(At_k_A)

    K_global = np.zeros([18,18])
    for i in AKs:
        K_global += i

    load_index=np.array([0,4,12,16])

    L = 10
    force = np.zeros([18,1])
    for i in range(len(load_index)):
        if (i)%2 == 0:
            force[load_index[i],0] = -L

        elif (i)%2 != 0:
            force[load_index[i],0] = L


    disp=np.linalg.solve(K_global,force)

    Us=[]
    for i in elements:
        Ux_Uy = [0,0,0,0,0,0,0,0]
        i = list(np.asarray(i) - 1)
        for j in i:
            Ux_Uy[i.index(j)*2] = disp[int(j)*2]
            Ux_Uy[i.index(j)*2+1] = disp[int(j)*2+1]

        Us.append(Ux_Uy)

    MID = [2,3,1,0]
    P = Us[0]
    print(P[MID[0]*2] , P[MID[0]*2+1])
    Q = Us[1]
    print(Q[MID[1]*2] , Q[MID[1]*2+1])
    R = Us[2]
    print(R[MID[2]*2] , R[MID[2]*2+1])
    S = Us[3]
    print(S[MID[3]*2] , S[MID[3]*2+1])



    # StressesX, StressesY, StressesXY  = [] ,[],[]
    # for i,j in zip(all_ns, Us):
    #     for points in GaussPoint_1to4:
    #         xi_1 = points[0]
    #         xi_2 = points[1]


    #         dNdxi = (1/4) * np.array([[-(1-xi_2), 1-xi_2, 1+xi_2, -(1+xi_2)],
    #                                   [-(1-xi_1), -(1+xi_1), 1+xi_1, 1-xi_1]])

    #         jacobi = np.matmul(dNdxi, i)

    #         inverse_jacobi = linalg.inv(jacobi)

    #         dN = np.matmul(inverse_jacobi, dNdxi)

    #         B_std = np.array([[dN[0, 0], 0, dN[0, 1], 0, dN[0, 2], 0, dN[0, 3], 0],
    #                           [0, dN[1, 0], 0, dN[1, 1], 0, dN[1, 2], 0, dN[1, 3]],
    #                           [dN[1, 0], dN[0, 0], dN[1, 1], dN[0, 1], dN[1, 2], dN[0, 2], dN[1, 3], dN[0, 3]]])

    #         Strain = np.matmul(B_std, j)
    #         # print(Strain)
    #         Stress = np.matmul(D_plane_stress, Strain)
    #         # print(Stress)
    #         StressesX.append(Stress[0])
    #         StressesY.append(Stress[1])
    #         StressesXY.append(Stress[2])


    # fig = plt.figure(figsize=(10,10))
    # W = StressesX
    # U = StressesY
    # v = StressesXY
    # P = np.linspace(v[0], v[-1], 256).reshape(16,16)
    # X, Y = np.meshgrid(W, U)
    # # Z = np.reshape(V,8,2)
    # cs = plt.contourf(X, Y, P, cmap='hsv')
    # fig.colorbar(cs)