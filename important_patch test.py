import numpy as np
import matplotlib.pyplot as plt
import operator
from scipy import linalg

fig = plt.figure()
fig.set_figwidth(10)
fig.set_figheight(10)
def plot_mesh(NL):
    plt.scatter(NL[:,0], NL[:,1], c='black', s=50, label = "Node")
    plt.legend(loc="upper right")

if __name__ == "__main__":

    NL1 = np.array([0,0])
    NL2 = np.array([4,0])
    NL3 = np.array([10,0])
    NL4 = np.array([0,4.5])
    NL5 = np.array([5.5,5.5])
    NL6 = np.array([10,5])
    NL7 = np.array([0,10])
    NL8 = np.array([4.2,10])
    NL9 = np.array([10,10])
    NLll = np.array([NL1, NL2, NL3, NL4, NL5, NL6, NL7, NL8, NL9])
    plot_mesh(NLll)

    NL1 = [0,0]
    NL2 = [4,0]
    NL3 = [10,0]
    NL4 = [0,4.5]
    NL5 = [5.5,5.5]
    NL6 = [10,5]
    NL7 = [0,10]
    NL8 = [4.2,10]
    NL9 = [10,10]
    NL = [NL1, NL2, NL3, NL4, NL5, NL6, NL7, NL8, NL9]
    E1 = [0,1,4,3]
    E2 = [1,2,5,4]
    E3 = [3,4,7,6]
    E4 = [4,5,8,7]
    Es = [E1, E2, E3, E4]

    nodes1 = [NL1, NL2, NL5, NL4]
    nodes2 = [NL2, NL3, NL6, NL5]
    nodes3 = [NL4, NL5, NL8, NL7]
    nodes4 = [NL5, NL6, NL9, NL8]

    ALL_NL = [nodes1, nodes2, nodes3, nodes4]

    #=============================Material_properties===========================================

    D = 1000
    nu = 0.25
    D_plane_strain = (D / (1 + nu)* (1-2*nu)) * np.array([[(1 - nu), nu, 0],
                                                          [nu, (1 - nu), 0],
                                                          [0, 0, (1 - 2*nu)/2]])

    GaussPoint_1to4 = np.array([[ 1/np.sqrt(3),  1/np.sqrt(3)], [ 1/np.sqrt(3), -1/np.sqrt(3)],
                                [-1/np.sqrt(3),  1/np.sqrt(3)], [-1/np.sqrt(3), -1/np.sqrt(3)]])


    a = []
    Zero = []
    ZeroU = np.zeros([8,8])
    for i in ALL_NL:
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

            Bt_D_1 = np.matmul(B_std.T, D_plane_strain)
            Bt_D_B1 = (np.matmul(Bt_D_1, B_std)) * linalg.det(jacobi)
            a.append(Bt_D_B1)

        for j in a:
            ZeroU += j
        Zero.append(ZeroU)

    rows = 8
    columns = len(NL) * 2
    Assignment = []
    for i in Es:
        A_matrix = np.zeros([rows,columns])
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
        Assignment.append(A_matrix)

    AKs = []
    for i, j  in zip(Assignment, Zero):
        At_k = np.matmul(i.T, j)
        At_k_A = np.matmul(At_k,i)
        AKs.append(At_k_A)

    K_global = np.zeros([columns,columns])
    for i in AKs:
        K_global += i

    Boundarys = [1,4,7]
    force_vector = np.zeros([len(K_global),1])
    positions = [2,5,8]

    plots = np.array([NLll[2],  NLll[5],  NLll[8]])
    plt.scatter(plots[:,0], plots[:,1], color="g", s = 200)
    force_vector[2*2,0] = 2.5
    force_vector[5*2,0] = 5
    force_vector[8*2,0] = 2.5

    U_global = np.zeros([columns, 1])

    constraints = [0,1,6,12]
    reduced_k = np.delete(np.delete(K_global, constraints, 0),  constraints, 1)
    reduced_f = np.delete(force_vector, constraints, 0)
    u = np.delete(U_global, constraints, 0)

    n = 1
    Rs =[]
    counter = 1
    while n!=5:
        print("counter:",counter)
        P = np.matmul(reduced_k, u)
        R = reduced_f - P
        ZZ = np.round(R,5)
        XX = np.count_nonzero(ZZ==0)
        if XX == len(reduced_f):
            break
            counter += 1
        else:

            delta_disp=np.linalg.solve(reduced_k, R)
            u += delta_disp
            n = n+1
            counter += 1










