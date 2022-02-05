import numpy as np
import uniform_mesh; import matplotlib.pyplot as plt
import plots; #import new_crack
import operator; import heaviside_enrichment
import Tip_enrichment; import math as m
# import Assignment;
import enrichment_functions; from scipy import linalg
from shapely.geometry import Polygon, LineString, Point
from shapely.geometry.polygon import Polygon
from scipy import linalg

#     '''
#     ==========================================================================
#     This functions generates the mesh for the element which is partially cut
#     by the crack.
#      N_4                 N_5               N_6
#     +-------------------+------------------+
#     +                   +                  +
#     +                   +                  +
#     +$$$$$$$$$$$$$$$$$$$$(******)-> partial_crack
#     +                   +                  +
#     +                   +                  +
#     +-------------------+------------------+
#     N_1                 N_2                N_3

#     Suppose, the element is partially cut by the crack(as in the above illustration)
#     and therefore the ""length of the crack is << the element length"", hence,
#     this function creates a subdomain same as the elements which are
#     fully cut by the crack.
#     This function separates the main element into 2 sub domains only in case of
#     the length of the partial-crack << the element length
#     ==========================================================================
#     Parameters
#     ---------------------------------------------------------------------------
#     nodes :  list of 4 nodel coordinates.
#     c_1 : crack_tip 1
#     c_2 : crack_tip 2
#     length_element : Element length
#     Returns
#     ---------------------------------------------------------------------------
#     The output of the function is a nodal points which are used for Tip enrichment.
#     ===========================================================================
#     '''

def updated_cracktip(NODES, ELEMENTS, c_2, GaussPoint_1to4, D_plane_stress, alpha):

    r,theta =[],[]
    Nodes_list = np.array([NODES[0], NODES[1], NODES[2], NODES[3]])
    r1, theta1 = Tip_enrichment.r_theta(Nodes_list[0], c_2, alpha)
    r2, theta2 = Tip_enrichment.r_theta(Nodes_list[1], c_2, alpha)
    r3, theta3 = Tip_enrichment.r_theta(Nodes_list[2], c_2, alpha)
    r4, theta4 = Tip_enrichment.r_theta(Nodes_list[3], c_2, alpha)
    # print(f"{NODES}------------ is full_tip_enriched")
    x=3
    Nodes = (x+1)**2
    Elements = x**2
    Nodes_elements = 4

    domain_nodes = uniform_mesh.nodes(Nodes, Nodes_list, x)
    domain_elements = uniform_mesh.elements(Elements, Nodes_elements, x)

    # plots.plot_sub_elements(domain_nodes)
    W_nodes, low = uniform_mesh.dictionary(domain_nodes,domain_elements)

    Tip_matrix = Tip_enrichment.tip_enrichment(W_nodes, alpha, GaussPoint_1to4, D_plane_stress, c_2, r1, r2, r3, r4,
                                                                theta1, theta2, theta3, theta4)

    # KP = np.round(Tip_matrix,3)
    # print( KP[0:5, 0:5])
    Tip_E = list(ELEMENTS)
    # X = np.linalg.inv(Tip_matrix)
    # print("TRUE for pure TIP")
    # print("not", Tip_E)
    return Nodes_list, Tip_matrix, Tip_E


def updated_heaviside(NODES, ELEMENTS, GaussPoint_1to4, D_plane_stress, cracks):

    Nodes_list = np.array([NODES[0], NODES[1], NODES[2], NODES[3]])
    x = 3
    Nodes = (x+1)**2
    Elements = x**2
    Nodes_elements = 4

    domain_nodes = uniform_mesh.nodes(Nodes, Nodes_list, x)
    domain_elements = uniform_mesh.elements(Elements, Nodes_elements, x)
    H_matrices = []
    # print(f"{Nodes_list} is HEAVISIDE")
    W_nodes, W_elements = uniform_mesh.dictionary(domain_nodes,domain_elements)
    Cs = []

    # for j in range(0,len(cracks)-1):
    #     path = LineString([tuple(cracks[j]), tuple(cracks[j+1])])
    #     polygon = Polygon(NODES)
    #     if path.touches(polygon) or path.intersects(polygon) or path.within(polygon):

    #         Cs.append(cracks[j])
    #         Cs.append(cracks[j+1])

    # print(f"{NODES}------------------------- is HEAVISIDE")
    # plots.plot_sub_elements(domain_nodes)
    # print("cracks", Cs[-2], Cs[-1])
    final_matrix_H = heaviside_enrichment.Heaviside_enrichment(W_nodes, GaussPoint_1to4, D_plane_stress, cracks)
    # KP = np.round(final_matrix_H,3)
    # print("final_matrixH---------------", KP[0:5, 0:5])
    # K_mixed_TH = final_matrix_H
    # X = np.linalg.inv(K_mixed_TH)
    # print("TRUE for pure heavy")
    Z = list(ELEMENTS)
    return Nodes_list, Z, final_matrix_H

def updated_pretipenr(MIX_N, MIX_E, Hside, TS, GaussPoint_1to4, D_plane_stress, alpha, cracks):

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

    # Cs = []
    # Repeat = []
    # for j in range(0,len(c_2)-1):
    #     for i, k in zip(MIX_N, MIX_E):
    #         polygon = Polygon(i)
    #         path = LineString([tuple(c_2[j]), tuple(c_2[j+1])])
    #         if path.intersects(polygon) and i not in Repeat:
    #             Repeat.append(i)
    #             Cs.append(c_2[j])
    #             Cs.append(c_2[j+1])

    # RP = Cs[-1]
    # LP = Cs[-2]
    PT_mats = []
    for NODES, ELEMENTS in zip(MIX_N, MIX_E):
        Nodes_list = np.array([NODES[0], NODES[1], NODES[2], NODES[3]])
        r1, theta1 = Tip_enrichment.r_theta(Nodes_list[0], cracks[-1], alpha)
        r2, theta2 = Tip_enrichment.r_theta(Nodes_list[1], cracks[-1], alpha)
        r3, theta3 = Tip_enrichment.r_theta(Nodes_list[2], cracks[-1], alpha)
        r4, theta4 = Tip_enrichment.r_theta(Nodes_list[3], cracks[-1], alpha)

        x=3
        Nodes = (x+1)**2
        Elements = x**2
        Nodes_elements = 4

        domain_nodes = uniform_mesh.nodes(Nodes, Nodes_list, x)
        domain_elements = uniform_mesh.elements(Elements, Nodes_elements, x)

        S_nodes, low = uniform_mesh.dictionary(domain_nodes,domain_elements)

        E1, E2, E3, E4 = ELEMENTS[0], ELEMENTS[1], ELEMENTS[2], ELEMENTS[3]

        Hs =[]; Ts=[]
        matrices =[]
        for coordinates in S_nodes:
            G,H,I,J = coordinates[0], coordinates[1], coordinates[2], coordinates[3]
            for GP in GaussPoint_1to4:
                xi_1 = GP[0]
                xi_2 = GP[1]
                H1, H2, H3, H4 = 0,0,0,0
                B_std, Nan, N, dN, jacobi, dN_en = enrichment_functions.classic_B_matric(coordinates, D_plane_stress,
                                                                                          xi_1, xi_2, H1, H2, H3, H4)

                if E1 in HS:
                    # print(f"{E1} in HS")
                    H_1 =  enrichment_functions.step_function(G, cracks)
                    BH1 = enrichment_functions.heaviside_function1(dN_en, H_1)
                    Hs.append(BH1)

                elif E1 in TS:
                    F11, F21, F31, F41, dF1 = enrichment_functions.asymptotic_functions(r1, theta1, alpha)
                    T_1 = enrichment_functions.tip_enrichment_func_N1(F11, F21, F31, F41, dN, dF1, N)
                    Ts.append(T_1)

                else:
                    # print(f"{E1} in HS")
                    HS.append(E1)
                    H_1 =  enrichment_functions.step_function(G, cracks)
                    BH1 = enrichment_functions.heaviside_function1(dN_en, H_1)
                    Hs.append(BH1)

                if E2 in HS:
                    # print(f"{E2} in HS")
                    H_2 =  enrichment_functions.step_function(H, cracks)
                    BH2 = enrichment_functions.heaviside_function2(dN_en, H_2)
                    Hs.append(BH2)

                elif E2 in TS:
                    F12, F22, F32, F42, dF2 = enrichment_functions.asymptotic_functions(r2, theta2, alpha)
                    T_2 = enrichment_functions.tip_enrichment_func_N2(F12, F22, F32, F42, dN, dF2, N)
                    Ts.append(T_2)

                else:
                    HS.append(E2)
                    H_2 =  enrichment_functions.step_function(H, cracks)
                    BH2 = enrichment_functions.heaviside_function2(dN_en, H_2)
                    Hs.append(BH2)

                if E3 in HS:
                    H_3 =  enrichment_functions.step_function(I, cracks)
                    BH3 = enrichment_functions.heaviside_function3(dN_en, H_3)
                    Hs.append(BH3)

                elif E3 in TS:
                    F13, F23, F33, F43, dF3 = enrichment_functions.asymptotic_functions(r3, theta3, alpha)
                    T_3 = enrichment_functions.tip_enrichment_func_N3(F13, F23, F33, F43, dN, dF3, N)
                    Ts.append(T_3)

                else:
                    HS.append(E3)
                    H_3 =  enrichment_functions.step_function(I, cracks)
                    BH3 = enrichment_functions.heaviside_function3(dN_en, H_3)
                    Hs.append(BH3)

                if E4 in HS:
                    H_4 =  enrichment_functions.step_function(J, cracks)
                    BH4 = enrichment_functions.heaviside_function4(dN_en, H_4)
                    Hs.append(BH4)

                elif E4 in TS:
                    F14, F24, F34, F44, dF4 = enrichment_functions.asymptotic_functions(r4, theta4, alpha)
                    T_4 = enrichment_functions.tip_enrichment_func_N4(F14, F24, F34, F44, dN, dF4, N)
                    Ts.append(T_4)

                else:
                    HS.append(E4)
                    H_4 =  enrichment_functions.step_function(J, cracks)
                    BH4 = enrichment_functions.heaviside_function4(dN_en, H_4)
                    Hs.append(BH4)

                for B in Hs:
                    B_std = np.concatenate((B_std, B), axis = 1)

                for T in Ts:
                    B_std = np.concatenate((B_std, T), axis = 1)

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
        # print("final_matrixH---------------", PT_matrix.shape)
        # print(KP[20:28, 20:28])
        # K_mixed_TH = PT_matrix
        # X = np.linalg.inv(K_mixed_TH)
        # print("TRUE for pre tip")
        PT_mats.append(PT_matrix)

    return HS, PT_mats

def Normal_blended_en(NON_N, NON_E, HS, TS, GaussPoint_1to4, D_plane_stress, alpha, cracks):

    Normal_N =[]
    Normal_E =[]
    Normal_K=[]
    Blended_K =[]
    Blended_E =[]
    Blended_N =[]
    for NODES, ELEMENTS in zip(NON_N, NON_E):

        Nodes_list = np.array([NODES[0], NODES[1], NODES[2], NODES[3]])

        r1, theta1 = Tip_enrichment.r_theta(Nodes_list[0], cracks[-1], alpha)
        r2, theta2 = Tip_enrichment.r_theta(Nodes_list[1], cracks[-1], alpha)
        r3, theta3 = Tip_enrichment.r_theta(Nodes_list[2], cracks[-1], alpha)
        r4, theta4 = Tip_enrichment.r_theta(Nodes_list[3], cracks[-1], alpha)

        E1, E2, E3, E4 = ELEMENTS[0], ELEMENTS[1], ELEMENTS[2], ELEMENTS[3]

        H_mat =[]; T_mat=[]
        matrices =[]
        G,H,I,J = NODES[0], NODES[1], NODES[2], NODES[3]
        if ((E1 not in TS and E1 not in HS) and (E2 not in TS and E2 not in HS)
            and (E3 not in TS and E3 not in HS) and (E4 not in TS and E4 not in HS)) and NODES not in Normal_N:
                # plt.scatter(Nodes_list[:,0], Nodes_list[:,1], marker ='s', s=100, c="b", )
                # print("normal_N----", NODES)
                # print("normal_E----", ELEMENTS)
                C_M = enrichment_functions.classical_FE(NODES, GaussPoint_1to4, D_plane_stress)
                KP = np.round(C_M,3)
                Normal_E.append(ELEMENTS)
                Normal_N.append(NODES)
                Normal_K.append(C_M)

        else:
            Blended_E.append(ELEMENTS)
            Blended_N.append(NODES)
            # plt.scatter(Nodes_list[:,0], Nodes_list[:,1], marker ='v', s=100, c="r", )
            for GP in GaussPoint_1to4:
                xi_1 = GP[0]
                xi_2 = GP[1]
                H1, H2, H3, H4 = 0,0,0,0
                B_std, Null, N, dN, jacobi, dN_en = enrichment_functions.classic_B_matric(NODES, D_plane_stress,
                                                                                          xi_1, xi_2, H1, H2, H3, H4)

                if E1 in HS:
                    H_1 =  enrichment_functions.step_function(G, cracks)
                    BH1 = enrichment_functions.heaviside_function1(dN_en, H_1)
                    H_mat.append(BH1)

                elif E1 in TS:
                    F11, F21, F31, F41, dF1 = enrichment_functions.asymptotic_functions(r1, theta1, alpha)
                    T_1 = enrichment_functions.tip_enrichment_func_N1(F11, F21, F31, F41, dN, dF1, N)
                    T_mat.append(T_1)


                if E2 in HS:
                    H_2 =  enrichment_functions.step_function(H, cracks)
                    BH2 = enrichment_functions.heaviside_function2(dN_en, H_2)
                    H_mat.append(BH2)

                elif E2 in TS:
                    F12, F22, F32, F42, dF2 = enrichment_functions.asymptotic_functions(r2, theta2, alpha)
                    T_2 = enrichment_functions.tip_enrichment_func_N2(F12, F22, F32, F42, dN, dF2, N)
                    T_mat.append(T_2)

                if E3 in HS:
                    H_3 =  enrichment_functions.step_function(I, cracks)
                    BH3 = enrichment_functions.heaviside_function3(dN_en, H_3)
                    H_mat.append(BH3)

                elif E3 in TS:
                    F13, F23, F33, F43, dF3 = enrichment_functions.asymptotic_functions(r3, theta3, alpha)
                    T_3 = enrichment_functions.tip_enrichment_func_N3(F13, F23, F33, F43, dN, dF3, N)
                    T_mat.append(T_3)

                if E4 in HS:
                    H_4 =  enrichment_functions.step_function(J, cracks)
                    BH4 = enrichment_functions.heaviside_function4(dN_en, H_4)
                    H_mat.append(BH4)

                elif E4 in TS:
                    F14, F24, F34, F44, dF4 = enrichment_functions.asymptotic_functions(r4, theta4, alpha)
                    T_4 = enrichment_functions.tip_enrichment_func_N4(F14, F24, F34, F44, dN, dF4, N)
                    T_mat.append(T_4)

                for B in H_mat:
                    B_std = np.concatenate((B_std, B), axis = 1)

                for T in T_mat:
                    B_std = np.concatenate((B_std, T), axis = 1)

                Bt_D_TH = np.matmul(B_std.T, D_plane_stress)
                Bt_D_B_TH = (np.matmul(Bt_D_TH, B_std)) * linalg.det(jacobi)
                K_mixed_TH = Bt_D_B_TH
                # X = np.linalg.inv(K_mixed_TH)
                # print("TRUE for blended", X)
                F = K_mixed_TH.shape
                matrices.append(K_mixed_TH)
                H_mat.clear()
                T_mat.clear()

            Enr_K = np.zeros([F[1],F[1]])
            for i in matrices:
                Enr_K += i
            Blended_K.append(Enr_K)

    return Normal_K, Blended_K, Normal_E, Blended_E, Normal_N, Blended_N

















