import numpy as np
from scipy import linalg
import math as m
import enrichment_functions

def conditions(E, c_1, c_2, Element_length, r, theta, alpha, D_plane_stress, GaussPoint_1to4):

    F1, F2, F3, F4, dF = enrichment_functions.asymptotic_functions(r, theta, alpha)

#==================="conditions to check when length==5"========================

    if len(E) == 5:
         print("length is 5")
         store5 = []
         L_0 = E[-1]
         X0 = E[L_0]
         E.remove(L_0)
         W = E[0]
         for points in GaussPoint_1to4:
            xi_1 = points[0]
            xi_2 = points[1]
            if W[1] > c_2[1]:
                H2=1
                B_std_5, B_heavy_5, N, dN, jacobi = enrichment_functions.classic_B_matric(E, D_plane_stress,
                                                                                          xi_1, xi_2, H2)
            else:
                H2 = 0
                B_std_5, B_heavy_5, N, dN, jacobi = enrichment_functions.classic_B_matric(E, D_plane_stress,
                                                                                          xi_1, xi_2, H2)

# conditions for last index=========================================
#====================== L_1 == 0 ====================================

            dist0 = abs(c_2[0] - X0[0])

            if L_0 == 0 and X0[1] > c_2[1] and dist0 > Element_length:
                 print(f"Node {X0} to Heaviside")
                 B_5 = enrichment_functions.heaviside_function1(B_heavy_5)

            elif L_0 == 0 and X0[1] > c_2[1] and dist0 <= Element_length:
                 print(f"Node {X0} to be Tip_enriched")
                 B_5 = enrichment_functions.tip_enrichment_func_N1(F1, F2, F3, F4, dN, dF, N)

            elif L_0 == 0 and X0[1] < c_2[1] and dist0 > Element_length:
                 print(f"Node {X0} to Heaviside")
                 B_5 = enrichment_functions.heaviside_function1(B_heavy_5)

            elif L_0 == 0 and X0[1] < c_2[1] and dist0 <= Element_length:
                 print(f"Node {X0} to be Tip_enriched")
                 B_5 = enrichment_functions.tip_enrichment_func_N1(F1, F2, F3, F4, dN, dF, N)

#====================== L_1 == 1 ====================================

            if L_0 == 1 and X0[1] > c_2[1] and dist0 > Element_length:
                 print(f"Node {X0} to Heaviside")
                 B_5 = enrichment_functions.heaviside_function2(B_heavy_5)

            elif L_0 == 1 and X0[1] > c_2[1] and dist0 <= Element_length:
                 print(f"Node {X0} to be Tip_enriched")
                 B_5 = enrichment_functions.tip_enrichment_func_N2(F1, F2, F3, F4, dN, dF, N)

            elif L_0 == 1 and X0[1] < c_2[1] and dist0 > Element_length:
                 print(f"Node {X0} to Heaviside")
                 B_5 = enrichment_functions.heaviside_function2(B_heavy_5)

            elif L_0 == 1 and X0[1] < c_2[1] and dist0 <= Element_length:
                 print(f"Node {X0} to be Tip_enriched")
                 B_5 = enrichment_functions.tip_enrichment_func_N2(F1, F2, F3, F4, dN, dF, N)

#====================== L_1 == 2 ====================================

            if L_0 == 2 and X0[1] > c_2[1] and dist0 > Element_length:
                 print(f"Node {X0} to Heaviside")
                 B_5 = enrichment_functions.heaviside_function3(B_heavy_5)

            elif L_0 == 2 and X0[1] > c_2[1] and dist0 <= Element_length:
                 print(f"Node {X0} to be Tip_enriched")
                 B_5 = enrichment_functions.tip_enrichment_func_N3(F1, F2, F3, F4, dN, dF, N)

            elif L_0 == 2 and X0[1] < c_2[1] and dist0 > Element_length:
                 print(f"Node {X0} to Heaviside")
                 B_5 = enrichment_functions.heaviside_function3(B_heavy_5)

            elif L_0 == 2 and X0[1] < c_2[1] and dist0 <= Element_length:
                 print(f"Node {X0} to be Tip_enriched")
                 B_5 = enrichment_functions.tip_enrichment_func_N3(F1, F2, F3, F4, dN, dF, N)

#====================== L_1 == 3 ====================================

            if L_0 == 3 and X0[1] > c_2[1] and dist0 > Element_length:
                 print(f"Node {X0} to Heaviside")
                 B_5 = enrichment_functions.heaviside_function4(B_heavy_5)

            elif L_0 == 3 and X0[1] > c_2[1] and dist0 <= Element_length:
                 print(f"Node {X0} to be Tip_enriched")
                 B_5 = enrichment_functions.tip_enrichment_func_N4(F1, F2, F3, F4, dN, dF, N)

            elif L_0 == 3 and X0[1] < c_2[1] and dist0 > Element_length:
                 print(f"Node {X0} to Heaviside")
                 B_5 = enrichment_functions.heaviside_function4(B_heavy_5)

            elif L_0 == 3 and X0[1] < c_2[1] and dist0 <= Element_length:
                 print(f"Node {X0} to be Tip_enriched")
                 B_5 = enrichment_functions.tip_enrichment_func_N4(F1, F2, F3, F4, dN, dF, N)

            S5 = np.concatenate((B_std_5, B_5), axis=1)
            Bt_D_1 = np.matmul(S5.T, D_plane_stress)
            Bt_D_B1 = (np.matmul(Bt_D_1, S5)) * linalg.det(jacobi)
            K5 = np.round(Bt_D_B1, 3)
            Q = S5.shape
            Zero_5 = np.zeros([Q[1],Q[1]])
            store5.append(K5)


         # print(len(Zero_5))
         for i in store5:
            Zero_5 += i
         # print(Zero_5[0:5,0:5])

#==============================================================================
#=========================="conditions to check if length==6"==================

    if len(E) == 6:
         print("length is 6")
         store6 = []
         L_1 = E[-2]
         X1 = E[L_1]
         L_2 = E[-1]
         X2 = E[L_2]
         E.remove(L_1)
         E.remove(L_2)
         V = E[0]
         dist = abs(c_2[0] - X1[0])
         print("-------", E)
         for points in GaussPoint_1to4:
            xi_1 = points[0]
            xi_2 = points[1]
            if V[1] > c_2[1]:
                H2 = 1
                B_std_6, B_heavy_6, N, dN, jacobi = enrichment_functions.classic_B_matric(E, D_plane_stress,
                                                                                 xi_1, xi_2, H2)
            else:
                H2 = 0
                B_std_6, B_heavy_6, N, dN, jacobi = enrichment_functions.classic_B_matric(E, D_plane_stress,
                                                                                  xi_1, xi_2, H2)

# conditions for 2nd last index=========================================
#================================ L_1 == 0 ====================================
            if L_1 == 0 and X1[1] > c_2[1] and dist > Element_length:
                  print(f"Node {X1} to Heaviside")
                  B_rand1 = enrichment_functions.heaviside_function1(B_heavy_6)

            elif L_1 == 0 and X1[1] > c_2[1] and dist <= Element_length:
                  print(f"Node {X1} to Tip_enriched")
                  B_rand1 = enrichment_functions.tip_enrichment_func_N1(F1, F2, F3, F4, dN, dF, N)

            elif L_1 == 0 and X1[1] < c_2[1] and dist > Element_length:
                  print(f"Node {X1} to Heaviside")
                  B_rand1 = enrichment_functions.heaviside_function1(B_heavy_6)

            elif L_1 == 0 and X1[1] < c_2[1] and dist <= Element_length:
                  print(f"Node {X1} to Tip_enriched")
                  B_rand1 = enrichment_functions.tip_enrichment_func_N1(F1, F2, F3, F4, dN, dF, N)

#================================ L_1 == 1 ====================================

            if L_1 == 1 and X1[1] > c_2[1] and dist > Element_length:
                  print(f"Node {X1} to Heaviside")
                  B_rand1 = enrichment_functions.heaviside_function2(B_heavy_6)

            elif L_1 == 1 and X1[1] > c_2[1] and dist <= Element_length:
                  print(f"Node {X1} to Tip_enriched")
                  B_rand1 = enrichment_functions.tip_enrichment_func_N2(F1, F2, F3, F4, dN, dF, N)

            elif L_1 == 1 and X1[1] < c_2[1] and dist > Element_length:
                  print(f"Node {X1} to Heaviside")
                  B_rand1 = enrichment_functions.heaviside_function2(B_heavy_6)

            elif L_1 == 1 and X1[1] < c_2[1] and dist <= Element_length:
                  print(f"Node {X1} to Tip_enriched")
                  B_rand1 = enrichment_functions.tip_enrichment_func_N2(F1, F2, F3, F4, dN, dF, N)

#================================ L_1 == 2 ====================================

            if L_1 == 2 and X1[1] > c_2[1] and dist > Element_length:
                  print(f"Node {X1} to Heaviside")
                  B_rand1 = enrichment_functions.heaviside_function3(B_heavy_6)

            elif L_1 == 2 and X1[1] > c_2[1] and dist <= Element_length:
                  print(f"Node {X1} to Tip_enriched")
                  B_rand1 = enrichment_functions.tip_enrichment_func_N3(F1, F2, F3, F4, dN, dF, N)

            elif L_1 == 2 and X1[1] < c_2[1] and dist > Element_length:
                  print(f"Node {X1} to Heaviside")
                  B_rand1 = enrichment_functions.heaviside_function3(B_heavy_6)

            elif L_1 == 2 and X1[1] < c_2[1] and dist <= Element_length:
                  print(f"Node {X1} to Tip_enriched")
                  B_rand1 = enrichment_functions.tip_enrichment_func_N3(F1, F2, F3, F4, dN, dF, N)

#====================== L_1 == 3 ==============================================

            if L_1 == 3 and X1[1] > c_2[1] and dist > Element_length:
                  print(f"Node {X1} to Heaviside")
                  B_rand1 = enrichment_functions.heaviside_function4(B_heavy_6)

            elif L_1 == 3 and X1[1] > c_2[1] and dist <= Element_length:
                  print(f"Node {X1} to Tip_enriched")
                  B_rand1 = enrichment_functions.tip_enrichment_func_N4(F1, F2, F3, F4, dN, dF, N)

            elif L_1 == 3 and X1[1] < c_2[1] and dist > Element_length:
                  print(f"Node {X1} to Heaviside")
                  B_rand1 = enrichment_functions.heaviside_function4(B_heavy_6)

            elif L_1 == 3 and X1[1] < c_2[1] and dist <= Element_length:
                  print(f"Node {X1} to Tip_enriched")
                  B_rand1 = enrichment_functions.tip_enrichment_func_N4(F1, F2, F3, F4, dN, dF, N)

#==============================================================================
#==============================================================================

# conditions for last index====================================================
#================================ L_2 == 0 ====================================

            dist2 = abs(c_2[0] - X2[0])
            if L_2 == 0 and X2[1] > c_2[1] and dist2 > Element_length:
                  print(f"Node {X2} to Heaviside")
                  B_rand2 = enrichment_functions.heaviside_function1(B_heavy_6)

            elif L_2 == 0 and X2[1] > c_2[1] and dist2 <= Element_length:
                  print(f"Node {X2} to Tip_enriched")
                  B_rand2 = enrichment_functions.tip_enrichment_func_N1(F1, F2, F3, F4, dN, dF, N)

            elif L_2 == 0 and X2[1] < c_2[1] and dist2 > Element_length:
                  print(f"Node {X2} to Heaviside")
                  B_rand2 = enrichment_functions.heaviside_function1(B_heavy_6)

            elif L_2 == 0 and X2[1] < c_2[1] and dist2 <= Element_length:
                  print(f"Node {X2} to Tip_enriched")
                  B_rand2 = enrichment_functions.tip_enrichment_func_N1(F1, F2, F3, F4, dN, dF, N)

#================================ L_2 == 1 ====================================

            if L_2 == 1 and X2[1] > c_2[1] and dist2 > Element_length:
                print(f"Node {X2} to Heaviside")
                B_rand2 = enrichment_functions.heaviside_function2(B_heavy_6)

            elif L_2 == 1 and X2[1] > c_2[1] and dist2 <= Element_length:
                print(f"Node {X2} to Tip_enriched")
                B_rand2 = enrichment_functions.tip_enrichment_func_N2(F1, F2, F3, F4, dN, dF, N)

            elif L_2 == 1 and X2[1] < c_2[1] and dist2 > Element_length:
                print(f"Node {X2} to Heaviside")
                B_rand2 = enrichment_functions.heaviside_function2(B_heavy_6)

            elif L_2 == 1 and X2[1] < c_2[1] and dist2 <= Element_length:
                print(f"Node {X2} to Tip_enriched")
                B_rand2 = enrichment_functions.tip_enrichment_func_N2(F1, F2, F3, F4, dN, dF, N)

#====================== L_2 == 2 ====================================

            if L_2 == 2 and X2[1] > c_2[1] and dist2 > Element_length:
               print(f"Node {X2} to Heaviside")
               B_rand2 = enrichment_functions.heaviside_function3(B_heavy_6)

            elif L_2 == 2 and X2[1] > c_2[1] and dist2 <= Element_length:
               print(f"Node {X2} to Tip_enriched")
               B_rand2 = enrichment_functions.tip_enrichment_func_N3(F1, F2, F3, F4, dN, dF, N)

            elif L_2 == 2 and X2[1] < c_2[1] and dist2 > Element_length:
               print(f"Node {X2} to Heaviside")
               B_rand2 = enrichment_functions.heaviside_function3(B_heavy_6)

            elif L_2 == 2 and X2[1] < c_2[1] and dist2 <= Element_length:
               print(f"Node {X2} to Tip_enriched")
               B_rand2 = enrichment_functions.tip_enrichment_func_N3(F1, F2, F3, F4, dN, dF, N)

# #====================== L_2 == 3 ====================================

            if L_2 == 3 and X2[1] > c_2[1] and dist2 > Element_length:
               print(f"Node {X2} to Heaviside")
               B_rand2 = enrichment_functions.heaviside_function4(B_heavy_6)

            elif L_2 == 3 and X2[1] > c_2[1] and dist2 <= Element_length:
               print(f"Node {X2} to Tip_enriched")
               B_rand2 = enrichment_functions.tip_enrichment_func_N4(F1, F2, F3, F4, dN, dF, N)

            elif L_2 == 3 and X2[1] < c_2[1] and dist2 > Element_length:
               print(f"Node {X2} to Heaviside")
               B_rand2 = enrichment_functions.heaviside_function4(B_heavy_6)

            elif L_2 == 3 and X2[1] < c_2[1] and dist2 <= Element_length:
               print(f"Node {X2} to Tip_enriched")
               B_rand2 = enrichment_functions.tip_enrichment_func_N4(F1, F2, F3, F4, dN, dF, N)

            S6 = np.concatenate((B_std_6, B_rand1, B_rand2), axis=1)
            Bt_D_1 = np.matmul(S6.T, D_plane_stress)
            Bt_D_B1 = (np.matmul(Bt_D_1, S6)) * linalg.det(jacobi)
            K6 = np.round(Bt_D_B1, 3)
            Q = S6.shape
            Zero_6 = np.zeros([Q[1],Q[1]])
            store6.append(K6)

         for i in store6:
           Zero_6 += i
         print(Zero_6[0:8,0:8])
         # print(f"--{E} is affected--5----")


#==============================================================================
#======================="conditions to check when length==7"===================


    if len(E) == 7:
         print("length is 7")
         store7 = []
         L_1 = E[-3]
         X1 = E[L_1]

         L_2 = E[-2]
         X2 = E[L_2]

         L_3 = E[-1]
         X3 = E[L_3]

         E.remove(L_1)
         E.remove(L_2)
         E.remove(L_3)

         for points in GaussPoint_1to4:
            xi_1 = points[0]
            xi_2 = points[1]
            B_std_7, B_heavy_7, N, dN, jacobi = enrichment_functions.classic_B_matric(E, D_plane_stress, xi_1, xi_2)

#==================== conditions for E[-3]=====================================
#========================== L_1 == 0 ==========================================
            dist = abs(c_2[0] - X1[0])
            if L_1 == 0 and X1[1] > c_2[1] and dist > Element_length:
              print("Node 0 to Heaviside", X1)
              B_rand1 = enrichment_functions.heaviside_function1(B_heavy_7)

            elif L_1 == 0 and X1[1] > c_2[1] and dist <= Element_length:
              print("Node 0 to Tip_enriched",X1)
              B_rand1 = enrichment_functions.tip_enrichment_func_N1(F1, F2, F3, F4, dN, dF, N)

            elif L_1 == 0 and X1[1] < c_2[1] and dist > Element_length:
              print("Node 0 to Heaviside", X1)
              B_rand1 = enrichment_functions.heaviside_function1(B_heavy_7)

            elif L_1 == 0 and X1[1] < c_2[1] and dist <= Element_length:
              print("Node 0 to Tip_enriched",X1)
              B_rand1 = enrichment_functions.tip_enrichment_func_N1(F1, F2, F3, F4, dN, dF, N)

#================================ L_1 == 1 ====================================

            if L_1 == 1 and X1[1] > c_2[1] and dist > Element_length:
                print("Heaviside1",X1)
                B_rand1 = enrichment_functions.heaviside_function2(B_heavy_7)

            elif L_1 == 1 and X1[1] > c_2[1] and dist <= Element_length:
               print("Tip_enriched1",X1)
               B_rand1 = enrichment_functions.tip_enrichment_func_N2(F1, F2, F3, F4, dN, dF, N)

            elif L_1 == 1 and X1[1] < c_2[1] and dist > Element_length:
               print("Heaviside1", X1)
               B_rand1 = enrichment_functions.heaviside_function2(B_heavy_7)

            elif L_1 == 1 and X1[1] < c_2[1] and dist <= Element_length:
               print("Tip_enriched1", X1)
               B_rand1 = enrichment_functions.tip_enrichment_func_N2(F1, F2, F3, F4, dN, dF, N)

#====================== L_1 == 2 ==============================================

            if L_1 == 2 and X1[1] > c_2[1] and dist > Element_length:
              print("Heaviside2", X1)
              B_rand1 = enrichment_functions.heaviside_function3(B_heavy_7)

            elif L_1 == 2 and X1[1] > c_2[1] and dist <= Element_length:
              print("Tip_enriched2", X1)
              B_rand1 = enrichment_functions.tip_enrichment_func_N3(F1, F2, F3, F4, dN, dF, N)

            elif L_1 == 2 and X1[1] < c_2[1] and dist > Element_length:
              print("Heaviside2", X1)
              B_rand1 = enrichment_functions.heaviside_function3(B_heavy_7)

            elif L_1 == 2 and X1[1] < c_2[1] and dist <= Element_length:
              print("Tip_enriched2", X1)
              B_rand1 = enrichment_functions.tip_enrichment_func_N3(F1, F2, F3, F4, dN, dF, N)

#====================== L_1 == 3 ==============================================

            if L_1 == 3 and X1[1] > c_2[1] and dist > Element_length:
              print("Heaviside3", X1)
              B_rand1 = enrichment_functions.heaviside_function4(B_heavy_7)

            elif L_1 == 3 and X1[1] > c_2[1] and dist <= Element_length:
              print("Tip_enriched3", X1)
              B_rand1 = enrichment_functions.tip_enrichment_func_N4(F1, F2, F3, F4, dN, dF, N)

            elif L_1 == 3 and X1[1] < c_2[1] and dist > Element_length:
              print("Heaviside3", X1)
              B_rand1 = enrichment_functions.heaviside_function4(B_heavy_7)

            elif L_1 == 3 and X1[1] < c_2[1] and dist <= Element_length:
              print("Tip_enriched3", X1)
              B_rand1 = enrichment_functions.tip_enrichment_func_N4(F1, F2, F3, F4, dN, dF, N)

#==================== conditions for E[-2]=====================================
#====================== L_2 == 0 ==============================================

            dist2 = abs(c_2[0] - X2[0])
            if L_2 == 0 and X2[1] > c_2[1] and dist2 > Element_length:
                print("Heaviside0", X2)
                B_rand2 = enrichment_functions.heaviside_function1(B_heavy_7)

            elif L_2 == 0 and X2[1] > c_2[1] and dist2 <= Element_length:
                print("Tip_enriched0", X2)
                B_rand2 = enrichment_functions.tip_enrichment_func_N1(F1, F2, F3, F4, dN, dF, N)

            elif L_2 == 0 and X2[1] < c_2[1] and dist2 > Element_length:
                print("Heaviside0", X2)
                B_rand2 = enrichment_functions.heaviside_function1(B_heavy_7)

            elif L_2 == 0 and X2[1] < c_2[1] and dist2 <= Element_length:
                print("Tip_enriched0", X2)
                B_rand2 = enrichment_functions.tip_enrichment_func_N1(F1, F2, F3, F4, dN, dF, N)

#====================== L_2 == 1 ==============================================

            if L_2 == 1 and X2[1] > c_2[1] and dist2 > Element_length:
                print("Heaviside1", X2)
                B_rand2 = enrichment_functions.heaviside_function2(B_heavy_7)

            elif L_2 == 1 and X2[1] > c_2[1] and dist2 <= Element_length:
                print("Tip_enriched1", X2)
                B_rand2 = enrichment_functions.tip_enrichment_func_N2(F1, F2, F3, F4, dN, dF, N)

            elif L_2 == 1 and X2[1] < c_2[1] and dist2 > Element_length:
                print("Heaviside1", X2)
                B_rand2 = enrichment_functions.heaviside_function2(B_heavy_7)

            elif L_2 == 1 and X2[1] < c_2[1] and dist2 <= Element_length:
                print("Tip_enriched1", X2)
                B_rand2 = enrichment_functions.tip_enrichment_func_N2(F1, F2, F3, F4, dN, dF, N)

#====================== L_2 == 2 ==============================================

            if L_2 == 2 and X2[1] > c_2[1] and dist2 > Element_length:
                print("Heaviside2", X2)
                B_rand2 = enrichment_functions.heaviside_function3(B_heavy_7)
            elif L_2 == 2 and X2[1] > c_2[1] and dist2 <= Element_length:
                print("Tip_enriched2", X2)
                B_rand2 = enrichment_functions.tip_enrichment_func_N3(F1, F2, F3, F4, dN, dF, N)

            elif L_2 == 2 and X2[1] < c_2[1] and dist2 > Element_length:
                print("Heaviside2", X2)
                B_rand2 = enrichment_functions.heaviside_function3(B_heavy_7)

            elif L_2 == 2 and X2[1] < c_2[1] and dist2 <= Element_length:
                print("Tip_enriched2", X2)
                B_rand2 = enrichment_functions.tip_enrichment_func_N3(F1, F2, F3, F4, dN, dF, N)

#====================== L_2 == 3 ==============================================

            if L_2 == 3 and X2[1] > c_2[1] and dist2 > Element_length:
                print("Heaviside3", X2)
                B_rand2 = enrichment_functions.heaviside_function4(B_heavy_7)

            elif L_2 == 3 and X2[1] > c_2[1] and dist2 <= Element_length:
                print("Tip_enriched3", X2)
                B_rand2 = enrichment_functions.tip_enrichment_func_N4(F1, F2, F3, F4, dN, dF, N)

            elif L_2 == 3 and X2[1] < c_2[1] and dist2 > Element_length:
                print("Heaviside3", X2)
                B_rand2 = enrichment_functions.heaviside_function4(B_heavy_7)

            elif L_2 == 3 and X2[1] < c_2[1] and dist2 <= Element_length:
                print("Tip_enriched3", X2)
                B_rand2 = enrichment_functions.tip_enrichment_func_N4(F1, F2, F3, F4, dN, dF, N)


#==================== conditions for E[-1]=====================================
#========================= L_3 == 0 ===========================================
            dist3 = abs(c_2[0] - X3[0])
            if L_3 == 0 and X3[1] > c_2[1] and dist3 > Element_length:
                print("Heaviside0", X3)
                B_rand3 = enrichment_functions.heaviside_function1(B_heavy_7)

            elif L_3 == 0 and X3[1] > c_2[1] and dist3 <= Element_length:
                print("Tip_enriched0", X3)
                B_rand3 = enrichment_functions.tip_enrichment_func_N1(F1, F2, F3, F4, dN, dF, N)

            elif L_3 == 0 and X3[1] < c_2[1] and dist3 > Element_length:
                print("Heaviside0", X3)
                B_rand3 = enrichment_functions.heaviside_function1(B_heavy_7)

            elif L_3 == 0 and X3[1] < c_2[1] and dist3 <= Element_length:
                print("Tip_enriched0", X3)
                B_rand3 = enrichment_functions.tip_enrichment_func_N1(F1, F2, F3, F4, dN, dF, N)

#========================= L_3 == 1 ===========================================

            if L_3 == 1 and X3[1] > c_2[1] and dist3 > Element_length:
                print("Heaviside0", X3)
                B_rand3 = enrichment_functions.heaviside_function2(B_heavy_7)

            elif L_3 == 1 and X3[1] > c_2[1] and dist3 <= Element_length:
                print("Tip_enriched0", X3)
                B_rand3 = enrichment_functions.tip_enrichment_func_N2(F1, F2, F3, F4, dN, dF, N)

            elif L_3 == 1 and X3[1] < c_2[1] and dist3 > Element_length:
                print("Heaviside0", X3)
                B_rand3 = enrichment_functions.heaviside_function2(B_heavy_7)

            elif L_3 == 1 and X3[1] < c_2[1] and dist3 <= Element_length:
                print("Tip_enriched0", X3)
                B_rand3 = enrichment_functions.tip_enrichment_func_N2(F1, F2, F3, F4, dN, dF, N)

#========================= L_3 == 2 ===========================================

            if L_3 == 2 and X3[1] > c_2[1] and dist3 > Element_length:
                print("Heaviside2", X3)
                B_rand3 = enrichment_functions.heaviside_function3(B_heavy_7)

            elif L_3 == 2 and X3[1] > c_2[1] and dist3 <= Element_length:
                print("Tip_enriched2", X3)
                B_rand3 = enrichment_functions.tip_enrichment_func_N3(F1, F2, F3, F4, dN, dF, N)

            elif L_3 == 2 and X3[1] < c_2[1] and dist3 > Element_length:
                print("Heaviside2", X3)
                B_rand3 = enrichment_functions.heaviside_function3(B_heavy_7)

            elif L_3 == 2 and X3[1] < c_2[1] and dist3 <= Element_length:
                print("Tip_enriched2", X3)
                B_rand3 = enrichment_functions.tip_enrichment_func_N3(F1, F2, F3, F4, dN, dF, N)

#========================= L_3 == 3 ===========================================

            if L_3 == 3 and X3[1] > c_2[1] and dist3 > Element_length:
                print("Heaviside3", X3)
                B_rand3 = enrichment_functions.heaviside_function4(B_heavy_7)

            elif L_3 == 3 and X3[1] > c_2[1] and dist3 <= Element_length:
                print("Tip_enriched3", X3)
                B_rand3 = enrichment_functions.tip_enrichment_func_N4(F1, F2, F3, F4, dN, dF, N)

            elif L_3 == 3 and X3[1] < c_2[1] and dist3 > Element_length:
                print("Heaviside3", X3)
                B_rand3 = enrichment_functions.heaviside_function4(B_heavy_7)

            elif L_3 == 3 and X3[1] < c_2[1] and dist3 <= Element_length:
                print("Tip_enriched3", X3)
                B_rand3 = enrichment_functions.tip_enrichment_func_N4(F1, F2, F3, F4, dN, dF, N)

            S7 = np.concatenate((B_std_7, B_rand1, B_rand2, B_rand3), axis=1)
            Bt_D_1 = np.matmul(S7.T, D_plane_stress)
            Bt_D_B1 = (np.matmul(Bt_D_1, S7)) * linalg.det(jacobi)
            K7 = np.round(Bt_D_B1, 3)
            store7.append(K7)

         Q = S7.shape
         Zero_7 = np.zeros([Q[1],Q[1]])
         for i in store7:
             Zero_7 += i
        # print(Zero_7)
         print(f"--{E} is affected--5----")

