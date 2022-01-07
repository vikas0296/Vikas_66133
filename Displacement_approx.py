import numpy as np
from scipy import linalg
import math
import operator
import matplotlib.pyplot as plt
import pandas as pd


def displacement_approximation(N, c_1, c_2, E, r, theta, alpha, Tside, Hside, C_Dofs,
                               disp):

    DISPLACEMENTS = []

    H_dof = len(Hside)*2
    T_dof = len(Tside)*8

    Tside = list(np.asarray(Tside) -1)
    Hside = list(np.asarray(Hside) -1)

    F1 = np.sqrt(r) * np.sin(theta/2)
    F2 = np.sqrt(r) * np.sin(theta/2)
    F3 = np.sqrt(r) * np.sin(theta/2) * np.sin(theta)
    F4 = np.sqrt(r) * np.cos(theta/2) * np.sin(theta)

    for i, node in zip(list(E), N):
        Ux_Uy = [0,0,0,0,0,0,0,0]
        i = list(np.asarray(i) - 1)
        # print(i)
        for j, n in zip(i,node):
            Ux_Uy[i.index(j)*2] = disp[int(j)*2]
            Ux_Uy[i.index(j)*2+1] = disp[int(j)*2+1]
            if j in Hside:
                for k in Hside:
                    if j == k:
                        if n[1] < c_1[1]:
                            H1 = 0
                            # print("-----------", n)
                            # print(f"{n}is below {c_1[1]}", H1)
                            # print(f"{j}  in Hside{k} and {Hside.index(k)}")
                            Ux_Uy[i.index(j)*2] = disp[int(j)*2] + disp[(C_Dofs + Hside.index(k)*2)] * H1
                            Ux_Uy[i.index(j)*2+1] = disp[int(j)*2+1] + disp[(C_Dofs + Hside.index(k)*2+1)] * H1
                        else:
                            H1 = 1
                            # print("-----------", n)
                            # print(f"{n}, is above {c_1[1]}", H1)
                            # print(f"{j}  in Hside{k} and {Hside.index(k)}")
                            Ux_Uy[i.index(j)*2] = disp[int(j)*2] + disp[(C_Dofs + Hside.index(k)*2)] * H1
                            Ux_Uy[i.index(j)*2+1] = disp[int(j)*2+1] + disp[(C_Dofs + Hside.index(k)*2+1)] * H1

            elif j in Tside:
                for l in Tside:
                    if j == l:
                        # print(f"{j}  in Tside{l} and {Tside.index(l)}")
                        Ux_Uy[i.index(j)*2] = (disp[int(j)*2] + disp[(C_Dofs + H_dof + Tside.index(l)*8)] * F1 +
                                                disp[(C_Dofs + H_dof + Tside.index(l)*8)+2] * F2 +
                                                disp[(C_Dofs + H_dof + Tside.index(l)*8)+4] * F3 +
                                                disp[(C_Dofs + H_dof + Tside.index(l)*8)+6] * F4)

                        Ux_Uy[i.index(j)*2+1] = (disp[int(j)*2+1] + disp[(C_Dofs + H_dof + Tside.index(l)*8+1)] * F1 +
                                                  disp[(C_Dofs + H_dof + Tside.index(l)*8+3)] * F2 +
                                                  disp[(C_Dofs + H_dof + Tside.index(l)*8+5)] * F3 +
                                                  disp[(C_Dofs + H_dof + Tside.index(l)*8+7)] * F4)

        Uxy = np.nan_to_num(Ux_Uy)

        DISPLACEMENTS.append(Uxy)

    return DISPLACEMENTS
