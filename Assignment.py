import numpy as np
import heaviside_enrichment
import class_crack
import math as m
import enrichment_functions
import plots
from scipy import linalg
import matplotlib.pyplot as plt

def set_matrix(Nodes, H2, GaussPoint_1to4, D_plane_stress):
    final_matrix = np.zeros([16,16])
    for i in Nodes:
        A = heaviside_enrichment.Heaviside_enrichment(i, H2, GaussPoint_1to4, D_plane_stress)
    return final_matrix


def to_polar_in_radians(x,y):
    '''
    Convertion of cartesian to polar coordinates in radiance
    Parameters
    ----------
    x : x-coordinate of crack tip
    y : y-coordinate of crack tip
    Returns
    -------
    r and, theta in radiance
    '''
    r = np.sqrt(x**2 + y**2)
    theta = m.atan(y/x)
    return np.round(r, 3), np.round(theta, 3)

def remove(v):
    A=tuple(v[0])
    B=tuple(v[1])
    C=tuple(v[2])
    D=tuple(v[3])
    l = [A,B,C,D]
    return l

def filtering(Jarvis, all_ns):
    b=[]
    v=[]
    for i in Jarvis:
        if type(i).__name__ != "NoneType":
                b.append(i)

    result = []
    for r in all_ns:
        if r not in b:
            result.append(r)

    return result,b

def ele_filtering(EL, NL):

    '''
    the function filters the elements which do not require enrichment
    Parameters
    ----------
    EL : List of elements
    NL : nodal list
    '''
    b =[]
    for i in EL:
        outer = list(i.values())
        inner = outer[0]
        for k in NL:
            minus = remove(k)
            if inner == minus:
                b.append(i.keys())

    return b

def G_points(SE_ME):
    GPs = []

    for i in SE_ME:
        GP = np.zeros([4,2])
        # print(i)
        Q1, Q2, Q3, Q4 = i[0], i[1], i[2], i[3]
        lengthX = abs(Q1[0] - Q2[0]) / 4
        lengthY = abs(Q1[1]- Q4[1]) / 4

        GP[0, 0] = Q1[0] + lengthX
        GP[0, 1] = Q1[1] + lengthY

        GP[1, 0] = abs(Q2[0] - lengthX)
        GP[1, 1] = abs(Q2[1] + lengthY)


        GP[2, 0] = abs(Q3[0] - lengthX)
        GP[2, 1] = abs(Q3[1] - lengthY)


        GP[3, 0] = abs(Q4[0] + lengthX)
        GP[3, 1] = abs(Q4[1] - lengthY)
        GPs.append(GP)


    G=[]
    for i in GPs:
        for k in i:
            G.append(k)

    beta = np.asarray(G)
    plt.scatter(beta[:,0], beta[:,1], marker = '*', color = 'r', s=100, label = "GaussPoints")
    plt.legend(loc="upper right")

