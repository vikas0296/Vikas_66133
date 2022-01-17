import numpy as np
import heaviside_enrichment
import class_crack
import math as m
import enrichment_functions; import uniform_mesh
import plots
from scipy import linalg
import matplotlib.pyplot as plt

# def waste(ygod, egod, b):
#     for i,j in zip(ygod, egod):
#         if i in b:
#             division = np.array([i[0], i[1], i[2], i[3]])
#             e = np.array([int(j[0]), int(j[1]),int(j[2]), int(j[3])])
#             x=3
#             Nodes = (x+1)**2
#             Elements = x**2
#             Nodes_elements = 4
#             NODES2 = uniform_mesh.nodes(Nodes,division,x)
#             ELEMENTS2 = uniform_mesh.elements(Elements, Nodes_elements, x)
#             plots.plot_sub_elements(NODES2)
#             NODES, ELEMENTS = uniform_mesh.dictionary(NODES2,ELEMENTS2)
#             return NODES


def intersect(c_2, length_element):
    r = 3*np.sqrt(length_element*0.5)
    P1, P2, P3, P4, P5 = [0,0], [0,0], [0,0], [0,0], [0,0]
    P1[0] = abs(c_2[0] - r)
    P1[1] = c_2[1]
    P2[0] = abs(c_2[0] - r)
    P2[1] = c_2[1] + r
    P3[0] = abs(c_2[0] + r)
    P3[1] = c_2[1] + r
    P4[0] = c_2[0] + r
    P4[1] = abs(c_2[1] - r)
    P5[0] = abs(c_2[0] - r)
    P5[1] = abs(c_2[1] - r)
    hexa = [P1, P2, P3, P4, P5, P1]
    plots.extra_dofs(hexa)
    return r, P1, P2, P3, P4, P5

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

    return result, b

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
        # print("GPS****", i)
        for k in i:
            # print(k)
            G.append(k)

    # # print(G)
    # print(f"there are {len(G)} Gauss points in the geometry")
    beta = np.asarray(G)
    plt.scatter(beta[:,0], beta[:,1], marker = '*', color = 'r', s=100, label = "GaussPoints")
    plt.legend(loc="upper right")
    return G, GPs
