import numpy as np
import heaviside_enrichment
import class_crack
import math as m
import enrichment_functions
import plots
from scipy import linalg

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



