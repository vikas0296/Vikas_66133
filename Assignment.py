import numpy as np
import plots
import matplotlib.pyplot as plt

def filtering(EN_list, GL_nodes):
    '''
    This function filters the nodes which do not require enrichment
    Parameters
    ----------
    EN_list : List of nodes which has been enriched
    GL_nodes : list containing all the nodes
    Returns
    -------
    result : Filtered nodes
    b :

    '''
    result = []
    for r in GL_nodes:
        if r not in EN_list:
            result.append(r)

    return result


def E_filter(EN_list, GL_elements):
    '''
    This function filters the elements which do not require enrichment
    Parameters
    ----------
    EN_list : Enriched element list
    GL_elements :  list containing all the elements

    Returns
    -------
    result : filtered elements
    '''

    c= []
    d= []
    result =[]
    for i in GL_elements:
        c.append(tuple(i))

    for j in EN_list:
        d.append(tuple(j))

    for i in c:
        if i not in d:
            result.append(i)
    return result

def G_points(SE_ME):
    '''
    The function plots the Gauss points in the global coordinate system

      N_4                 N_5               N_6
    +-------------------+------------------+
    +                   +  Gauss points    +
    +  x          x     +  x           x   +
    +                   +                  +
    +  x          x     +  x           x   +
    +                   +                  +
    +-------------------+------------------+
    N_1                 N_2                N_3

    In the above illustration, global coordinates of the 'x' positions are calculated from this function

    Parameters
    ----------
    SE_ME : list of nodes to calculate the coordinates of new Gauss points
    Returns
    -------
    G : List of 4 gauss points
    GPs : returns 1 gauss point
    '''
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
        for k in i:
            G.append(k)

    # beta = np.asarray(G)
    # plt.scatter(beta[:,0], beta[:,1], marker = '*', color = 'r', s=100, label = "GaussPoints")
    # plt.legend(loc="upper right")
    return G, GPs





































