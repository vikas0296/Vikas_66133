'''
AUTHOR : VIKAS MELKOT BALASUBRAMANYAM
MATRICULATION NUMBER : 66133
Personal Programming Project: IMPLEMENTATION OF XFEM AND FRACTURE MECHANICS
#=============================================================================
'''

import matplotlib.pyplot as plt
import numpy as np
import math as m
from scipy.interpolate import Rbf
from mpl_toolkits.axes_grid1 import make_axes_locatable


fig = plt.figure()
fig.set_figwidth(10)
fig.set_figheight(10)

def plot_mesh(NL, A, B, X, length_element):
    '''
    The function is used to plot the mesh
    Parameters
    ----------
    NL : Nodal coordinates [x,y]
    A, B : length of the geometry along x and y axes
    X : no of divisions along x and y direction
    length_element : Length of the element
    '''
    plt.scatter(NL[:,0], NL[:,1], c='black', s=50, label = "Node")
    plt.legend(loc="upper right")
    plt.ylabel('Y-AXIS')
    plt.xlabel('X-AXIS')
    plt.title(f'2D_UNIFORM_MESH of size {X}x{X}')
    fig.savefig("Mesh_generated.png")


def plot_crack(cracktip_1, cracktip_2, length_element):
    '''
    The function plots the crack segment
    Parameters
    ----------
    cracktip_1, cracktip_2: Cracktip coordinates [x1,y1],[x2,y2]
    length_element : Length of the element

    '''
    plt.plot(cracktip_1, cracktip_2, linewidth=3, c="green", label = "Crack")

def DOFs(fixed_nodes):
    '''
    The function scatters the points which have 0 degrees of freedom

    Parameters
    ----------
    fixed_nodes : nodal coordinates [x,y]

    '''
    #forming it as an array
    Z = np.asarray(fixed_nodes)
    plt.scatter(Z[:,0], Z[:,1], c='Blue', s=200, label = "DOFs")
    plt.legend(loc="upper right")

def force(DISPS, label):
    '''
    The function scatters the point whose displacements are known
    Parameters
    ----------
    DISP : nodal coordinates [x,y]
    '''
    #forming it as an array
    Z = np.asarray(DISPS)
    plt.scatter(Z[:,0], Z[:,1], c='orange', s=200, label = label)
    plt.legend(loc="upper right")


def plot_gausspoints(G):
    '''
    The function scatters the Gauss points
    Parameters
    ----------
    G : Gauss point coordinates in global system [x,y]
    '''
    #forming it as an array
    beta = np.asarray(G)
    plt.scatter(beta[:,0], beta[:,1], marker = '*', color = 'r', label = "GaussPoints")
    plt.legend(loc="upper right")

def circle(lc_R, radius):
    '''
    The function is plots a circle around the crack tip for a given radius
    Parameters
    ----------
    lc_R : right hand crack tip [x,y]
    radius : radius of the circle
    '''
    ax = plt.gca()
    circle1 = plt.Circle(lc_R, radius ,linewidth=3, color='darkblue', fill=False)
    ax.add_patch(circle1)

def contour_plot(K, S1, A, No, T, Id):
    '''
    The function plots the stress distribution contour plots
    # the code skeleton has been taken from https://stackoverflow.com/
    Parameters
    ----------
    K : Gauss coordinates [x,y]
    S1 : Stress [val1, val2.........]
    A : length of the geometry along x-axis
    T : 'x' , 'y' or 'xy'
    Id : 'I' or 'II' or 'I_II'
    -------
    Output: plots a contour plot
    '''

    #to get the unique values for x and y coordinate
    Xu = np.unique(K[:,0])
    Yu = np.unique(K[:,1])

    #forming a grid
    grid_x, grid_y = np.mgrid[0:Xu.max():10, 0:Yu.max():10]

    #interpolating the data using Rbf library
    rbfi_1 = Rbf(K[:,0], K[:,1], S1)

    # Predict on the regular grid. Line 5.
    di = rbfi_1(grid_x, grid_y)

    #plotting function
    fig, (ax1) = plt.subplots(1, figsize=(8,8))
    c1 = ax1.imshow(di.T, origin="lower", cmap="hsv", extent=[0, A, 0, A])

    ax1.set_title(f'Distribution of $\sigma_{T}$, N/mm2')
    ax1.set_xlabel('Gauss_x_global')
    ax1.set_ylabel('Gauss_y_global')

    # this code skeleton has been taken from https://stackoverflow.com
    #to make the axes visible
    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes('right', size='8%', pad=0.5)
    fig.colorbar(c1, cax=cax1, orientation='vertical')
    fig.savefig(f"{Id}Stress_plot_{No}.png")
    plt.close()









