import numpy as np
import matplotlib.pyplot as plt
import uniform_mesh 
import heaviside_enrichment

A = 5             #length of the specimen along x and y axis
B = 5
x = 3             #3x3 will be elements, 4x4 will be nodes
# NL, EL = uniform_mesh(A,B,x)

cracktip_1 = np.array([1, 3.5])
cracktip_2 = np.array([1.5, 1.5])
D = 20000
nu = 0.33
D_plane_stress = (D/(1-nu**2)) * np.array([[1, nu, 0],
                                           [nu, 1, 0],
                                           [0, 0, (1-nu)/2]])

GaussPoint_1to4 = np.array([[ 1/np.sqrt(3),  1/np.sqrt(3)],
                            [ 1/np.sqrt(3), -1/np.sqrt(3)],
                            [-1/np.sqrt(3),  1/np.sqrt(3)],
                            [-1/np.sqrt(3), -1/np.sqrt(3)]])

D_plane_strain = (D / (1 + nu)* (1-2*nu)) * np.array([[(1 - nu), nu, 0],
                                                      [nu, (1 - nu), 0],
                                                      [0, 0, (1 - 2*nu)/2]])

condition = False
while condition == False:
    
    FEM = input("Please provide input as '1' for plane_stress and '0' for plane strain to solve the FEM problem: ")
    
    if FEM == '1':
        print("plane_stress")
        condition = True
        
    elif FEM == '0':
        print("plane strain")
        condition = True
        
    else:
        print("*"*50)
        print("Please type '1' or '0' as the input")
        print("*"*50)
        condition = False
    
    
