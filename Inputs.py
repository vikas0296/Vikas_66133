import numpy as np
import matplotlib.pyplot as plt
# import uniform_mesh
# import heaviside_enrichment

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

# south = []
# south_points = []
# west = []
# west_points =[]
# north = []
# north_points = []
# east=[]
# east_points=[]
# DOFs=[]
# side = 'south'
# if side == 'south':
#     for nodes, numbers in zip(Nodes, enumerate(Nodes)):
#         if nodes[0] <= A and nodes[1] < 1+element_length:
#             south.append(nodes)
#             south_points.append(numbers[0])
#             DOFs.append(numbers[0])
#     plots.DOFs(south)

# elif side == 'east':
#     for nodes, numbers in zip(Nodes, enumerate(Nodes)):
#         if nodes[0] == A and nodes[1] <= B:
#             east.append(nodes)
#             east_points.append(numbers[0])
#             DOFs.append(numbers[0])
#     plots.DOFs(east)

# elif side == 'north':
#     for nodes, numbers in zip(Nodes, enumerate(Nodes)):
#         if nodes[0] <= A and nodes[1] == B:
#             north.append(nodes)
#             north_points.append(numbers[0])
#             DOFs.append(numbers[0])
#     plots.DOFs(north)

# elif side == 'west':
#     for nodes, numbers in zip(Nodes, enumerate(Nodes)):
#         if nodes[0] == 1 and nodes[1] <= B:
#             west.append(nodes)
#             west_points.append(numbers[0])
#             DOFs.append(numbers[0])
#     plots.DOFs(west)


# # Application of Neumann and Derechlet boundary conditions


    # location = "3"
    # loc = []
    # loc_elements = []
    # if location == "3":
    #     for N, E in zip(Nodes, enumerate(Nodes)):
    #         if N[0] == 1 and N[1] == B:
    #             loc.append(N)
    #             loc_elements.append(E[0])

    # elif location == '2':
    #     for N, E in zip(Nodes, enumerate(Nodes)):
    #         if N[0] == A and N[1] == B:
    #             loc.append(N)
    #             loc_elements.append(E[0])

    # elif location == '1':
    #     for N, E in zip(Nodes, enumerate(Nodes)):
    #         if N[0] == A and N[1] == 1:
    #             loc.append(N)
    #             loc_elements.append(E[0])

    # elif location == '0':
    #     for N, E in zip(Nodes, enumerate(Nodes)):
    #         if N[0] == 0 and N[1] == 0:
    #             loc.append(N)
    #             loc_elements.append(E[0])
