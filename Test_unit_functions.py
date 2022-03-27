'''
AUTHOR : VIKAS MELKOT BALASUBRAMANYAM
MATRICULATION NUMBER : 66133
Personal Programming Project: IMPLEMENTATION OF XFEM AND FRACTURE MECHANICS
#=============================================================================
'''

import numpy as np
import uniform_mesh
import J_integral
import math as m
import matplotlib.pyplot as plt
import enrichment_functions
from scipy import linalg
import KU_F; import Assignment

def areSame(A,B): #code snippet taken from https://www.tutorialspoint.com
   '''
   The function checks if the elements in A are equal to elements in B
   A and B are the matrices of order nxn
   '''
   for i in range(len(A)-1):
      for j in range(len(B)-1):
         if (A[i][j] != B[i][j]):
            return 0
   return 1
#-(1)-------------------------test_uniform_mesh---------------------------------

def test_uniform_mesh():
    '''
    This function checks if the nodal coordinates are generated properly
    '''
    #defining geometric dimensions and divisions
    A = 1
    B = 1
    x = 1

    #callin the actual function
    output_N, output_E = uniform_mesh.mesh(A,B,x)
    #defining the expected results beforehand
    expected_nodes = [[0, 0],[1, 0],[0, 1],[1, 1]]
    expcted_elements = [1,2,4,3]

    #comparing the nodes
    # test nodes(Nodes, corners, x)
    for i,j in zip(output_N,expected_nodes):

        if i[0] != j[0]:
            print("Nodal positions are wrong")
            assert False
    else:
        print("Actual output", output_N)
        print("Expected output", expected_nodes)
        print("Nodal positions have been calculated properly")
        assert True

    #comparing the elements
    #test elements(Elements, Nodes_elements, x)
    ELE = output_E
    for i,j in zip(ELE[0],expcted_elements):
        if i != j:
            print("Elements are not properly numbered")
            assert False
    else:
        print("Actual output", output_E)
        print("Expected output", expcted_elements)
        assert True

#-(2)-------------------------test_kinematics---------------------------------
def test_kinematics():
    '''
    The function checks if the strains are generated properly
    '''
    #predefining the desplacement matrix
    Displacements = np.array([[1, 1],
                              [2, 2]])

    #calling actual function
    E1, E2, E3, E4 = J_integral.Kinematics(Displacements)

    #expected strain variables
    expected_strains = np.array([[1, 1.5],
                                 [1.5, 2]])
    #formatting the actual output in the form of an array
    output = np.array([[E1, E2],
                       [E3, E4]])
    #comprisons
    if (areSame(output, expected_strains)==1):
        print("Strain elements are identical and the output is as expected")
        print("Actual output", output)
        print("Expected output", expected_strains)
        assert True
    else:
        print("Strain elements are not identical")
        assert False

#-(3)-------------------------test_Global_to_local_CT---------------------------------
def test_Global_to_local_CT():
    '''
    The function checks if the matrix transformation is proper or not
    '''
    #defining a random angle
    alpha = 90
    #defining a random list
    rand_matrix = [1, 2, 3]

    #defining a random matrix
    random = np.array([[1, 3],
                       [3, 2]])

    #defining the rotation matrix
    CTCS = np.array([[m.cos(alpha), m.sin(alpha)],
                     [-m.sin(alpha), m.cos(alpha)]])

    #calling the actual function
    S1, S2, S3, S4 = J_integral.Global_to_local_CT(rand_matrix, CTCS)
    #performing the transformaton operation
    Exp_matrix = CTCS @ random @ CTCS.T
    #formatting actual output
    output = np.array([[S1, S2],
                       [S3, S4]])

    #comparing the results
    if (areSame(output, Exp_matrix)==1):
        print("Stress elements are identical and the output is as expected")
        print("Actual output", output)
        print("Expected output", Exp_matrix)
        assert True
    else:
        print("Stress elements are not identical")
        assert False
#--(4)------------------------test_inside_circ---------------------------------

def test_inside_circ():
    '''
    The function checks if the point is inside or outside the circle(domain area)
    '''
    #defining a random crack tip acts as a circle center
    c_2 = [0.5,0.5]
    #std length of the element along x and y axes
    length_element_x = 1
    length_element_y = 1
    #point under query
    P = [2,2]
    #scaling factor for the circle
    scale = 1.5
    Expected_bool = False
    #calling the actual function
    X = J_integral.inside_circ(c_2, length_element_x, length_element_y, P, scale)
    #doing the comparison
    if X == Expected_bool:
        print("Function is working fine")
        print("The current point is outside the circle")
        assert True
    else:
        print("Function is ill-conditioned")
        assert False

#--(5)----------------------step function tests---------------------------------

def test_step_function():
    '''
    The function checks if the point is above tha crack segment or below the crack segment
    '''
    print("test for a straight crack")
    print("----------------------------")

    #defining a random crack segment, that lies within element
    cracks1 = [[2.25 ,2.25],
               [2.75 ,2.25]]

    Nodes  =  [(2.0, 2.0), (3.0, 2.0), (3.0, 3.0), (2.0, 3.0)]

    #for plotting
    x = [2.25, 2.75]
    y = [2.25, 2.25]

    cracks2 = [[2.5,2.75],
               [2.75,2.5]]

    #for plotting
    c_1 = [2.5,2.75]
    c_2 = [2.75,2.5]

    #plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
    for i in Nodes:
        ax1.scatter(i[0], i[1], marker="s", s=100, color="r")
        ax1.set_title('element with inclined crack')
        ax1.set_xlabel('X-axis')
        ax1.set_ylabel('Y-axis')
        ax2.scatter(i[0], i[1], marker="s", s=100, color="r")
        ax2.set_title('Element with staight crack')
        ax2.set_xlabel('X-axis')
        ax2.set_ylabel('Y-axis')
    for i in cracks2:
        ax1.scatter(i[0], i[1], color='r')
    ax1.plot(c_1, c_2, linewidth = 3)

    for i in cracks1:
        ax2.scatter(i[0], i[1], color='r')
    ax2.plot(x, y, linewidth = 3)
    fig.savefig("Step_function.png")
    plt.close()

    #to check the output is as expected
    Expected_determinant_vals = [-1, -1, 1, -1]
    for i,j in zip(Nodes,Expected_determinant_vals):
        A = enrichment_functions.step_function(i, cracks2)

        if A not in Expected_determinant_vals:
            assert False
        else:
            print("Expected output", Expected_determinant_vals)
            print("Actual output", A)
            assert True
    print("")
    print("***********************************************************")
    print("")
    #defining inclined crack---------------------------------------------------------------
    print("test for an inclined crack")
    print("----------------------------")
    #expected output
    determinant_vals = [-1,-1, 1, 1]
    for i,j in zip(Nodes,determinant_vals):
        A = enrichment_functions.step_function(i, cracks1)
        if A not in determinant_vals:
            assert False
        else:
            print("Expected output", determinant_vals)
            print("Actual output", A)
            assert True

#--(6)------------------------test_heaviside_functions---------------------------------

def test_heaviside_functions():
    # sample for heaviside_function1
    Enr_bmatrix = np.array([[1, 1, 1, 1],
                            [1, 1, 1, 1]])

    #defining hesviside function value
    H1 = 1
    Expected_matrix1 = np.array([[Enr_bmatrix[0, 0]*H1, 0],
                                 [0, Enr_bmatrix[1, 0]*H1],
                                 [Enr_bmatrix[1, 0]*H1, Enr_bmatrix[0, 0]*H1]])

    #generating B-matrix
    B_matrix1 = enrichment_functions.heaviside_function1(Enr_bmatrix, H1)

    #comparing the matrices, Element to element
    if (areSame(B_matrix1, Expected_matrix1)==1):
        print("Matrices are identical and the output is as expected therefore, the 'heaviside_function1' has passed")
        print("Expected output", Expected_matrix1)
        print("Actual output", B_matrix1)
        assert True
    else:
        print("Matrices are not identical and the output is not as expected therefore, the test has failed")
        assert False

#--(7)------------------------test_addAtPos---------------------------------

def test_addAtPos():
    #defining a matrix of size 4x4
    Matrix = np.arange(0,16,1).reshape(4,4)
    #defining a dummy matrix to which
    dummy = np.zeros([5,5])
    #to define the expected output beforehand
    Expected_output = np.array([[0,1,2,3,0],
                                [4,5,6,7,0],
                                [8,9,10,11,0],
                                [12,13,14,15,0],
                                [0,0,0,0,0]])
    #actual result from calling the function
    Output = KU_F.addAtPos(Matrix, dummy)
    #comparing the results
    if (areSame(Output, Expected_output)==1):
        print("Matrices are identical and the output is as expected therefore, the 'addAtPos' has passed")
        print("Expected output: \n", Expected_output)
        print("Actual output: \n", Output)
        assert True
    else:
        print("Matrices are not identical and the output is not as expected therefore, the test has failed")
        assert False

#--(8)------------------------test_connectivity_matrix---------------------------------

def test_connectivity_matrix():

    #defining a dummy k-matrix just for an input
    K_Matrix = np.arange(0,64,1).reshape(8,8)
    EL = [[0,1,2,3]]
    length_nodes = 4 #4 nodal coordinates
    Hside = []
    Tside = []
    K_global, Total_Normal_DOFs, Total_Dofs, A = KU_F.connectivity_matrix(EL, [K_Matrix], length_nodes, Hside, Tside)

    #random expedted output
    Expected_output = np.eye(8)
    connect = A[0]

    #comparing the matrices
    if (areSame(connect, Expected_output)==1):
        print("Matrices are identical, the'connectivity matrix function' has passed")
        print("Expected output: \n", Expected_output)
        print("Actual output: \n", connect)
        assert True
    else:
        print("Matrices are not identical, the test has failed")
        assert False

#----------------------------------------------------------------------
def validate(result, Expected_result):
    '''
    Function to compare the results
    '''
    #looping through the actual result list and comparing it with the expected_result list
    for i in result:
        if i not in Expected_result:
            return 0
    return 1

#--(9)------------------------test_node_filtering--------------------------------

def test_node_filtering():
    #This function checks if the enriched nodal list has been filtered or not

    #defing random set of nodal coordinates
    Nodes_list  =  [[(2.0, 2.0), (3.0, 2.0), (3.0, 3.0), (2.0, 3.0)],
                    [(0.0, 2.0), (1.0, 2.0), (1.0, 3.0), (0.0, 3.0)],
                    [(1.0, 2.0), (2.0, 2.0), (2.0, 3.0), (1.0, 3.0)],
                    [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
                    [(1.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0)],
                    [(2.0, 0.0), (3.0, 0.0), (3.0, 1.0), (2.0, 1.0)],
                    [(3.0, 0.0), (4.0, 0.0), (4.0, 1.0), (3.0, 1.0)]]

    #select a sub list from the nodes_list as an eneriched list
    Enriched_nodes = [[(3.0, 0.0), (4.0, 0.0), (4.0, 1.0), (3.0, 1.0)]]
    #call the actual function
    result = Assignment.filtering(Enriched_nodes, Nodes_list)
    #predefining the output
    Expected_result = [[(2.0, 2.0), (3.0, 2.0), (3.0, 3.0), (2.0, 3.0)],
                       [(0.0, 2.0), (1.0, 2.0), (1.0, 3.0), (0.0, 3.0)],
                       [(1.0, 2.0), (2.0, 2.0), (2.0, 3.0), (1.0, 3.0)],
                       [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
                       [(1.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0)],
                       [(2.0, 0.0), (3.0, 0.0), (3.0, 1.0), (2.0, 1.0)]]

    #call the comparing function
    if (validate(result, Expected_result)==1):
        print("the test has passed, lists are identical")
        print("Expected output: \n", Expected_result)
        print("Actual output: \n", result)
        assert True
    else:
        print("the test has failed")
        assert False

#--(10)------------------------test_asymptotic_functions--------------------------------
def test_asymptotic_functions():
    r  = 0.25
    theta = 0.15935
    alpha = 0

    A, B, C, D, Act_output = enrichment_functions.asymptotic_functions(r, theta, alpha)

    theta = theta*180/np.pi

    F1 = np.sqrt(r) * np.sin(theta/2)
    F2 = np.sqrt(r) * np.cos(theta/2)
    F3 = np.sqrt(r) * np.sin(theta/2) * np.sin(theta)
    F4 = np.sqrt(r) * np.cos(theta/2) * np.sin(theta)

    F1x1 = (-1/(2 * np.sqrt(r))) * np.sin(theta/2)
    F1y1 =  (1/(2 * np.sqrt(r))) * np.cos(theta/2)

    F2x1 = (1/(2 * np.sqrt(r))) * np.cos(theta/2)
    F2y1 = (1/(2 * np.sqrt(r))) * np.sin(theta/2)

    F3x1 = (-1/(2 * np.sqrt(r))) * np.sin(3 * theta/2) * np.sin(theta)
    F3y1 =  (1/(2 * np.sqrt(r))) * (np.sin(theta/2) + np.sin(3 * theta/2) * np.cos(theta))

    F4x1 = (-1/(2 * np.sqrt(r))) * np.cos(3 * theta/2) * np.sin(theta)
    F4y1 =  (1/(2 * np.sqrt(r))) * (np.cos(theta/2) + np.cos(3 * theta/2) * np.cos(theta))

    '''
    the derivatives of crack tip asymptotic functions with respect to the global coordinate system (x, y)
    ∂Fα/∂x = ∂Fα/∂x1 * cosα − ∂Fα/∂x2 * sinα
    ∂Fα/∂x = ∂Fα/∂x1 * sinα + ∂Fα/∂x2 * cosα
    α = 1,2,3,4
    '''

    dF1X = F1x1 * np.cos(alpha) - F1y1 * np.sin(alpha)
    dF1Y = F1x1 * np.sin(alpha) + F1y1 * np.cos(alpha)

    dF2X = F2x1 * np.cos(alpha) - F2y1 * np.sin(alpha)
    dF2Y = F2x1 * np.sin(alpha) + F2y1 * np.cos(alpha)

    dF3X = F3x1 * np.cos(alpha) - F3y1 * np.sin(alpha)
    dF3Y = F3x1 * np.sin(alpha) + F3y1 * np.cos(alpha)

    dF4X = F4x1 * np.cos(alpha) - F4y1 * np.sin(alpha)
    dF4Y = F4x1 * np.sin(alpha) + F4y1 * np.cos(alpha)

    # array used for defining the B-matrix for the tip enriched elements
    Exp_output = np.array([dF1X, dF1Y, dF2X, dF2Y, dF3X, dF3Y, dF4X, dF4Y])

    for i,j in zip(Exp_output,Act_output):
        if i != j:
            print("Asymptotic function' has failed the test")
            assert False

    else:
        print("Asymptotic function' has passed the test")
        print("Expected output: \n", Exp_output)
        print("Actual output: \n", Act_output)
        assert True

#--(11)------------------------test_Gausspoints--------------------------------
def test_Gausspoints():
    '''
    Function to to check the Gauss point coordinates generation
    '''
    #defining a random nodal coordinate
    Nodes  =  [[(2.0, 2.0), (3.0, 2.0), (3.0, 3.0), (2.0, 3.0)]]

    #expected output
    Expected_pts = [[2.25, 2.25], [2.75, 2.25], [2.75, 2.75], [2.25, 2.75]]
    #calling actual function
    points, lis_t = Assignment.G_points(Nodes)

    #comparing the results
    if len(points) == len(Expected_pts):
        print("initial test passed")
        for i,j in zip(points, Expected_pts):
            for k,l in zip(i,j):
                if k!=l:
                    print("test Passed")
                    assert False
        else:
            print("test passed")
            print("Expected output: \n", Expected_pts)
            print("Actual output: \n", points)
            assert True

#--(12)------------------------test_tip_enrichment_func_N1--------------------------------
def test_tip_enrichment_func_N1():
    '''
    The function checks if the matrix generated is as per the requirement
    '''
    #defining a known inputs
    F1, F2, F3, F4 = 1, 1, 1, 1
    dF = np.array([1,1,1,1,1,1,1,1])
    coordinates = [(2.0, 2.0), (3.0, 2.0), (3.0, 3.0), (2.0, 3.0)]
    #defining a single Gauss point
    xi_1, xi_2 = 1/np.sqrt(3), 1/np.sqrt(3)
    N1 = 0.25* (1-xi_1) * (1-xi_2)
    N2 = 0.25* (1+xi_1) * (1-xi_2)
    N3 = 0.25* (1+xi_1) * (1+xi_2)
    N4 = 0.25* (1-xi_1) * (1+xi_2)

    #usual mapping from local to global coordinate system
    dNdxi = (1/4) * np.array([[-(1-xi_2), 1-xi_2, 1+xi_2, -(1+xi_2)],
                              [-(1-xi_1), -(1+xi_1), 1+xi_1, 1-xi_1]])

    jacobi = np.matmul(dNdxi, coordinates)
    inverse_jacobi = linalg.inv(jacobi)
    dN = np.matmul(inverse_jacobi, dNdxi)

    N = [N1, N2, N3, N4]
    #calling the actual function
    B_tip1 = enrichment_functions.tip_enrichment_func_N1(F1, F2, F3, F4, dN, dF, N)

    #B-matrix for F1 function
    B11 = np.array([[(F1 * dN[0, 0]) + (dF[0] * N[0]), 0],
                    [0, (F1 * dN[1, 0]) + (dF[1] * N[0])],
                    [(F1 * dN[1, 0]) + (dF[1] * N[0]), (F1 * dN[0, 0]) + (dF[0] * N[0])]])
    #B-matrix for F2 function
    B21 = np.array([[(F2 * dN[0, 0]) + (dF[2] * N[0]), 0],
                    [0, (F2 * dN[1, 0]) + (dF[3] * N[0])],
                    [(F2 * dN[1, 0]) + (dF[3] * N[0]), (F2 * dN[0, 0]) + (dF[2] * N[0])]])
    #B-matrix for F3 function
    B31 = np.array([[(F3 * dN[0, 0]) + (dF[4] * N[0]), 0],
                    [0, (F3 * dN[1, 0]) + (dF[5] * N[0])],
                    [(F3 * dN[1, 0]) + (dF[5] * N[0]), (F3 * dN[0, 0]) + (dF[4] * N[0])]])
    #B-matrix for F4 function
    B41 = np.array([[(F4 * dN[0, 0]) + (dF[6] * N[0]), 0],
                    [0, (F4 * dN[1, 0]) + (dF[7] * N[0])],
                    [(F4 * dN[1, 0]) + (dF[7] * N[0]), (F4 * dN[0, 0]) + (dF[6] * N[0])]])

    # 1 node will have 8 additional DOF, hence 4 individual B-matrices have been concatenated
    Expected_tip1 = np.concatenate((B11, B21, B31, B41), axis=1)

    #calling the comparing function
    if (areSame(B_tip1, Expected_tip1)==1):
        print("Expected output: \n", Expected_tip1)
        print("Actual output: \n", B_tip1)
        print("Matrices are identical and the output is as expected therefore, the 'tip_enrichment_func_N1' has passed")
        assert True
    else:
        print("Matrices are not identical and the output is not as expected therefore, the test has failed")
        assert False
#--(13)------------------------test_asymptotic_functions--------------------------------
def test_E_filter():
    #defining the list of elements
    G_elements = [(14.0, 15.0, 21.0, 20.0), (12.0, 13.0, 19.0, 18.0),
                  (16.0, 17.0, 23.0, 22.0), (22.0, 23.0, 29.0, 28.0)]
    #predefining the enriched element list
    EN_list = [(22.0, 23.0, 29.0, 28.0), (16.0, 17.0, 23.0, 22.0)]

    #defining the expected output beforehand
    Expected_output = [(14.0, 15.0, 21.0, 20.0), (12.0, 13.0, 19.0, 18.0)]
    #calling the actual function
    Actual_output = Assignment.E_filter(EN_list, G_elements)
    #comparing the results
    if Expected_output == Actual_output:
        print("the expected output: ", Expected_output)
        print("the actual output: ", Actual_output)
        assert True
    else:
        assert False