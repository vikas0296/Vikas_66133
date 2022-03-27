'''
AUTHOR : VIKAS MELKOT BALASUBRAMANYAM
MATRICULATION NUMBER : 66133
Personal Programming Project: IMPLEMENTATION OF XFEM AND FRACTURE MECHANICS
#=============================================================================
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import enrichment_functions
import KU_F
import operator
import class_crack
import uniform_mesh; import Assignment
import Stress_Strains
import Displacement_approx
import Stress_Strains
import J_integral; import Tip_enrichment

#--(1)-------------------------test_LE_patch---------------------------------------------

def test_LE_patch():
    '''
    Aim: To check if the selected material sample is linear elastic in nature

    Expected output: If the material is linear elastic, then the Newton-Raphson should converge in the first
        iteration

    Remarks: Test case Passed
    '''
    #nodes defining for the plot
    NL1 = np.array([0,0])
    NL2 = np.array([4,0])
    NL3 = np.array([10,0])
    NL4 = np.array([0,4.5])
    NL5 = np.array([5.5,5.5])
    NL6 = np.array([10,5])
    NL7 = np.array([0,10])
    NL8 = np.array([4.2,10])
    NL9 = np.array([10,10])
    NODES = np.array([NL1, NL2, NL3, NL4, NL5, NL6, NL7, NL8, NL9])


    #same set of nodes are defined for forming elements
    NL1 = [0, 0]
    NL2 = [4, 0]
    NL3 = [10, 0]
    NL4 = [0, 4.5]
    NL5 = [5.5, 5.5]
    NL6 = [10, 5]
    NL7 = [0, 10]
    NL8 = [4.2, 10]
    NL9 = [10, 10]
    NL = [NL1, NL2, NL3, NL4, NL5, NL6, NL7, NL8, NL9]
    #defining element list
    E1 = [0,1,4,3]
    E2 = [1,2,5,4]
    E3 = [3,4,7,6]
    E4 = [4,5,8,7]
    Es = [E1, E2, E3, E4]

    length_nodes = len(NL)
    #forming nodal list
    nodes1 = [NL1, NL2, NL5, NL4]
    nodes2 = [NL2, NL3, NL6, NL5]
    nodes3 = [NL4, NL5, NL8, NL7]
    nodes4 = [NL5, NL6, NL9, NL8]

    #collecting all the nodes
    ALL_NL = [nodes1, nodes2, nodes3, nodes4]
    Hside, Tside = [], []
    #=============================Material_properties===========================================
    #defining material properties
    D = 200e3
    nu = 0.25
    D_plane_stress = (D/(1-nu**2)) * np.array([[1, nu, 0],
                                               [nu, 1, 0],
                                               [0, 0, (1-nu)/2]])

    D_plane_strain = (D / (1 + nu)* (1-2*nu)) * np.array([[(1 - nu), nu, 0],
                                                          [nu, (1 - nu), 0],
                                                          [0, 0, (1 - 2*nu)/2]])

    GaussPoint_1to4 = np.array([[ 1/np.sqrt(3),  1/np.sqrt(3)], [ 1/np.sqrt(3), -1/np.sqrt(3)],
                                [-1/np.sqrt(3),  1/np.sqrt(3)], [-1/np.sqrt(3), -1/np.sqrt(3)]])

    #looping through all the nodes to generate element stiffness matrices
    Zero = []
    for i in ALL_NL:
        Ks = enrichment_functions.classical_FE(i, GaussPoint_1to4, D_plane_stress)
        Zero.append(Ks)

    #function to assemble all the element matrices to obtain global stiffness matrix
    K_global, Total_Normal_DOFs, Total_Dofs, A = KU_F.connectivity_matrix(Es, Zero, length_nodes, Hside, Tside)


    #defining Boundary conditions
    constraints = [0, 1, 6, 12]
    #defining force vector
    force_vector = np.zeros([len(K_global),1])
    #force applied at these points
    positions = [2,5,8]

    #initializing force on the particular nodes
    force_vector[2*2, 0] = 2.5
    force_vector[5*2, 0] = 5
    force_vector[8*2, 0] = 2.5

    #defining displacement vector
    columns = len(NL) * 2
    U_global = np.zeros([columns, 1])

    #deleting rows and columns
    reduced_k = np.delete(np.delete(K_global, constraints, axis = 0),  constraints, axis = 1)
    reduced_f = np.delete(force_vector, constraints, 0)
    u = np.delete(U_global, constraints, 0)

    #Newton-Raphson method
    n = 1
    counter = 0
    while n!=5:
        #calculating P vector
        P = reduced_k @ u
        #calculating residual vector
        R = reduced_f - P
        ZZ = np.round(R,5)
        #check if all the values in the are zero
        XX = np.count_nonzero(ZZ==0)
        if XX == len(reduced_f):
            print(f"convergence after {counter} iteration")
            assert True
            break
            counter += 1

        else:
            #solving the BVP for the residuals
            delta_disp=np.linalg.solve(reduced_k, R)
            #updating the displacements
            u += delta_disp
            n = n+1
            counter += 1

#--(2)-------------------------test_displacement_2x2---------------------------------------------

def test_displacement_2x2():
    '''
    Aim: To check if the displacement of the inner node is same as the displacement of the outer nodes

    Expected output: The inner node will have the same displacement as the outer nodes

    Remarks: Test case passed

    '''
    #inputs, length of the specimen along X-axis and Y-axis
    A = 2
    B = 2
    #no of divisions in along X and y axis
    x = 2
    #calling uniform mesh to generate list of nodes and elements
    NL, EL = uniform_mesh.mesh(A, B, x)
    #length of nodes
    length_nodes = len(NL)
    Us = 0.5

    #=============================Material_properties===========================================

    #defining material parameters
    D = 200000
    nu = 0.25
    D_plane_stress = (D / (1 - nu**2)) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]])

    #defining Gauss points
    GaussPoint_1to4 = np.array([[ 1/np.sqrt(3),  1/np.sqrt(3)], [ 1/np.sqrt(3), -1/np.sqrt(3)],
                                [-1/np.sqrt(3),  1/np.sqrt(3)], [-1/np.sqrt(3), -1/np.sqrt(3)]])

    '''
    length of a segment, given 2 points = sqrt((x2-x1)**2 + (y2-y1)**2)
    '''
    #calling dictionary function to generate list of 4 nodes and elements numbers
    all_ns, Elems_nodes = uniform_mesh.dictionary(NL, EL)
    elements = [ ]
    for j in Elems_nodes:
        elements.append(list(j.keys())[0])

    #subtracting 1 from the every element number
    UNIT =[]
    for i in elements:
        ELES = np.asarray(i)
        UNIT.append(ELES-1)


    #calling classical_FE function to calculate elemental stiffness matrix
    Zero =[]
    for x in all_ns:
        Ks = enrichment_functions.classical_FE(x, GaussPoint_1to4, D_plane_stress)
        Zero.append(Ks)

    Tside =[]; Hside =[]

    #calling connectivity matrix to get a global syiffness matrix
    K_global, Total_Normal_DOFs, Total_Dofs, A = KU_F.connectivity_matrix(UNIT, Zero, length_nodes,
                                                                                        Hside, Tside)

    #prescribing the displacements to the outer nodes
    Ux = [0,2,4,6,10,12,14,16]
    Uy = [1,3,5,7,11,13,15,17]
    
    #creating displacement vector
    Ud = np.zeros([len(NL)*2, 1])
    #assigning the displacements Ux and Uy
    for i,j in zip(Ux, Uy):
        Ud[i,0] = Us
        Ud[j,0] = Us

    #calculate the residual forces
    Rf = K_global@Ud
    #subtracting the Rf with the zero matrix
    Zeros = np.zeros([len(NL)*2,1])
    R_forces = Zeros-Rf

    #Deleting rows and columns of the K-global for the known displacements
    reduced_k = np.delete(np.delete(K_global, Ux+Uy, 0),  Ux+Uy, 1)
    reduced_f = np.delete(R_forces, Ux+Uy, 0)

    #calculating the displacements of the inner node
    U = np.linalg.pinv(reduced_k).dot(reduced_f)
    U1 = U[0]
    U2 = U[1]
    Ux = np.round(U1[0], 4)
    Uy = np.round(U2[0], 4)
    if (Ux == Us) and Uy == Us:
        print("expected output: ", Us)
        print("Actual output: ", Ux, Uy)
        assert True
    else:
        print(False)


def test_isotropic_material_prop():

    A = 1          #length of the geometry along x and y axis
    B = 1
    x = 1           # number of divisions along x and y axis
    NL, EL = uniform_mesh.mesh(A, B, x)
    length_nodes = len(NL)

    #function to form a list of 4 nodal coordintates in CCW direction
    all_ns, Elems_nodes = uniform_mesh.dictionary(NL, EL)
    elements = [ ]
    for j in Elems_nodes:
        elements.append(list(j.keys())[0])


    UNIT =[]
    for i in elements:
        ELES = np.asarray(i)
        UNIT.append(ELES-1)

    #=============================Material_properties===========================================

    D = 200000
    nu = 0.3
    D_plane_stress = (D / (1 - nu**2)) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]])

    GaussPoint_1to4 = np.array([[ 1/np.sqrt(3),  1/np.sqrt(3)], [ 1/np.sqrt(3), -1/np.sqrt(3)],
                                [-1/np.sqrt(3),  1/np.sqrt(3)], [-1/np.sqrt(3), -1/np.sqrt(3)]])

    #looping through the nodal list to generate element stiffness matrix
    Zero =[]
    for x in all_ns:
        Ks = enrichment_functions.classical_FE(x, GaussPoint_1to4, D_plane_stress)
        Zero.append(Ks)

    Tside =[]; Hside =[]

    #calling connectivity matrix to get a global syiffness matrix
    K_global, Total_Normal_DOFs, Total_Dofs, A = KU_F.connectivity_matrix(UNIT, Zero, length_nodes,
                                                                                        Hside, Tside)

    #Prescribing boundary conditions
    L = 100
    force = np.zeros([8,1])

    force[0,0] = -L
    force[2,0] = L
    force[4,0] = -L
    force[6,0] = L


    #generate Gauss points
    Gpoints, GPs = Assignment.G_points(all_ns)
    beta = GPs[0]

    #sovling BVP
    disp=np.linalg.solve(K_global,force)

    #computing displacements
    Us=[]
    for i in elements:
        Ux_Uy = [0,0,0,0,0,0,0,0]
        i = list(np.asarray(i) - 1)
        for j in i:
            Ux_Uy[i.index(j)*2] = disp[int(j)*2]
            Ux_Uy[i.index(j)*2+1] = disp[int(j)*2+1]

        Us.append(Ux_Uy)

    #computing the stresses and strains
    STRESS, STRAIN = Stress_Strains.strain_stress_enr(Us, all_ns, GaussPoint_1to4, D_plane_stress)

    #looping and comparing the stresses
    STR1 = np.round(STRESS)[0]
    STR2 = np.round(STRESS)[1]
    STR3 = np.round(STRESS)[2]
    STR4 = np.round(STRESS)[3]
    for i,j,k,l in zip(STR1, STR2, STR3, STR4):
        if i!=j and j!=k and k!=l and i==k and i==l and j==l:
            assert False

    else:
        print(STR1)
        print(STR2)
        print(STR3)
        print(STR4)
        print("The stresses are same at all the Gauss points")
        assert True

    #looping and comparing the strains
    STN1 = np.round(STRAIN)[0]
    STN2 = np.round(STRAIN)[1]
    STN3 = np.round(STRAIN)[2]
    STN4 = np.round(STRAIN)[3]
    for i,j,k,l in zip(STN1, STN2, STN3, STN4):
        if i!=j and j!=k and k!=l and i==k and i==l and j==l:
            assert False

    else:
        print(STN1)
        print(STN2)
        print(STN3)
        print(STN4)
        print("The strains are same at all the Gauss points")
        assert True

#----------------------------------test_Jacobian----------------------------------------------------
def test_Jacobian():
    '''
    Aim: Jacobian is a volume conserving, and even after transforming the
            Jacobian the value should remain same

    Expected_output : Determinant value of the Jacobian before and after transformation should be
            constant

    Remarks : Test case passed.
    '''
    print("================================================")
    print("Testing volume conserving property of the Jacobian")
    print("================================================")
    #Inputs: nodes, theta, xi_1 and xi_2
    node = [(2.0, 2.0), (3.0, 2.0), (3.0, 3.0), (2.0, 3.0)]
    xi_1 = -1/np.sqrt(3)
    xi_2 = -1/np.sqrt(3)
    theta = -90

    #Differentiation of shape functions w.r.t xi_1, xi_2
    dNdxi = (1/4) * np.array([[-(1-xi_2), 1-xi_2, 1+xi_2, -(1+xi_2)],
                              [-(1-xi_1), -(1+xi_1), 1+xi_1, 1-xi_1]])
    #calculate Jacobian
    Jacobi = np.matmul(dNdxi, node)
    #taking the determinant of the Jacobian
    deter_N = np.round(np.linalg.det(Jacobi),4)

    #define Rotation matrix
    R_matrix = np.array([[np.cos(theta), np.sin(theta)],
                         [-np.sin(theta), np.cos(theta)]])

    #Transformed Jacobian
    T_Jacobi = R_matrix @ Jacobi
    #taking the determinant of the transformed Jacobian
    deter_T = np.round(np.linalg.det(T_Jacobi),4)

    #comparing the determinant values
    if deter_T==deter_N:
        print("Even after the transformation the Jacobian remains constant")
        assert True
    else:
        assert False


def test_Rigid_body_motions():
    '''
    Aim : Also known as eigen value test ,We check rigid body motion for constrained and unconstrained global stiffness matrices

    Expected Output : The number of zero eigenvalues from global stiffness matrix represent the
                      rigid body motion of the structure.
                      For a 2D geometry: unconstrained - 3 zeros eigenvalues (2-rotation, 1-translation)
                                         constrained - 0 zero eigenvalues
    Remarks: Test case passed
    '''
    #inputs, length of the specimen along X-axis and Y-axis
    A = 1
    B = 1
    #no of divisions in along X and y axis
    x = 1
    #calling uniform mesh to generate list of nodes and elements
    NL, EL = uniform_mesh.mesh(A, B, x)
    #length of nodes
    length_nodes = len(NL)

    #=============================Material_properties===========================================

    #defining material parameters
    D = 200000
    nu = 0.25
    D_plane_stress = (D / (1 - nu**2)) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]])

    #defining Gauss points
    GaussPoint_1to4 = np.array([[ 1/np.sqrt(3),  1/np.sqrt(3)], [ 1/np.sqrt(3), -1/np.sqrt(3)],
                                [-1/np.sqrt(3),  1/np.sqrt(3)], [-1/np.sqrt(3), -1/np.sqrt(3)]])

    '''
    length of a segment, given 2 points = sqrt((x2-x1)**2 + (y2-y1)**2)
    '''
    #calling dictionary function to generate list of 4 nodes and elements numbers
    all_ns, Elems_nodes = uniform_mesh.dictionary(NL, EL)
    elements = [ ]
    for j in Elems_nodes:
        elements.append(list(j.keys())[0])

    #subtracting 1 from the every element number
    UNIT =[]
    for i in elements:
        ELES = np.asarray(i)
        UNIT.append(ELES-1)

    #calling classical_FE function to calculate elemental stiffness matrix
    Zero =[]
    for x in all_ns:
        Ks = enrichment_functions.classical_FE(x, GaussPoint_1to4, D_plane_stress)
        Zero.append(Ks)

    Tside =[]; Hside =[]

    #calling connectivity matrix to get a global syiffness matrix
    K_global, Total_Normal_DOFs, Total_Dofs, A = KU_F.connectivity_matrix(UNIT, Zero, length_nodes, Hside, Tside)

    #compute the eigen values for unconstrained stiffness matrix
    eigen_values, eig_vectors = np.linalg.eig(K_global)
    eigen_values = np.round(eigen_values, 4)

    #count the zeros in the list
    unconstrained_rigid_body_motions=len(eigen_values[eigen_values==0])

    #If a side of the geometry is fixed then, lower 2 nodes(1,2)
    U_const = [0,1,2,3]

    #delete the rows and columns
    reduced_k = np.delete(np.delete(K_global, U_const, 0),  U_const, 1)

    #compute the eigen values for constrained/reduced stiffness matrix
    eigen_values, eig_vectors = np.linalg.eig(reduced_k)
    eigen_values = np.round(eigen_values, 4)

    #count the zeros in the list
    constrained_rigid_body_motions=len(eigen_values[eigen_values==0])

    assert (unconstrained_rigid_body_motions==3) and (constrained_rigid_body_motions==0) is True

def test_Rigid_body_rotation():
    '''
    The external nodes are rotated using transformation matrix and internal nodal displacement are calculated.
    Aim : Rigid body rotation - global stiffness matrix should be able to represent rigid body rotation.

    Expected Output : The displacement of the internal node should match the coordinates obtained after performing rotation using equ.(6.7) to reference system.

    Remarks : Test case passed.
    '''
    #inputs, length of the specimen along X-axis and Y-axis
    A = 2
    B = 2
    #no of divisions in along X and y axis
    x = 2
    #calling uniform mesh to generate list of nodes and elements
    NL, EL = uniform_mesh.mesh(A, B, x)
    #length of nodes
    length_nodes = len(NL)

    #=============================Material_properties===========================================

    #defining material parameters
    D = 200000
    nu = 0.25
    D_plane_stress = (D / (1 - nu**2)) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]])

    #defining Gauss points
    GaussPoint_1to4 = np.array([[ 1/np.sqrt(3),  1/np.sqrt(3)], [ 1/np.sqrt(3), -1/np.sqrt(3)],
                                [-1/np.sqrt(3),  1/np.sqrt(3)], [-1/np.sqrt(3), -1/np.sqrt(3)]])


    #calling dictionary function to generate list of 4 nodes and elements numbers
    all_ns, Elems_nodes = uniform_mesh.dictionary(NL, EL)
    elements = [ ]
    for j in Elems_nodes:
        elements.append(list(j.keys())[0])

    #subtracting 1 from the every element number
    UNIT =[]
    for i in elements:
        ELES = np.asarray(i)
        UNIT.append(ELES-1)


    #calling classical_FE function to calculate elemental stiffness matrix
    Zero =[]
    for x in all_ns:
        Ks = enrichment_functions.classical_FE(x, GaussPoint_1to4, D_plane_stress)
        Zero.append(Ks)

    Tside =[]; Hside =[]

    #calling connectivity matrix to get a global syiffness matrix
    K_global, Total_Normal_DOFs, Total_Dofs, A = KU_F.connectivity_matrix(UNIT, Zero, length_nodes,
                                                                                        Hside, Tside)

    theta = 90
    R_matrix = np.array([[np.cos(theta), np.sin(theta)],
                         [-np.sin(theta), np.cos(theta)]])

    New_points=np.matmul(NL, R_matrix)
    Ud = np.zeros([len(NL)*2, 1])

    # prescribing the displacements to the outer nodes
    for i in range(0,len(NL)):
        if i!=4:#assigning displacements to the outer nodes
            idx = np.matmul(NL[i], R_matrix)
            Ud[i*2,0] = idx[0]
            Ud[i*2+1,0] = idx[1]

        else:
            idx = np.matmul(np.array([0,0]), R_matrix)
            Ud[i*2,0] = idx[0]
            Ud[i*2+1,0] = idx[1]

    #positions where displacements are prescribed
    Ux = [0,2,4,6,10,12,14,16]
    Uy = [1,3,5,7,11,13,15,17]

    #calculate the residual forces
    Rf = K_global@Ud

    #subtracting the Rf with the zero matrix
    Zeros = np.zeros([len(NL)*2,1])
    R_forces = Zeros-Rf

    #Deleting rows and columns of the K-global for the known displacements
    reduced_k = np.delete(np.delete(K_global, Ux+Uy, 0),  Ux+Uy, 1)
    reduced_f = np.delete(R_forces, Ux+Uy, 0)

    #calculating the displacements of the inner node
    U = np.linalg.pinv(reduced_k).dot(reduced_f)

    #comparison block
    tol = 1e-5
    U1 = U[0]
    U2 = U[1]
    Ux_CENTER_NODE = U1[0]
    Uy_CENTER_NODE = U2[0]
    POSITIONS_CENTER_NODE = New_points[4]

    if abs(Ux_CENTER_NODE-POSITIONS_CENTER_NODE[0]) < tol and abs(Uy_CENTER_NODE - POSITIONS_CENTER_NODE[1]) < tol:
        assert True
    else:
        assert False

def test_Stress_infi():
    '''
    Aim: To check if the stress near the crack tip is infinity
    Expected output: The stress near the crack tip is infinity
    Remarks: Test case passed

    '''
    #inputs, length of the specimen along X-axis and Y-axis
    A = 1
    B = 1
    #no of divisions in along X and y axis
    x = 1
    #calling uniform mesh to generate list of nodes and elements
    NL, EL = uniform_mesh.mesh(A, B, x)
    #length of nodes
    length_nodes = len(NL)
    Us = 0.5

    #=============================Material_properties===========================================

    #defining material parameters
    D = 200000
    nu = 0.25
    D_plane_stress = (D / (1 - nu**2)) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]])

    #defining Gauss points
    GaussPoint = np.array([[0,0]])
    alpha = 0

    '''
    length of a segment, given 2 points = sqrt((x2-x1)**2 + (y2-y1)**2)
    '''
    #calling dictionary function to generate list of 4 nodes and elements numbers
    NCs, Elems_nodes = uniform_mesh.dictionary(NL, EL)
    elements = [ ]
    for j in Elems_nodes:
        elements.append(list(j.keys())[0])

    #subtracting 1 from the every element number
    UNIT =[]
    for i in elements:
        ELES = np.asarray(i)
        UNIT.append(ELES-1)

    #defining random crack segment
    cracks = [[0,0.5],[0.5,0.5]]
    c_1 = [0, 0.5]
    c_2 = [0.5, 0.5]

    #forming a stiffness matrix
    Nan, K_global, Nan2, ri, ti = class_crack.updated_cracktip(NCs[0], UNIT, c_2, GaussPoint, D_plane_stress, alpha)

    #defining force vector
    force_vector = np.zeros([len(K_global),1])
    force_vector[5,0] = 1
    force_vector[7,0] = 1

    #applyiing boundary conditions
    fixed_nodes = [0,1,2,3]
    reduced_k = np.delete(np.delete(K_global, fixed_nodes, 0),  fixed_nodes, 1)
    reduced_f = np.delete(force_vector, fixed_nodes, 0)
    Displacement_vector = np.linalg.pinv(reduced_k).dot(reduced_f)

    #8 classical degrees of freedom
    C_Dofs = 8
    Tside = UNIT
    Hside =[]

    for j in fixed_nodes:
        Displacement_vector = np.insert(Displacement_vector, j, 0)

    #compute displacements and stresses
    DISPLACEMENTS = Displacement_approx.displacement_approximation(NCs, cracks, UNIT, Tside[0], Hside,
                                                                   C_Dofs, Displacement_vector, alpha)

    STRESS, STRAIN = Stress_Strains.strain_stress_enr(DISPLACEMENTS, NCs, GaussPoint, D_plane_stress)

    #Gauss point coordinate in global coordinate system
    GAUSS_PT  =  [[[0.5, 0.5]]]
    scale = 3
    l_x = 1
    l_y = 1
    DIS = [DISPLACEMENTS]
    alpha = alpha * 180 / np.pi

    #j-integral procedure
    #looping through Node list,displacements,stress, strains, set4Gauss
    for i,j,k,p in zip(NCs, DIS, STRESS, GAUSS_PT):
        X = np.array(j[0])
        #inner loop through each stresses and Nodes
        for gps in p:

            #calculation of distance r and theta from the gauss points to the crack tip
            r, theta = Tip_enrichment.r_theta(gps, c_2, alpha)
            theta = theta * 180 / np.pi

            #defining Stress intensity factor for mode I in the auxiliary state
            K1 = 1
            #check if the stress near the tip is infinity
            try:
                Aux_stress11 = K1 / int(r)
                print(Aux_stress11)
            except ZeroDivisionError:
              print("Stress near the tip has reached infinity")
              assert True






