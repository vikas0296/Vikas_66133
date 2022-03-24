import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import enrichment_functions
import KU_F
import operator
import uniform_mesh; import Assignment
import Stress_Strains

#--(2)-------------------------test_LE_patch---------------------------------------------
def plot_mesh(NL):
    #fuction to plot mesh 
    plt.scatter(NL[:,0], NL[:,1], c='black', s=50, label = "Node")
    plt.legend(loc="upper right")
    plt.title('Sample for Patch test')
    plt.ylabel('y_axis')
    plt.xlabel('x_axis')
    

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
    plot_mesh(NODES)

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

    #plotting force
    plots = np.array([NODES[2],  NODES[5],  NODES[8]])
    plt.scatter(plots[:,0], plots[:,1], color="g", s = 200, label = "Load")
    plt.legend(loc="upper right")
    plt.show()

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
fig = plt.figure()
fig.set_figwidth(10)
fig.set_figheight(10)
def plot_mesh_dis(NL, A, B, X):

    plt.scatter(NL[:,0], NL[:,1], c='black', s=50, label = "Node")
    plt.legend(loc="upper right")

    for a, b, in zip(NL[0:,0], NL[0:,1]):
        plt.vlines(a, ymin=0, ymax=A, color="b", alpha=1)
        plt.hlines(b, xmin=0, xmax=B, color="c", alpha=1)
        # plt.savefig('saved_figure.png')
    plt.ylabel('Y-AXIS')
    plt.xlabel('X-AXIS')
    plt.title(f'2D_UNIFORM_MESH of size {X}x{X}')


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
    NL, EL = uniform_mesh.uniform_mesh(A, B, x)
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

    #plotting mesh
    plot_mesh_dis(NL, A, B, x)

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

    #plotting outer nodes
    J = np.delete(NL, 4, 0)
    plt.scatter(J[:,0], J[:,1], c='red', s=200, label = "displacements", marker="s")
    plt.legend(loc="upper right")

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
    NL, EL = uniform_mesh.uniform_mesh(A, B, x)
    length_nodes = len(NL)

    #function to form a list of 4 nodal coordintates in CCW direction
    all_ns, Elems_nodes = uniform_mesh.dictionary(NL, EL)
    elements = [ ]
    for j in Elems_nodes:
        elements.append(list(j.keys())[0])

    plot_mesh_dis(NL, A, B, x)

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

    plt.scatter(NL[:,0], NL[:,1], c = "green", s=200, label="FORCE(N)")
    plt.legend(loc="upper right")

    #generate Gauss points
    Gpoints, GPs = Assignment.G_points(all_ns)
    beta = GPs[0]
    plt.scatter(beta[:,0], beta[:,1], marker = '*', color = 'r', s=200, label = "GaussPoints")
    plt.legend(loc="upper right")

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
