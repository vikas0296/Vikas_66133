'''
AUTHOR : VIKAS MELKOT BALASUBRAMANYAM
MATRICULATION NUMBER : 66133
Personal Programming Project: IMPLEMENTATION OF XFEM AND FRACTURE MECHANICS
#=============================================================================
'''

import numpy as np
from scipy import linalg
import sys
import uniform_mesh
import math as m
import Tip_enrichment
import plots

def Kinematics(grad_disps):
    '''
    Applying small strain theory, the function computes auxiliary strains from the auxiliary displacements
    εij = 1/2 (ui,j + uj,i)
    Parameters
    ----------
    grad_disps : Auxiliary displacements

    Returns
    -------
    Auxiliary strains

    '''
    # εij = 0.5*[[(ui,i+ui,i), (ui,j + uj,i)]
    #            [(ui,j + uj,i),(uj,j+uij,j)]]

    Epsilon_ij = 0.5*np.array([[(grad_disps[0,0] + grad_disps[0,0]), (grad_disps[0,1] + grad_disps[1,0])],
                                [(grad_disps[1,0] + grad_disps[0,1]), (grad_disps[1,1] + grad_disps[1,1])]])

    return Epsilon_ij[0,0], Epsilon_ij[0,1], Epsilon_ij[1,0], Epsilon_ij[1,1]


def inside_circ(c_2, length_element_x, length_element_y, P, scale):
    '''
    The function checks if any of the nodes is outside the domain under consideration.
    if yes, it will return True, False otherwise

    Parameters
    ----------
    c_2 : right crack tip
    length_element_x : length of the element along x-direction in "meters"
    length_element_y : length of the element along y-direction in "meters"
    P : point under query (nodes)
    scale : scaling factor of the domain used for interaction integral
    Returns
    -------
    bool
        Returns nodal value (q) 1 or 0 based on the condition
    '''
    #defining the length of the radius usually scaling factor will vary from 3-4
    LR = length_element_x*length_element_y
    #defining the radius of the circular domain
    radius = scale*np.sqrt(LR)
    #using equation of the circle to calculate the distance of the nodes from the crack tip
    distance = np.sqrt((P[0]-c_2[0])**2 + (P[1]-c_2[1])**2)
    if distance < radius:
        return True
    elif distance > radius:
        return False

def Global_to_local_CT(S, CTCS):
    '''
    The function transforms the stresses from global co-ordinate to local
    crack tip coordinate system.

    CTCS = np.array([[m.cos(alpha), m.sin(alpha)],
                     [-m.sin(alpha), m.cos(alpha)]])
    Parameters
    ----------
    S : stresses in global co-ordinate system
    CTCS : crack tip coordinate system
    Returns
    -------
    Stresses in local crack coordinate system
    '''
    # defining a square matrix using computed stresses [[σxx, σxy],
                                                      # [σyx, σyy]]
    global_stress = np.array([[S[0], S[2]],
                              [S[2], S[1]]])

    #transforming global stresses to local crack tip stresses using transformation matrix

    sigma = CTCS @ global_stress @ CTCS.T

    return sigma[0,0], sigma[0,1], sigma[1,0], sigma[1,1]


def Interaction_integral(Nodes, displacements, stress, set4Gauss, c_2, GaussPoint_1to4,
                             l_x, l_y, D_plane_stress, alpha, CL, A, DISP, D, scale, Y, Mode):

    '''
    The function computes the SIFs using domain integral technique
    Parameters:
    ---------------------
    Nodes : List of 4 nodal coordinates
    displacements : List of Element displacements (8 per element)
    stress : List of 3 stress values
    set4Gauss: Gauss coordinates in global system
    c_2: Crack tip coordinates
    GaussPoint_1to4 : 4 Gauss coordinates
    l_x, l_y: Length of Element along x and y axes
    D_plane_stress: Plane stress relation
    alpha: Angle made by crack tip w.r.t x-axis
    CL, A: Crack length, Geometry length along X
    D: Young's modulus
    scale: Scaling factor for the domain
    DISP: Applied displacement
    Y: vetrical distance from crack tip
    Mode: Type of crack analysis

    Returns
    -------------
    KI, KII
    '''

    I1_2 = 0
    I1_22 = 0
    AuxDisp2 = np.zeros([2,2])
    Auxiliary_disps = np.zeros([2,2])
    alpha = alpha * 180 / np.pi
    CTCS = np.array([[m.cos(alpha), m.sin(alpha)],
                     [-m.sin(alpha), m.cos(alpha)]])

    J =[]
    IS =[]
    #looping through Node list,displacements,stress, strains, set4Gauss
    for i,j,k,p in zip(Nodes, displacements, stress, set4Gauss):
        X = np.array(j[0])
        #calling inside_circ function to check if the node is outside or inside the elements
        #if the node in inside the nodal value of that node is set to 1, 0 otherwise
        q1 = inside_circ(c_2, l_x, l_y, i[0], scale)
        q2 = inside_circ(c_2, l_x, l_y, i[1], scale)
        q3 = inside_circ(c_2, l_x, l_y, i[2], scale)
        q4 = inside_circ(c_2, l_x, l_y, i[3], scale)
        #inner loop through each stresses and Nodes
        for points, Ns, gps, Srs in zip(GaussPoint_1to4, i, p, k):
            '''
            Defining GaussPoints
            '''
            xi_1 = points[0]
            xi_2 = points[1]

            '''
            Differentiation of shape functions wrt xi_1, xi_2
            '''
            NxNy = (1/4) * np.array([[-(1-xi_2), 1-xi_2, 1+xi_2, -(1+xi_2)],
                                     [-(1-xi_1), -(1+xi_1), 1+xi_1, 1-xi_1]])

            '''
            convertion from local coordinate system to global coordinate system
            1. Calculate jacobian
            2. Take inverse of the Jacobian
            3. multiply with differentiated shape functions
            '''

            jacobi = np.matmul(NxNy, i)

            inverse_jacobi = linalg.inv(jacobi)
            dN = np.matmul(inverse_jacobi, NxNy)

            # standard B-matrix
            B_std = np.array([[dN[0, 0], 0, dN[0, 1], 0, dN[0, 2], 0, dN[0, 3], 0],
                              [0, dN[1, 0], 0, dN[1, 1], 0, dN[1, 2], 0, dN[1, 3]],
                              [dN[1, 0], dN[0, 0], dN[1, 1], dN[0, 1], dN[1, 2], dN[0, 2], dN[1, 3], dN[0, 3]]])

            #calculating the displacement gradients
            Ux = np.matmul(B_std[0,0:8:2], X[0:8:2])                                    # Derivative of U with respect to X
            Uy = np.matmul(B_std[1,1:8:2], X[0:8:2])                                    # Derivative of U with respect to Y
            Vx = np.matmul(B_std[0,0:8:2], X[1:8:2])                                    # Derivative of V with respect to X
            Vy = np.matmul(B_std[1,1:8:2], X[1:8:2])                                    # Derivative of V with respect to Y


            # computed displacements using XFEM in "meters", they have been transformed to local
            # crack tip coordinates

            grad_disps = np.matrix(np.array([[Ux/1e3, Uy/1e3],
                                             [Vx/1e3, Vy/1e3]]))

            CTCS_displacements = np.matmul((np.matmul(CTCS, grad_disps)), CTCS.T)

            #calculation of distance r and theta from the gauss points to the crack tip
            r, theta = Tip_enrichment.r_theta(gps, c_2, alpha)
            theta = theta * 180 / np.pi
            r = r/1000

            #converting global stresses to local stress w.r.t crackt tip coordinates
            Stress11, Stress12, Stress21, Stress22 = Global_to_local_CT(Srs, CTCS)

            #assigning nodal values (q-vector) and converting it to global coordinate system using jacobian
            nodal_vals = np.array([[dN[0, 0]*q1, dN[0, 1]*q2, dN[0, 2]*q3, dN[0, 3]*q4],
                                   [dN[1, 0]*q1, dN[1, 1]*q2, dN[1, 2]*q3, dN[1, 3]*q4]])

            # doing a matrix multiplication of q vector and shape function, we get a matrix of shape of 2x1
            q_vector = np.array([[np.sum(nodal_vals[0])],
                                 [np.sum(nodal_vals[1])]])

            # converting the global q-vector to local crack tip system
            CTCS_Q = np.matmul(CTCS, q_vector)
            CTCS_Q1 = CTCS_Q[0]
            CTCS_Q2 = CTCS_Q[1]

            # Defining Auxilary terms for mode I------------------------------------------------------
            # the terms that have a maximum usage have been written here
            CT   = m.cos(theta)
            ST   = m.sin(theta)
            CT2  = m.cos(theta/2)
            ST2  = m.sin(theta/2)
            C3T2 = m.cos(3*theta/2)
            S3T2 = m.sin(3*theta/2)

            drdx =  CT
            drdy =  ST
            dtdx = -ST/r
            dtdy =  CT/r

            #defining Stress intensity factor for mode I in the auxiliary state
            K1 = 1
            nu = 0.25

            # "k" is the equation for plane stress condition
            k = (3-nu)/(1+nu)

            #defining auxiliart displacements and its gradients w.r.t to r and theta
            du1dr = (K1*(1+nu)/(2*D)) * (1/np.sqrt(2*np.pi*r)) * (CT2*(k-CT))
            du1dt = (K1*(1+nu)/D) * np.sqrt((r)/(2*np.pi)) * (-0.5*ST2*(k-CT) + ST*CT2)
            du2dr = (K1*(1+nu)/(2*D)) * (1/np.sqrt(2*np.pi*r)) * (ST2*(k-CT))
            du2dt = (K1*(1+nu)/D) * np.sqrt(r/(2*np.pi)) * (0.5*CT2*(k-CT) + ST*ST2)

            #Auxiliary displacements in cartesian coordinates for mode I
            Auxiliary_disps[0,0] = (du1dr * drdx) + (du1dt * dtdx) #Auxiliary displacement gradient of u1 with respect to x
            Auxiliary_disps[0,1] = (du1dr * drdy) + (du1dt * dtdy) #Auxiliary displacement gradient of u1 with respect to y
            Auxiliary_disps[1,0] = (du2dr * drdx) + (du2dt * dtdx) #Auxiliary displacement gradient of u2 with respect to x
            Auxiliary_disps[1,1] = (du2dr * drdy) + (du2dt * dtdy) #Auxiliary displacement gradient of u2 with respect to y

            #defining Auxiliary strains using Auxiliary_displacements for mode I
            Aux_Strain11, Aux_Strain12, Aux_Strain21, Aux_Strain22 = Kinematics(Auxiliary_disps)

            #defining Auxiliary stresses for mode I type crack propagation
            Aux_stress11 = (K1 / np.sqrt(2*np.pi*r)) * CT2 * (1 - ST2 * S3T2)
            Aux_stress22 = (K1 / np.sqrt(2*np.pi*r)) * CT2 * (1 + ST2 * S3T2)
            Aux_stress12 = (K1 / np.sqrt(2*np.pi*r)) * ST2 * CT2 * C3T2
            Aux_stress21 = Aux_stress12

            #Actual interaction integral terms
            '''
            integration [(-Wij + ui,j1*sigma + sigma1*ui,j)]
            implementing gauss type integration
            sum((-Wij + ui,j1*sigma + sigma1*ui,j) * qi,x * |jacobian|)

            '''
            Strain_energy = ((Stress11 * Aux_Strain11) + (Stress12 * Aux_Strain12) +
                              (Stress21 * Aux_Strain21) + (Stress22 * Aux_Strain22)) * CTCS_Q1

            I1 = ((Stress11 * Auxiliary_disps[0,0] * CTCS_Q1)  + (Stress12 * Auxiliary_disps[0,0] * CTCS_Q2) +
                  (Stress21 * Auxiliary_disps[1,0] * CTCS_Q1)  + (Stress22 * Auxiliary_disps[1,0] * CTCS_Q2))


            I2 = ((Aux_stress11 * CTCS_displacements[0,0] * CTCS_Q1) + (Aux_stress12 * CTCS_displacements[0,0] * CTCS_Q2)+
                  (Aux_stress21 * CTCS_displacements[1,0] * CTCS_Q1) + (Aux_stress22 * CTCS_displacements[1,0] * CTCS_Q2))

            I1_2 = (I1 + I2 - Strain_energy) * linalg.det(jacobi)


            # Defining Auxilary terms for mode II------------------------------------------------------
            K2 = 1

            #defining Auxiliary stresses for mode II type crack propagation
            Aux_stress112 = -(K2/np.sqrt(2*np.pi*r)) * ST2*(2+CT2*C3T2)
            Aux_stress222 =  (K2/np.sqrt(2*np.pi*r)) * ST2*CT2*C3T2
            Aux_stress122 =  (K2/np.sqrt(2*np.pi*r)) * CT2*(1-ST2*S3T2)
            Aux_stress212 =  Aux_stress122

            #defining auxiliart displacements and its gradients w.r.t to r and theta for mode II
            du1dr2 =  (K2*(1+nu)/(2*D)) * (1/np.sqrt(2*np.pi*r)) * ST2*(k+2+CT)             #Derivative of auxiliary x-displacement with respect to r
            du1dt2 =  (K2*(1+nu)/(D)) * np.sqrt((r)/(2*np.pi)) * (0.5*CT2*(k+2+CT)-ST2*ST)  #Derivative of auxiliary x-displacement with respect to theta
            du2dr2 =  (K2*(1+nu)/(2*D)) * (1/np.sqrt(2*np.pi*r)) * CT2*(k-2+CT)             #Derivative of auxiliary x-displacement with respect to r
            du2dt2 = -(K2*(1+nu)/(D)) * np.sqrt((r)/(2*np.pi)) * (-0.5*ST2*(k-2+CT)-CT2*ST) #Derivative of auxiliary x-displacement with respect to theta

            #Auxiliary displacements in cartesian coordinates for mode II
            AuxDisp2[0,0] = du1dr2 * drdx + du1dt2 * dtdx    #Auxiliary displacement gradient of u1 with respect to x
            AuxDisp2[0,1] = du1dr2 * drdy + du1dt2 * dtdy    #Auxiliary displacement gradient of u1 with respect to y
            AuxDisp2[1,0] = du2dr2 * drdx + du2dt2 * dtdx    #Auxiliary displacement gradient of u2 with respect to x
            AuxDisp2[1,1] = du2dr2 * drdy + du2dt2 * dtdy    #Auxiliary displacement gradient of u2 with respect to y


            #defining Auxiliary strains using Auxiliary_displacements for mode II
            Aux_Strain112, Aux_Strain122, Aux_Strain212, Aux_Strain222 = Kinematics(AuxDisp2)

            Strain_energy2 =  ((Stress11 * Aux_Strain112) + (Stress12 * Aux_Strain122) +
                                (Stress21 * Aux_Strain212) + (Stress22 * Aux_Strain222)) * CTCS_Q1

            I12 = ((Stress11 * AuxDisp2[0,0] * CTCS_Q1)  + (Stress12 * AuxDisp2[0,0] * CTCS_Q2) +
                  (Stress21 * AuxDisp2[1,0] * CTCS_Q1)  + (Stress22 * AuxDisp2[1,0] * CTCS_Q2))

            I22 = ((Aux_stress112 * CTCS_displacements[0,0] * CTCS_Q1) + (Aux_stress122 * CTCS_displacements[0,0] * CTCS_Q2)+
                  (Aux_stress212 * CTCS_displacements[1,0] * CTCS_Q1) + (Aux_stress222 * CTCS_displacements[1,0] * CTCS_Q2))

            I1_22 += (I12 + I22 - Strain_energy2) * linalg.det(jacobi)

    if Mode == "I" or Mode == 'II':
        K_1 = (I1_2 * D)/2
        print("K1 in Mpa root(mm):", K_1/np.sqrt(1000))

        K_2 = (I1_22 * D)/2
        print("K2 in Mpa root(mm):", K_2/np.sqrt(1000))

        K1=K_1/np.sqrt(1000)
        G_computed_I = K1**2 / D
        print("computed energy_release_rate for K_I: ", G_computed_I)

        K2=K_2/np.sqrt(1000)
        G_computed_II = K2**2 / D
        print("computed energy_release_rate for K_II: ", G_computed_II)

        Total_energy_release_rate = (K1**2 + K2**2)/D
        print("Computed total energy_release_rate: ", Total_energy_release_rate)

    elif Mode == "MIX":
        K_1 = (I1_2 * D)/2
        print("K1 in Mpa root(mm):", K_1/np.sqrt(1000))

        K_2 = (I1_22 * D)/2
        print("K2 in Mpa root(mm):", K_2/np.sqrt(1000))

        a = (1+(2*Y**2))/((1+Y**2)**(3/2))
        b = 1.3 - (0.3*(Y/m.sqrt(1+Y**2))**(5/4))
        c = 0.665- (0.267*(Y/m.sqrt(1+Y**2))**(5/4))*((Y/m.sqrt(1+Y**2))-0.73)

        F1 = a*b*c

        d = (1/(1+Y**2)) + ((Y**2)/((1+Y**2)**(3/2)))*(np.arctanh(1/m.sqrt(1+Y**2)))
        e = 1.3- (0.375*(Y/m.sqrt(1+Y**2)))*(1-0.4*(Y/m.sqrt(1+Y**2)))

        F2 = d*e

        f = (1/((1+Y**2)**1.5))
        g = 1.3-(0.75*(Y/m.sqrt(1+Y**2))*(1-1.184*(Y/m.sqrt(1+Y**2)) + (0.512 * Y**2 / (1+Y**2))))

        F3 = f*g

        Ist = (1/(m.sqrt(np.pi*CL))) * (DISP * F1)
        IInd = (1/(m.sqrt(np.pi*CL))) * (DISP * 2* F2/np.pi)
        KI_th = Ist - IInd

        IIIrd = (1/(m.sqrt(np.pi*CL))) * (DISP * F3)
        IVth = (1/(m.sqrt(np.pi*CL))) * (DISP * 2* F2/np.pi)
        KII_th = IIIrd - IVth
        print("Theoretical K_I", KI_th)
        print("Theoretical K_II", KII_th)
        Total_energy_release_rate = (K_1**2 + K_2**2)/D
        print("Computed total energy_release_rate under mixed mode: ", Total_energy_release_rate)
        Theor_Total_energy_release_rate = (KI_th**2 + KII_th**2)/D
        print("Theoretical total energy_release_rate under mixed mode: ", Theor_Total_energy_release_rate)








