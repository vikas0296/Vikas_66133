import numpy as np
from scipy import linalg
import sys
import uniform_mesh
import math as m
import Tip_enrichment; import plots

def r_t(QP, c_2, alpha):

    CTCS = np.array([[m.cos(alpha), m.sin(alpha)],
                     [-m.sin(alpha), m.cos(alpha)]])

    Xn = (QP[0] - c_2[0])
    Yn = (QP[1] - c_2[1])
    XYdist = np.matmul(CTCS, np.array([Xn,Yn]))
    r = np.sqrt(XYdist[1]**2 + XYdist[0]**2)
    theta = m.atan2(XYdist[1], XYdist[0])
    return round(r,3), round(theta,3)

def Kinematics(grad_disps):

    Epsilon_ij = 0.5*np.array([[(grad_disps[0,0] + grad_disps[0,0]), (grad_disps[0,1] + grad_disps[1,0])],
                                [(grad_disps[1,0] + grad_disps[0,1]), (grad_disps[1,1] + grad_disps[1,1])]])

    return Epsilon_ij[0,0], Epsilon_ij[0,1], Epsilon_ij[1,0], Epsilon_ij[1,1]

def inside_circ(c_2, length_element, P, scale):

    area = length_element * length_element
    radius = scale*np.sqrt(area)
    distance = np.sqrt((P[0]-c_2[0])**2 + (P[1]-c_2[1])**2)
    if distance <= radius:
        return True
    elif distance > radius:
        return False

def Global_to_local_CT(S, CTCS):

    Y = np.matrix(np.array([[S[0], S[2]],
                            [S[2], S[1]]]))
    sigma = np.matmul((np.matmul(CTCS,Y)), CTCS.T)

    return sigma[0,0], sigma[0,1], sigma[1,0], sigma[1,1]


def Interaction_integral(Nodes, displacements, stress, strains, set4Gauss, c_2, GaussPoint_1to4,
                             length_element, D_plane_stress, alpha, CL, A, force, D, scale, increment):


    I1_2 = 0
    I1_22 = 0
    AuxDisp2 = np.zeros([2,2])
    Auxiliary_disps = np.zeros([2,2])
    CTCS = np.array([[m.cos(alpha), m.sin(alpha)],
                     [-m.sin(alpha), m.cos(alpha)]])

    J =[]
    IS =[]
    for i,j,k,l,p in zip(Nodes, displacements,stress, strains, set4Gauss):
        X = np.array(j[0])
        Q = inside_circ(c_2, length_element, i[0], scale)
        W = inside_circ(c_2, length_element, i[1], scale)
        V = inside_circ(c_2, length_element, i[2], scale)
        R = inside_circ(c_2, length_element, i[3], scale)

        if Q == True:
            q1 = 1
            J.append(i[0])
        else:
            q1 = 0

        if W == True:
            q2 = 1
            J.append(i[1])
        else:
            q2 = 0

        if V == True:
            q3 = 1
            J.append(i[2])
        else:
            q3 =0

        if R == True:
            q4 = 1
            J.append(i[3])
        else:
            q4 = 0

        for points, Ns, gps, Srs, Str in zip(GaussPoint_1to4,i, p, k, l):
            xi_1 = points[0]
            xi_2 = points[1]

            NxNy = (1/4) * np.array([[-(1-xi_2), 1-xi_2, 1+xi_2, -(1+xi_2)],
                                     [-(1-xi_1), -(1+xi_1), 1+xi_1, 1-xi_1]])

            jacobi = np.matmul(NxNy, i)

            inverse_jacobi = linalg.inv(jacobi)
            dN = np.matmul(inverse_jacobi, NxNy)

            B_std = np.array([[dN[0, 0], 0, dN[0, 1], 0, dN[0, 2], 0, dN[0, 3], 0],
                              [0, dN[1, 0], 0, dN[1, 1], 0, dN[1, 2], 0, dN[1, 3]],
                              [dN[1, 0], dN[0, 0], dN[1, 1], dN[0, 1], dN[1, 2], dN[0, 2], dN[1, 3], dN[0, 3]]])


            Ux = np.matmul(B_std[0,0:8:2], X[0:8:2])                                    # Derivative of U with respect to X
            Uy = np.matmul(B_std[1,1:8:2], X[0:8:2])                                    # Derivative of U with respect to Y
            Vx = np.matmul(B_std[0,0:8:2], X[1:8:2])                                    # Derivative of V with respect to X
            Vy = np.matmul(B_std[1,1:8:2], X[1:8:2])

            grad_disps = np.matrix(np.array([[(Ux/1e3), (Uy/1e3)],
                                              [(Vx/1e3), (Vy/1e3)]]))

            # grad_disps = np.matrix(np.array([[Ux, Uy],
            #                                   [Vx, Vy]]))

            CTCS_displacements = np.matmul((np.matmul(CTCS, grad_disps)), CTCS.T)
            # print(CTCS_displacements)

            R, theta = r_t(gps, c_2, alpha)
            r = R/1000

            Stress11, Stress12, Stress21, Stress22 = Global_to_local_CT(Srs, CTCS)


            nodal_vals = np.array([[dN[0, 0]*q1, dN[0, 1]*q2, dN[0, 2]*q3, dN[0, 3]*q4],
                                   [dN[1, 0]*q1, dN[1, 1]*q2, dN[1, 2]*q3, dN[1, 3]*q4]])

            OP = np.array([[np.sum(nodal_vals[0])],
                           [np.sum(nodal_vals[1])]])

            CTCS_Q = np.matmul(CTCS, OP)
            CTCS_Q1 = CTCS_Q[0]
            CTCS_Q2 = CTCS_Q[1]

            # Auxilary terms-------------------------------------------------------
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
            K1 = 1
            nu = 0.25

            k = (3-nu)/(1+nu)
            du1dr = (K1*(1+nu)/(2*D)) * (1/np.sqrt(2*np.pi*r)) * (CT2*(k-CT))
            du1dt = (K1*(1+nu)/D) * np.sqrt((r)/(2*np.pi)) * (-0.5*ST2*(k-CT) + ST*CT2)
            du2dr = (K1*(1+nu)/(2*D)) * (1/np.sqrt(2*np.pi*r)) * (ST2*(k-CT))
            du2dt = (K1*(1+nu)/D) * np.sqrt(r/(2*np.pi)) * (0.5*CT2*(k-CT) + ST*ST2)

            Auxiliary_disps[0,0] = (du1dr * drdx) + (du1dt * dtdx)
            Auxiliary_disps[0,1] = (du1dr * drdy) + (du1dt * dtdy)
            Auxiliary_disps[1,0] = (du2dr * drdx) + (du2dt * dtdx)
            Auxiliary_disps[1,1] = (du2dr * drdy) + (du2dt * dtdy)

            Aux_Strain11, Aux_Strain12, Aux_Strain21, Aux_Strain22 = Kinematics(Auxiliary_disps)

            Aux_stress11 = (K1 / np.sqrt(2*np.pi*r)) * CT2 * (1 - ST2 * S3T2)
            Aux_stress22 = (K1 / np.sqrt(2*np.pi*r)) * CT2 * (1 + ST2 * S3T2)
            Aux_stress12 = (K1 / np.sqrt(2*np.pi*r)) * ST2 * CT2 * C3T2
            Aux_stress21 = Aux_stress12

            Strain_energy =  ((Stress11 * Aux_Strain11) + (Stress12 * Aux_Strain12) +
                              (Stress21 * Aux_Strain21) + (Stress22 * Aux_Strain22)) * CTCS_Q1

            I1 = ((Stress11 * Auxiliary_disps[0,0] * CTCS_Q1)  + (Stress12 * Auxiliary_disps[0,0] * CTCS_Q2) +
                  (Stress21 * Auxiliary_disps[1,0] * CTCS_Q1)  + (Stress22 * Auxiliary_disps[1,0] * CTCS_Q2))


            I2 = ((Aux_stress11 * CTCS_displacements[0,0] * CTCS_Q1) + (Aux_stress12 * CTCS_displacements[0,0] * CTCS_Q2)+
                  (Aux_stress21 * CTCS_displacements[1,0] * CTCS_Q1) + (Aux_stress22 * CTCS_displacements[1,0] * CTCS_Q2))

            I1_2 += ((I1 + I2 - Strain_energy)) * linalg.det(jacobi)

            K2 = 1
            Aux_stress112 = -(K2/np.sqrt(2*np.pi*r)) * ST2*(2+CT2*C3T2)
            Aux_stress222 = (K2/np.sqrt(2*np.pi*r)) * ST2*CT2*C3T2
            Aux_stress122 = (K2/np.sqrt(2*np.pi*r)) * CT2*(1-ST2*S3T2)
            Aux_stress212 =  Aux_stress122

            du1dr2 =  (K2*(1+nu)/(2*D)) * (1/np.sqrt(2*np.pi*r)) * ST2*(k+2+CT)
            du1dt2 =  (K2*(1+nu)/(D)) * np.sqrt((r)/(2*np.pi)) * (0.5*CT2*(k+2+CT)-ST2*ST)
            du2dr2 =  (K2*(1+nu)/(2*D)) * (1/np.sqrt(2*np.pi*r)) * CT2*(k-2+CT)
            du2dt2 = -(K2*(1+nu)/(D)) * np.sqrt((r)/(2*np.pi)) * (-0.5*ST2*(k-2+CT)-CT2*ST)

            AuxDisp2[0,0] = du1dr2 * drdx + du1dt2 * dtdx
            AuxDisp2[0,1] = du1dr2 * drdy + du1dt2 * dtdy
            AuxDisp2[1,0] = du2dr2 * drdx + du2dt2 * dtdx
            AuxDisp2[1,1] = du2dr2 * drdy + du2dt2 * dtdy

            Aux_Strain112, Aux_Strain122, Aux_Strain212, Aux_Strain222 = Kinematics(AuxDisp2)

            Strain_energy2 =  ((Stress11 * Aux_Strain112) + (Stress12 * Aux_Strain122) +
                              (Stress21 * Aux_Strain212) + (Stress22 * Aux_Strain222)) * CTCS_Q1

            I12 = ((Stress11 * AuxDisp2[0,0] * CTCS_Q1)  + (Stress12 * AuxDisp2[0,0] * CTCS_Q2) +
                  (Stress21 * AuxDisp2[1,0] * CTCS_Q1)  + (Stress22 * AuxDisp2[1,0] * CTCS_Q2))

            I22 = ((Aux_stress112 * CTCS_displacements[0,0] * CTCS_Q1) + (Aux_stress122 * CTCS_displacements[0,0] * CTCS_Q2)+
                  (Aux_stress212 * CTCS_displacements[1,0] * CTCS_Q1) + (Aux_stress222 * CTCS_displacements[1,0] * CTCS_Q2))

            I1_22 += (I12 + I22 - Strain_energy2) * linalg.det(jacobi)


    K_1 = (I1_2 * D)/2
    print("K1 is", K_1)
    K_2 = (I1_22 * D)/2
    print("K2 is", K_2)
    denom = 1 + np.sqrt(1+(8*(K_2/K_1)**2))
    T = 2 * m.atan(-2*(K_2/K_1) / denom)  #critical_angle

    print("Tcr", T*180/np.pi)

    Keff = np.sqrt(K_1**2 + K_2**2)
    K_IC = 50
    if Keff >= K_IC:
        print("its greater >>>")
        NP = plots.new_crack(c_2[0], c_2[1], T, increment)
        print("new/ppoo", NP)
        return NP, K_1[0], T

    else:
        print("keff<kic")
        NP = c_2
        T=0
        return NP, K_1[0], T


    # GG=[]
    # for i in J:
    #     if i not in GG:
    #         GG.append(i)

    # # print(len(GG))
    # plots.normal(GG)
    # K_TH = ((force * A*1e-6) * m.sqrt(np.pi*CL*1e-3)
    #             *(1.122 - 0.231*(CL/A) + 10.55*(CL/A)**2 - 21.71*(CL/A)**3 + 30.328 * (CL/A)**4))
    # print("theoretical SIF", K_TH)
