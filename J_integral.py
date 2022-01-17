import numpy as np
from scipy import linalg
import new_crack
import sys
import uniform_mesh
import math as m

def Kinematics(grad_disps):

    Epsilon_ij = 0.5*np.array([[(grad_disps[0,0] + grad_disps[0,0]), (grad_disps[0,1] + grad_disps[1,0])],
                               [(grad_disps[1,0] + grad_disps[0,1]), (grad_disps[1,1] + grad_disps[1,1])]])


    return Epsilon_ij[0,0], Epsilon_ij[0,1], Epsilon_ij[1,0], Epsilon_ij[1,1]

def inside_rect(bl, tr, point):

   if (point[0] > bl[0] and point[0] < tr[0] and point[1] > bl[1] and point[1] < tr[1]):
      return True
   else:
      return False

def Global_to_local_CT(S, alpha):
    sigma_11 = ((S[0] + S[1])/2) + ((S[0] - S[1])/2 )*m.cos(2*alpha) + S[2]*m.sin(2*alpha)
    sigma_22 = ((S[0] + S[1])/2) - ((S[0] - S[1])/2 )*m.cos(2*alpha) - S[2]*m.sin(2*alpha)
    sigma_12 = -((S[0] - S[1])/2)*m.sin(2*alpha) + S[2]*m.cos(2*alpha)
    sigma_21 = sigma_12

    return sigma_11, sigma_12, sigma_21, sigma_22


def Interaction_integral(Nodes, displacements, stress, strains, set4Gauss, P2, P3, P4, P5,
                         c_2, GaussPoint_1to4, D_plane_stress, alpha, CL, A, force):

    I1_2 = 0
    CTCS = np.array([[m.cos(alpha), m.sin(alpha)],
                     [-m.sin(alpha), m.cos(alpha)]])

    for N, i, Sg, Ss, Sn in zip(Nodes, displacements, set4Gauss, stress, strains):
        X = np.array(i[0])

        for points, gps, Srs, Str in zip(GaussPoint_1to4, Sg, Ss, Sn):
            xi_1 = points[0]
            xi_2 = points[1]

            NxNy = (1/4) * np.array([[-(1-xi_2), 1-xi_2, 1+xi_2, -(1+xi_2)],
                                      [-(1-xi_1), -(1+xi_1), 1+xi_1, 1-xi_1]])

            jacobi = np.matmul(NxNy, N)

            inverse_jacobi = linalg.inv(jacobi)

            dN = np.matmul(inverse_jacobi, NxNy)

            B_std = np.array([[dN[0, 0], 0, dN[0, 1], 0, dN[0, 2], 0, dN[0, 3], 0],
                              [0, dN[1, 0], 0, dN[1, 1], 0, dN[1, 2], 0, dN[1, 3]],
                              [dN[1, 0], dN[0, 0], dN[1, 1], dN[0, 1], dN[1, 2], dN[0, 2], dN[1, 3], dN[0, 3]]])


            Ux = np.matmul(B_std[0,0:8:2], X[0:8:2])                                    # Derivative of U with respect to X
            Uy = np.matmul(B_std[1,1:8:2], X[0:8:2])                                    # Derivative of U with respect to Y
            Vx = np.matmul(B_std[0,0:8:2], X[1:8:2])                                    # Derivative of V with respect to X
            Vy = np.matmul(B_std[1,1:8:2], X[1:8:2])

            grad_disps = np.array([[Ux, Uy],
                                    [Vx, Vy]])

            CTCS_displacements = np.matmul((np.matmul(CTCS, grad_disps)),CTCS.T)

            Xgp = gps[0]
            Ygp = gps[1]
            XX = Xgp - c_2[0]
            YY = Ygp - c_2[1]
            XYdist = np.matmul(CTCS, np.array([XX,YY]).T)
            r = np.sqrt(XYdist[0]**2 + XYdist[1]**2)
            theta = m.atan2(XYdist[1], XYdist[0])

            Stress11, Stress12, Stress21, Stress22 = Global_to_local_CT(Srs, alpha)
            Strain11, Strain12, Strain21, Strain22 = Kinematics(grad_disps)


            E,F,G,H = N[0], N[1], N[2], N[3]

            E = inside_rect(P5, P3, N[0])
            F = inside_rect(P5, P3, N[1])
            G = inside_rect(P5, P3, N[2])
            H = inside_rect(P5, P3, N[3])

            if E == True:
                q1 = 1
            else:
                q1 = 0

            if F == True:
                q2 = 1
            else:
                q2 = 0

            if G == True:
                q3 = 1
            else:
                q3 = 0

            if H == True:
                q4 = 1
            else:
                q4 = 0

            nodal_vals = (1/4) * np.array([[-(1-xi_2)*q1, (1-xi_2)*q2, (1+xi_2)*q3, -(1+xi_2)*q4],
                                            [-(1-xi_1)*q1, -(1+xi_1)*q2, (1+xi_1)*q3, (1-xi_1)*q4]])

            CTCS_Q = np.matmul(CTCS, nodal_vals)
            CTCS_Q1 = np.sum(CTCS_Q[0])
            CTCS_Q2 = np.sum(CTCS_Q[1])
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

            D = 205680
            K1 = 1
            nu = 0.25

            k = (3-nu)/(1+nu)
            Auxiliary_disps = np.zeros([2,2])

            du1dr = (K1*(1+nu)/2*D) * (1/np.sqrt(2*np.pi*r)) * (CT2*(k-CT))
            du1dt = (K1*(1+nu)/D) * np.sqrt(r/(2*np.pi)) * (-0.5*ST2*(k-CT) + ST*CT2)
            du2dr = (K1*(1+nu)/2*D) * (1/np.sqrt(2*np.pi*r)) * (ST2*(k-CT))
            du2dt = (K1*(1+nu)/D) * np.sqrt(r/(2*np.pi)) * (0.5*CT2*(k-CT) + ST*ST2)

            Auxiliary_disps[0,0] = du1dr * drdx + du1dt * dtdx
            Auxiliary_disps[0,1] = du1dr * drdy + du1dt * dtdy
            Auxiliary_disps[1,0] = du2dr * drdx + du2dt * dtdx
            Auxiliary_disps[1,1] = du2dr * drdy + du2dt * dtdy

            Aux_Strain11, Aux_Strain22, Aux_Strain12, Aux_Strain21 = Kinematics(Auxiliary_disps)

            Aux_stress11 = (K1 / np.sqrt(2*np.pi*r)) * CT2* (1 - ST2 * S3T2)
            Aux_stress22 = (K1 / np.sqrt(2*np.pi*r)) * CT2* (1 + ST2 * S3T2)
            Aux_stress12 = (K1 / np.sqrt(2*np.pi*r)) * ST2 * CT2 * C3T2
            Aux_stress21 = Aux_stress12


            Strain_energy = (((Stress11 * Aux_Strain11) + (Stress21 * Aux_Strain21))+
                             ((Stress12 * Aux_Strain12) + (Stress22 * Aux_Strain22))) * CTCS_Q1

            I1 = (((Stress11 * Auxiliary_disps[0,0]) + (Stress21 * Auxiliary_disps[1,0]))*CTCS_Q1 +
                  ((Stress12 * Auxiliary_disps[0,1]) + (Stress22 * Auxiliary_disps[1,1]))*CTCS_Q2)

            I2 = (((Aux_stress11 * CTCS_displacements[0,0]) + (Aux_stress21 * CTCS_displacements[1,0]))*CTCS_Q1 +
                  ((Aux_stress12 * CTCS_displacements[0,1]) + (Aux_stress22 * CTCS_displacements[1,1]))*CTCS_Q2)

            I1_2 += (-Strain_energy + I1 + I2) * linalg.det(jacobi)


    # print(I1_2)
    K_1 = (I1_2 * D)/2
    print(K_1/1000)
    denom = 2 * np.sqrt(np.pi * CL/A) * (1-(CL/A))**1.5
    theory1 = (force * A) * m.sqrt(np.pi * CL) * ((1+(3*(CL/A)))/denom)
    print(theory1/1000)
    theory = (force * A) * m.sqrt(np.pi*CL) * (1.122 - 0.231*(CL/A) + 10.55*(CL/A)**2 - 21.71*(CL/A)**3 + 30.328 * (CL/A)**4)
    print(theory/1000)
    G = (K_1/1000)**2 / D
    print(G)



