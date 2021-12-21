import numpy as np
import operator
import math
import heaviside_enrichment 
import matplotlib.pyplot as plt
import class_crack
import plots
import new_crack

# def assembly(h,i, nodes):        
#     K = np.matmul(np.transpose(h), i)
#     S = np.matmul(K, h)
#     return S

# def Assignment(EL, K_matrices, Nodes):
#     Assignment_matrices = []
#     for k in EL:
#         P = k-1
#         A,B,C,D = int(P[0]), int(P[1]), int(P[2]), int(P[3])
#         Assignment_matrices.append((Shape_function.Assignment_matrix(A,B,C,D, Nodes)))

#     S = []
#     for h, i in zip(Assignment_matrices, K_matrices):
#         S.append(assembly(h,i, Nodes*2))
    
#     a = (np.array(S[1]) + np.array(S[0]) + np.array(S[2]) + np.array(S[3]))
#     # print('=='*20)  
#     b = (sum(S))
    
#     print(f'the sum of {len(K_matrices)} stiffness_matrices is: ', np.shape(b))
#     # np.savetxt('K_matrix_non_enriched.txt', b)


def dictionary(NL,EL):
    y =[]
    for j in range(len(EL)):
        for i in EL[j]:
            y.append(tuple(NL[int(i-1)]))  
    '''
    ===========================================================================
    Formation of list of 4 nodes per element in ccw direction
    ===========================================================================
    '''   
    all_ns = []
    Hit=[]
    for k in range(len(EL)):
        l_dict = {tuple(EL[k]): y[k*4: k*4+4]}
        Hit.append(l_dict)
        z = operator.itemgetter(*l_dict.keys())(l_dict)
        all_ns.append(z)
    return all_ns, Hit


def elements(Elements, Nodes_elements,x):
    '''
    ===========================================================================
    This function generates the node numbers in ccw with respect to 
    the elements
    list of all the nodes in a ccw order of each element
        4                   5                  6
        +-------------------+------------------+    
        +                   +                  +
        +                   +                  +
        +     Element_1     +     Element_2    +
        +                   +                  +
        +                   +                  +
        +-------------------+------------------+ 
        1                   2                  3
    
    [(1,2,5,4),......
    ........(2,3,6,5)]
    ===========================================================================
    Parameters
    --------------------------
    Elements : total numbe rof elements required
    Nodes_elements : 4, as each element will have 4 nodes
    x : number of elements per row
    Returns
    --------------------------
    EL : list of elements
    '''
    EL = np.zeros([Elements, Nodes_elements])
    for i in range(1,x+1):
        for j in range(1,x+1):
            if j==1:
            
                EL[(i-1)*x+j-1, 0] = (i-1)*(x+1) + j                #EL[0,0]
                EL[(i-1)*x+j-1, 1] = EL[(i-1)*x+j-1,0] + 1          #EL[0,1]
                EL[(i-1)*x+j-1, 3] = EL[(i-1)*x+j-1,0] + x+1        #EL[0,3]
                EL[(i-1)*x+j-1, 2] = EL[(i-1)*x+j-1,3] + 1          #EL[0,2]
                
            else:
                
                EL[(i-1)*x+j-1, 0] = EL[(i-1)*x+j-2, 1]
                EL[(i-1)*x+j-1, 3] = EL[(i-1)*x+j-2, 2]
                EL[(i-1)*x+j-1, 1] = EL[(i-1)*x+j-1, 0] + 1
                EL[(i-1)*x+j-1, 2] = EL[(i-1)*x+j-1, 3] + 1
                
    return EL

def nodes(Nodes,corners, x):
    '''
    This function generates the list of nodes
    Parameters
    --------------------------------
    Nodes : number of nodes(integers)
    corners : list of 4 corners of the mesh 
    x : number of elements per row
    Returns
    --------------------------------
    NL : list of nodes
    '''
    D_2 = 2
    NL = np.zeros([Nodes, D_2])
    a = (corners[1,0] - corners[0,0]) / x     #divisions along X-axis
    b = (corners[2,1] - corners[0,1]) / x     #divisions along y-axis
    n = 0
    
    for i in range(1,x+2):
        for j in range(1, x+2):            
            NL[n,0] = corners[0,0] + (j-1)*a  #x-values of nodes         
            NL[n,1] = corners[0,1] + (i-1)*b  #y-values of nodes
            n += 1
    
    return NL
    
def uniform_mesh(A,B,x):
    '''
    ===========================================================================
    This function computes all the prerequisites for the 
    nodes and elements generations
    ===========================================================================
    Parameters
    ----------------------------------
    A : length along x_axis
    B : length along b_axis
    x : number of elements per row
    Returns
    ---------------------------------
    NL : list of nodes
    EL : list of elements
    '''
    corners = np.array([[1, 1],[A, 1],[1, B],[A, B]])    
    Nodes = (x+1)**2
    Elements = x**2
    Nodes_elements = 4

    EL = elements(Elements, Nodes_elements, x)           # 4_nodes per_element list
    NL = nodes(Nodes, corners, x)                  # nodes_list
                                                             
    return NL, EL
    
if __name__ == "__main__":
    
    A = 4            #length of the geometry along x and y axis
    B = 4
    x = 3                                          #3x3 will be elements, 4x4 will be nodes
    NL, EL = uniform_mesh(A,B,x)
    x1 = NL[0] 
    x2 = NL[1]
    '''
    length of a segment, given 2 points = sqrt((x2-x1)**2 + (y2-y1)**2)
    '''
    length_element = np.sqrt((x2[0]-x1[0])**2 + (x2[1]-x1[1])**2)
    print("the length of each element is ", length_element)
    all_ns, Hit = dictionary(NL,EL)
    # for i in all_ns:
    #     print(i)
    plots.plot_mesh(NL, A, B)
    '''
    ===========================================================================
    list of all the nodes in a ccw order of each element
     N_4                    N_5                N_6
        +-------------------+------------------+    
        +                   +                  +
        +                   +                  +
        +     Element_1     +     Element_2    +
        +                   +                  +
        +                   +                  +
        +-------------------+------------------+ 
    N_1                    N_2                N_3
    
    (N_1, N_2, N_5, N_4, N_2, N_3, N_6, N_5...............)
    ===========================================================================
    '''

    '''
    ===========================================================================
    Crack modelling
    ===========================================================================
    '''
    cracktip_1 = np.array([1, 3.5])
    cracktip_2 = np.array([1.5, 1.5])
    
    plots.plot_crack(cracktip_1, cracktip_2)
    crack_length = np.sqrt((cracktip_1[0]-cracktip_1[1])**2 + (cracktip_2[0]-cracktip_2[1])**2)
    
    print("the crack length is", crack_length)    
    c_1 = np.array([cracktip_1[0], cracktip_2[0]]) 
    c_2 = np.array([cracktip_1[1], cracktip_2[1]])       
    zeta = 2

    N = (zeta+1)**2
    Jarvis=[]
    if c_2[0] >= A:
        print(f"The crack_length {crack_length} is >= to length {A}" 
              "therefore the sample is broken into two halves")
    
    elif crack_length < length_element:
        for nodes in all_ns:
            Jarvis.append(class_crack.domain_splitter(nodes, c_1, c_2, length_element))           
            
    elif crack_length == length_element:

        for nodes in all_ns:
            Jarvis.append(class_crack.Crack_length_PP_element_length(nodes, c_1, c_2, crack_length, length_element))
          
    else:
        meta = int(crack_length/length_element)
        gamma = class_crack.crack_splitter(c_1, c_2, length_element, meta)
        v_1 = (gamma[-1])
        v_2 = (gamma[-2])
        
        if abs(v_1[0]-v_2[0])<1e-3:
            print("The next element should not be sub-divided")
            
        else:
            for l in all_ns:
                # print("*******************")
                epsilon = class_crack.domain_splitter(l, v_2, v_1, length_element)
                Jarvis.append(epsilon)
                    
        for i in range(len(gamma)-2):
            for nodes in all_ns:  
              eta = class_crack.Crack_length_PP_element_length(nodes, gamma[i], gamma[i+1], crack_length, length_element)
              Jarvis.append(eta)
              
          
    # K_matrices=[]          
    # filtered_nodes = new_crack.filtering(Jarvis, all_ns)
      
    # new_crack.ele_filtering(Hit, filtered_nodes)
        
    # for points in filtered_nodes:
    #     K_matrices.append(shape_function(np.reshape(points, (4,2))))
    '''
    ===========================================================================
    '''