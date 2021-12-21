import numpy as np
import uniform_mesh
import matplotlib.pyplot as plt
import plots
import new_crack
import operator
import heaviside_enrichment
import Assignment
import Tip_enrichment

def domain_splitter(nodes, c_1, c_2, length_element):
    '''
    ==========================================================================
    This functions generates the mesh for the element which is partially cut 
    by the crack.  
     N_4                 N_5               N_6
    +-------------------+------------------+    
    +                   +                  +
    +                   +                  +
    +$$$$$$$$$$$$$$$$$$$$($$$$$$)-> partial_crack 
    +                   +                  +
    +                   +                  +
    +-------------------+------------------+ 
    N_1                 N_2                N_3
    
    Suppose the element is partially cut by the crack(as in the above illustration) 
    and therefore the ""length of the crack is << the element"" length, hence, 
    this function creates a subdomain same as the elements which are 
    fully cut by the crack.
    This function separates the main element into 2 sub domains only in case of
    the length of the crack << the element length
    ==========================================================================
    Parameters
    ---------------------------------------------------------------------------
    nodes :  list of 4 nodel coordinates.
    c_1 : crack_tip 1
    c_2 : crack_tip 2
    length_element : Element length
    Returns
    ---------------------------------------------------------------------------
    The output of the function is a nodal points which are used for Tip enrichment.
    ===========================================================================
    '''
    NL_1 = nodes[0]
    NL_2 = nodes[1]
    NL_3 = nodes[2]
    NL_4 = nodes[3] 
    Tip_matrix_L = np.zeros([40, 40])
    Tip_matrix_U = np.zeros([40, 40])
    if (c_2[0] >= NL_1[0] and c_2[0] <= NL_2[0] and c_2[1] <= NL_3[1] and c_2[1]<=NL_4[1] and
        c_2[1]>NL_1[1]):
        Nodes_list = np.array([NL_1,NL_2,NL_3,NL_4])
        
        # print("outer domain points where crack has not yet crossed")
        '''
        array: outer nodes of the element
        '''
        x=2
        Nodes = (x+1)**2
        Elements = x**2
        Nodes_elements = 4 
        c_add = [0,0]
        c_add[0] = c_1[0] + length_element
        c_add[1] = c_1[1]    
        domain_elements=[]
        
        Lower_domain = np.array([nodes[0], nodes[1], c_add, c_1]) #lower domain elements
        Upper_domain = np.array([c_1,c_add,nodes[2], nodes[3]])   #upper domain elements
        
        domain_elements.append(Lower_domain)
        domain_elements.append(Upper_domain) 
        
        for d in domain_elements:
            '''fox: inner nodes of the upper and lower domain
            '''
            fox = uniform_mesh.nodes(Nodes,d,x)
            plots.plot_sub_elements(fox)
        
        uppernodes = uniform_mesh.nodes(Nodes,Upper_domain,x)
        upperelements = (uniform_mesh.elements(Elements, Nodes_elements,x))  
        '''
        Q = []
        for i in upperelements:
            if x==2:
                Q.append(i+6)
            elif x==3:
                Q.append(i+12)
            elif x==4:
                Q.append(i+20)
        
        '''
        r, theta = Assignment.to_polar_in_radians(c_2[0], c_2[1])
        
        lowernodes = uniform_mesh.nodes(Nodes,Lower_domain,x)
        lowerelements = (uniform_mesh.elements(Elements, Nodes_elements,x))
        
        Nodes_L, low = uniform_mesh.dictionary(lowernodes,lowerelements)
        Nodes_U, up = uniform_mesh.dictionary(uppernodes,upperelements)
        alpha = 0
        for i,j in zip(Nodes_L, Nodes_U):
            
            final_matrix_L = Tip_enrichment.tip_enrichment(np.round(i,3), r, theta, alpha)
            Tip_matrix_L += final_matrix_L
            final_matrix_U = Tip_enrichment.tip_enrichment(np.round(j,3), r, theta, alpha)
            Tip_matrix_U += final_matrix_U
           
        Element_matrix_Full = np.add(Tip_matrix_L, Tip_matrix_U)
        # print("its------------", Element_matrix_Full[0:8, 0:8])
        # print("its------------", Element_matrix_Full[0:8, 8:16])
        # print("its------------", Element_matrix_Full[8:16, 0:8])
        # print("its------------", Element_matrix_Full[8:16, 8:16])
        print("full_tip_matrix--------------------------")
        return Nodes_list
        
def sub_nodes_liste(a, crack_length, length_element, c_1, c_2):
    '''
    ===========================================================================
    The output of this function is used to plot the sub domains of an enriched 
    element. This function establishes the upper domain and lower domain.
    ===========================================================================
    Parameters
    ----------
    a : an array 4 nodes to be enriched
    '''
    sub_elements = []
    Lower_domain = np.array([a[0], a[1], c_2, c_1])
    Upper_domain = np.array([c_1, c_2, a[2], a[3]])

    sub_elements.append(Lower_domain)
    sub_elements.append(Upper_domain)
    
    x = 2
    Nodes = (x+1)**2
    Elements = x**2
    Nodes_elements = 4     
    
    lowernodes = uniform_mesh.nodes(Nodes,Lower_domain,x)
    lowerelements = uniform_mesh.elements(Elements, Nodes_elements,x)    
    uppernodes = uniform_mesh.nodes(Nodes,Upper_domain,x)
    upperelements = uniform_mesh.elements(Elements, Nodes_elements,x)
    
    Nodes_U, up = uniform_mesh.dictionary(uppernodes,upperelements)
    # print(Nodes_U)
    Nodes_L, low = uniform_mesh.dictionary(lowernodes,lowerelements)
    # print("lower points", Nodes_L)
    '''
    ==========================================================================
    forms element list [1234, 2345]
    Q = []
    for i in upperelements:
        if x==2:
            Q.append(i+6)
        elif x==3:
            Q.append(i+12)
        elif x==4:
            Q.append(i+20)
            
    concat = np.concatenate((lowerelements,Q),axis=0) #elements of sub-polygons
    print(concat) 
    ======================================================================= 
    in assignment file "Heaviside enrichment" function has been called     
    '''
    final_matrix_L = Assignment.set_matrix(np.round(Nodes_L, 3), H2=0)
    final_matrix_U = Assignment.set_matrix(np.round(Nodes_U, 3), H2=1)

    Element_matrix_Full = np.add(final_matrix_L, final_matrix_U)
    
    # print("its------------", Element_matrix_Full[0:8, 0:8])
    # print("its------------", Element_matrix_Full[0:8, 8:16])
    # print("its------------", Element_matrix_Full[8:16, 0:8])
    # print("its------------", Element_matrix_Full[8:16, 8:16])
    # print("its------------", Element_matrix_Full)
    for m in sub_elements:
        '''
        nodes created after the sub_division
        '''
        zeta = uniform_mesh.nodes(Nodes,m,x)
        plots.plot_sub_elements(zeta)
        
    return sub_elements
             
def Crack_length_PP_element_length(nodes, c_1, c_2, crack_length, length_element):
    '''
    ===========================================================================
    This function will evaluate if the element contains a 
    crack and the crack length should be of the same length as that of
    the element and it returns the list of the nodes to be enriched.
    ===========================================================================
    Parameters
    ----------
    nodes : list of 4 nodel coordinates.
    c_1 : coordinates of crack_tip 1.
    c_2 : coordinates of crack_tip 2.
    counter : integer value to keep track of the element where crack exist. 
    Returns
    -------
    nodes : list of 4 nodes which require enrichment 
    '''
    NL_1 = nodes[0]
    NL_2 = nodes[1]
    NL_3 = nodes[2]
    NL_4 = nodes[3] 
    dist_1 = np.sqrt((NL_1[0]-c_1[0])**2 + (NL_1[1]-c_1[1])**2)
    dist_2 = np.sqrt((NL_2[0]-c_2[0])**2 + (NL_2[1]-c_2[1])**2)          
    
    if  dist_1 < length_element and dist_2 < length_element and c_1[1] > NL_1[1] and c_2[1] > NL_2[1]:
        array = np.array([NL_1,NL_2,NL_3,NL_4])
        '''
        selection of elements which require enrichment
        '''        
        sub_list = sub_nodes_liste(array, crack_length, length_element, c_1, c_2)
        # new_crack.list_of_Elements(sub_list)
        Nodes_list = np.array([NL_1,NL_2,NL_3,NL_4])
        # print("after filtering", Nodes_list)
        
        return Nodes_list


def crack_splitter(x_1, x_2, length_element,z):
    '''
    ===========================================================================
    This function splits the length of the crack in equal proportion as 
    that of the length of the element.
    This function is called only in case of crack_length >> element_length
    ===========================================================================
    Parameters
    ----------
    x_1 : tip_1
    x_2 : tip_2
    length_element : Element length
    xx : points holder
    z : integer value of the number of points to split
    Returns
    -------
    list of intersection points.
    N_4                 N_5               N_6
    +-------------------+------------------+    
    +                   +                  +
    +                   +                   +
    +@------------------@(----@)-> crack   +
    +                   +                  +
    +                   +                  +
    +-------------------+------------------+ 
    N_1                 N_2                N_3
    In the above example , the function outputs "@"-values
    '''    
    c_new = [0,0]
    xx=[x_1]
    for i in range(z):  
        c_new[0] = x_1[0] + (i * length_element)
        c_new[1] = x_1[1]
        c_new[0] += length_element
        if c_new[0] <= x_2[0]: 
            xx.append(np.append(c_new[0], c_new[1]))    
    xx.append(x_2)
    return xx

