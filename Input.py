'''
AUTHOR : VIKAS MELKOT BALASUBRAMANYAM
MATRICULATION NUMBER : 66133
Personal Programming Project: IMPLEMENTATION OF XFEM AND FRACTURE MECHANICS
#=============================================================================
'''
import main_pgm

def MainFunction():
    '''
    All the crack modes will be looped through in this function.
    3 examples have been discussed here
    '''
    MODES = ['MODE_I', 'MODE_II', 'MIX_MODE']
    #looping through the modes to achieve corresponding results
    for i in MODES:
        if i == 'MODE_I':
            print(60*"=")
            print("Mode_I crack analysis")
            print(60*"=")
            DISP = 5
            Id = "I"
            main_pgm.init_function("I", DISP, Id)

        elif i == 'MODE_II':
            print(60*"=")
            print("Mode_II crack analysis")
            print(60*"=")
            DISP = 5
            Id = "II"
            main_pgm.init_function("II", DISP, Id)

        elif i == 'MIX_MODE':
            print(60*"=")
            print("Mixed Mode crack analysis")
            print(60*"=")
            FORCE = 500
            Id = "I_II"
            main_pgm.init_function("MIX", FORCE, Id)

# MainFunction()


