'''
AUTHOR : VIKAS MELKOT BALASUBRAMANYAM
MATRICULATION NUMBER : 66133
Personal Programming Project: IMPLEMENTATION OF XFEM AND FRACTURE MECHANICS
#=============================================================================
'''

from time import perf_counter
import main_pgm
import cProfile
import pstats
import time
from io import StringIO
from pstats import SortKey
import Test_unit_functions
import test_patch
import Input

# code taken from https://stackoverflow.com/
def writer(tex):
    stream = StringIO()
    stats = pstats.Stats(tex, stream=stream);
    stats.print_stats()
    stream.seek(0)
    print(10*'=',"log.txt file has been generated in the same directory",10*'=')
    data = stream.read()
    return data


# code taken from https://stackoverflow.com/
prof = cProfile.Profile()
def F1():
    prof.enable()
    Test_unit_functions.test_uniform_mesh()
    prof.disable()
    prof.dump_stats("F1.prof")


prof = cProfile.Profile()
def F2():
    prof.enable()
    Test_unit_functions.test_kinematics()
    prof.disable()
    prof.dump_stats("F2.prof")


prof = cProfile.Profile()
def F3():
    prof.enable()
    Test_unit_functions.test_Global_to_local_CT()
    prof.disable()
    prof.dump_stats("F3.prof")


prof = cProfile.Profile()
def F4():
    prof.enable()
    Test_unit_functions.test_inside_circ()
    prof.disable()
    prof.dump_stats("F4.prof")


prof = cProfile.Profile()
def F5():
    prof.enable()
    Test_unit_functions.test_step_function()
    prof.disable()
    prof.dump_stats("F5.prof")


prof = cProfile.Profile()
def F6():
    prof.enable()
    Test_unit_functions.test_heaviside_functions()
    prof.disable()
    prof.dump_stats("F6.prof")


prof = cProfile.Profile()
def F7():
    prof.enable()
    Test_unit_functions.test_addAtPos()
    prof.disable()
    prof.dump_stats("F7.prof")


prof = cProfile.Profile()
def F8():
    prof.enable()
    Test_unit_functions.test_connectivity_matrix()
    prof.disable()
    prof.dump_stats("F8.prof")


prof = cProfile.Profile()
def F9():
    prof.enable()
    Test_unit_functions.test_node_filtering()
    prof.disable()
    prof.dump_stats("F9.prof")

prof = cProfile.Profile()
def F10():
    prof.enable()
    Test_unit_functions.test_asymptotic_functions()
    prof.disable()
    prof.dump_stats("F10.prof")


prof = cProfile.Profile()
def F11():
    prof.enable()
    Test_unit_functions.test_Gausspoints()
    prof.disable()
    prof.dump_stats("F11.prof")


prof = cProfile.Profile()
def F12():
    prof.enable()
    Test_unit_functions.test_tip_enrichment_func_N1()
    prof.disable()
    prof.dump_stats("F12.prof")


prof = cProfile.Profile()
def F13():
    prof.enable()
    Test_unit_functions.test_E_filter()
    prof.disable()
    prof.dump_stats("F13.prof")


#patch test---------------------------------------------------
prof = cProfile.Profile()
def F14():
    prof.enable()
    test_patch.test_LE_patch()
    prof.disable()
    prof.dump_stats("F14.prof")


prof = cProfile.Profile()
def F15():
    prof.enable()
    test_patch.test_displacement_2x2()
    prof.disable()
    prof.dump_stats("F15.prof")


prof = cProfile.Profile()
def F16():
    prof.enable()
    test_patch.test_isotropic_material_prop()
    prof.disable()
    prof.dump_stats("F16.prof")


prof = cProfile.Profile()
def F17():
    prof.enable()
    test_patch.test_Jacobian()
    prof.disable()
    prof.dump_stats("F17.prof")


prof = cProfile.Profile()
def F18():
    prof.enable()
    Input.MainFunction()
    prof.disable()
    prof.dump_stats("F18.prof")


prof = cProfile.Profile()
def F19():
    prof.enable()
    test_patch.test_Rigid_body_motions()
    prof.disable()
    prof.dump_stats("F19.prof")


prof = cProfile.Profile()
def F20():
    prof.enable()
    test_patch.test_Rigid_body_rotation()
    prof.disable()
    prof.dump_stats("F20.prof")


files = ['F1.prof','F2.prof','F3.prof','F4.prof','F5.prof','F6.prof','F7.prof','F8.prof','F9.prof',
        'F10.prof','F11.prof','F12.prof','F13.prof','F14.prof','F15.prof','F16.prof','F17.prof',
        'F18.prof','F19.prof', 'F20.prof']

Func_s = ['test_uniform_mesh', 'test_kinematics', 'test_Global_to_local_CT', 'test_inside_circ', 'test_step_function',
          'test_heaviside_functions', 'test_addAtPos', 'test_connectivity_matrix', 'test_node_filtering',
          'test_asymptotic_functions', 'test_Gausspoints', 'test_tip_enrichment_func_N1', 'test_E_filter',
          'test_LE_patch', 'test_displacement_2x2', 'test_isotropic_material_prop', 'test_Jacobian',
          'MainFunction', 'test_Rigid_body_motions','test_Rigid_body_rotation']

x = []
F1(); F2(); F3(); F4(); F5(); F6(); F7();
F8(); F9(); F10(); F11(); F12(); F13();
F14(); F15(); F16(); F17(); F18(); F19(); F20()

print("")
print(70*'=')
print(f"               TIME ANALYSIS OF ALL THE {len(files)} TEST FUNCTIONS               ")
print(70*'=')
print("")
for i in files:
    D = writer(i)
    x.append(D)

with open('log.txt', 'w') as myfile:
    myfile.write(50*'=')
    myfile.write("\n")

    for i,j in zip(x,Func_s):
        myfile.write(f"=========={j}==========")
        myfile.write("\n")
        myfile.write(i)
        myfile.write(50*'=')
        myfile.write("\n")



















