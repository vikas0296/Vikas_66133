# Implementation of XFEM and Fracture Mechanics to analyse stress field around the crack

## INTRODUCTION
In this program, XFEM and Fracture mechanins have been coupled to analyse the stress fields near the crack tip. A new and improved technique called Extended Finite Element Method (XFEM) for modelling of cracks in the finite element framework has been presented. Standard displacements are enriched near a crack by incorporating both discontinuous fields and the near tip asymptotic fields through a partition of unity method. The nodes that are present around the crack segments will be enriched giving rise to additional degree of freedoms. These additional DOFs are used to approximate displacements of the corresponding nodes. Displacements and stresses are computed after solving a BVP. 

The output from the XFEM will be used as the inputs to interaction integral to compute stress intensity factor. Finally, the crack is allowed to propagate. Numerical examples are provided to demonstrate the correctness and robustness of the implemented technique.

# USER MANUAL
The following steps should be performed to run the program and test cases. All
the files are written in python.

1. test_patch.py : The file consists of 
   a)Test case to validate global stiffness matrix and boundary conditions by performing rigid body translation.
   b)Test case to check Linear Elastic Material Response. 
   c)Test to check Isotropic material property.
   d)Test to check the sanity of Shape functions.
   e)Test to check the sanity of Jacobian matrix.
   f)Test case to validate global stiffness matrix and boundary conditions by performing rigid body rotation.
   g)Test case to validate global stiffness matrix and boundary conditions by performing rigid body motion.
   
2. Test_unit_functions.py: 13 unit tests have been recorded in this file

3. Profiler.py: The file consists of a set of functions to compute the elapsed time. When it runs, it generates 20 additional files with .prof extension.
   The output if the "files.prof" is written to "log.txt"

4. main_pgm.py: To start the program, one should run this file. Inputs are already embedded into the function.

5. Piechart.py: In order to execute the Time analysis chart, this file can be used. The data has been collected from the log file
   using bash script 

6. Documents: Here, all the necessary images, screen shots, latex files have been placed. 

To run all the test cases, copy all test files in same folder and enter `PYTEST`
command on the terminal.
