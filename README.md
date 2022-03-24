# Implementation of XFEM and Fracture Mechanics to analyse stress field around the crack

## INTRODUCTION
In this program, XFEM and Fracture mechanins have been coupled to analyse the stress fields near the crack tip. A new and improved technique called Extended Finite Element Method (XFEM) for modelling of cracks in the finite element framework has been presented. Standard displacements are enriched near a crack by incorporating both discontinuous fields and the near tip asymptotic fields through a partition of unity method. The nodes that are present around the crack segments will be enriched giving rise to additional degree of freedoms. These additional DOFs are used to approximate displacements of the corresponding nodes. Displacements and stresses are computed after solving a BVP. 

The output from the XFEM will be used as the inputs to interaction integral to compute stress intensity factor. Finally, the crack is allowed to propagate. Numerical examples are provided to demonstrate the correctness and robustness of the implemented technique.

## USER MANUAL
The following steps should be performed to run the program and test cases. All
the files are written in python.

1. test Inputs.py : Test cases for input parameters

2. test geometry.py : Test cases on shape function and assembly

3. test element routine.py : Test cases on integration scheme and sanity checks
on other element.

4. test optimization.py : Test cases on optimizer function OC(optimality cri-
terion) and MMA(method of moving asymptotes)

5. test rigid body motion.py : Test cases to validate global stiffness matrix
and boundary conditions by performing rigid body translation and rotation.

6. test patch.py : Test cases on constant stress patch test and comparing an-
alytical solution with numerical for lower and higher order shape functions.

To run all the test cases, copy all test files in same folder and enter `PYTEST`
command on the terminal.
