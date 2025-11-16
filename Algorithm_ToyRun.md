# Multistage-DRO-DDU
 Implementation of the multistage distributionally robust optimization algorithm with decision-dependent uncertainty 

This implementation is coded in Python 3.11 and it follows the Gurobi modelling style. We use Gurobi as the LP and MILP solver within the SND algorithm. Gurobi requires a license which can be obtained by visiting their webiste: https://www.gurobi.com/

This repository is organized as follows:

- Inputs: This folder contains the instances that are read by the algorithm
	- ToyGrid: Any instances will be composed of three files
        - ToyGrid.xlsx : This file contains information about buses and lines.
        - ToyGrid_TS.xlsx: This file contains the timeseries information associated with the buses in the previous excel file.
        - ToyGrid_DRO.pkl: This file contains the scenario tree.

- Algorithm: This folder contains the files required to run the SND algorithm
    - HelperFunctions.py: This file contains miscellaneous functions that are used to process data for use within the other files.
    - NodalOPF.py: This file contains the single-period optimal power flow problem and its extension into a overestimation and understimation of the cost-to-go function to solve the nodal problem within the SND algorithm.
    - StepsSND.py: This file contains the forward and backward passes of the SND algorithm.
    - ScenarioTree.py: This file contains functions to read and process the scenario tree from the xxx_DRO.pkl input file.
    - LazyConstraints.py: This file contains the functions to implement lazy constraints to generate the linear underestimations within the NodalOPF.py file.
    - CutConstructor.py: This file contains the functions that generate the different cuts to be implemented to construct the lower approximations of the cost'to'go functions within the SND algorithm.

- main.py: This file call all other files to run the SND algorithm given the indicated inputs
- InteractiveAlgorithm.ipynb: This file requires jupyter notbook and allows to have an interactive algorithm such that we can access any information we need from the algorithm.