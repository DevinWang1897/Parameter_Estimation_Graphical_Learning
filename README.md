# Parameter_Estimation_Graphical_Learning
Distribution line parameter (impedance) estimation with physics-informed graphical learning method.

# Overview
This project performs distribution line parameter (impedance) estimation with a physics-informed graphical learning method. The input and output are as follows:
Input of the model: initial inaccurate line parameters, distribution network topology, measurement data of smart meters, SCADA and PMU (if there are).
Output of the model: the estimated parameter estimation results.

# How to Use
## Before using the code, a user should prepare all the needed input data and variables. These data and variables are saved in a file containing a matlab structure called "Input_struct". 
A user can refer to an example of "Input_struct" in the file "SaveData/z20220220_input_for_para_est.mat", which is an example of the modified IEEE 37-bus feeder, with the PMUs installed at node 702, 708, and the network is partitioned by cutting line 702-703 and 708-733. The "Input_struct" contains the following fields:
1. w_start: The initial guess, or an inaccurate record of the line parameters from the GIS. It is stored as a structure with three variables--line, r (for resistance) and x (for reactance). The number of recordes is the number of three-phase lines. The "line" records the staring and ending node number of each line, the "r" records 6 resistance values of the correponding line, of phase (aa, ab, ac, bb, bc, cc), the "x" records 6 reactance values of the corresponding line, of similar phase orders.
2. w_z_correct: The correct value of line parameters. It has the same structure as w_start. If a user has the correct value of line parameters, w_z_correct can be used to evaluate the performance of the algorithm. If a use does not have the correct values, he/she can fill in randome fake values and w_z_correct is only a placeholder in the code.
3. line_node_list: The list of nodes in the distribution network's 3-phase primary lines.
4. main_line_config: The topology of the network. It has 2 columns. Each row is a line, and the two columns are the two node IDs of each line.
5. partition_table: The lines that seperate the network into smaller sub-networks for faster parallel estimaiton. Each row is a line, and the two columns are the two node IDs of each line. The lines in this table have 3-phase power flow measurements on each line. The first column is also the point where the PMU measures three-phase voltage magnitude and anlges. If no PMUs are installed, this table is set to an empty array.
6. global_source_node: The node of the whole network's head.
7. app_P: A matrix containing the real power measurement from smart meters. Each row is a smart meter, and rows corresponds to the load_id_phase's rows. Each column is a timestamp.
8. app_Q: A matrix containing the reactive power measurement from smart meters. It has the same format as app_P.
9. app_V_mag: A matrix containing the voltage magnitude measurement from smart meters. It has the same format as app_P.
10. app_PMU_P: The real power flow measurement of PMUs corresponding to partition_table. If partition_table has n rows, then app_PMU_P has 3n rows. Each 3 rows correspond to 3 phases (a, b, c). Each column is a timestamp. If no PMUs are used, then this filed is an empty array.
11. app_PMU_Q: The reactive power flow measurementof PMUs corresponding to partition_table. It has the same format as app_PMU_P. If no PMUs are used, then this filed is an empty array.
12. load_id_phase: The table of all the loads. Column 1 is the list of load IDs, column 2 is the phase connections. 1 for a, 2 for b, 3 for c. Column 3 represents the voltage measurement phase if the load is a 3-phase load; if it is not a 3-phase load, then column 3 is NaN.
13. load_to_main_node: The table of each load and its "main node", i.e., the 3-phase node in the 3-phase primary network. Column 1 is the load ID, column 2 is the main node ID.
14. source_V_mag: the 3-phase voltage magnitude of the network's head. Each column is a timestamp.
15. source_V_rad: the 3-phase voltage angle of the network's head. Each column is a timestamp.
16. source_P: the 3-phase real power injection of the network's head. Each column is a timestamp.
17. source_Q: the 3-phase reactive power injection of the network's head. Each column is a timestamp.
18. PMU_V_mag: the 3-phsae voltage magnitude of PMUs corresponding to partition_table. If partition_table has n rows, then PMU_V_mag has 3n rows. Each 3 rows correspond to 3 phases (a, b, c). Each column is a timestamp. If no PMUs are used, then this filed is an empty array.
19. PMU_V_rad: the 3-phsae voltage angle of PMUs corresponding to partition_table. It has the same format as PMU_V_mag. If no PMUs are used, then this filed is an empty array.

Note about the data:
1. All the voltage magnitude data are stored in per unit. 
2. All the voltage angle data are stored as radian.
3. All the power measuresent data are transformed as follows, the power in watt or var is devided by the square of nominal voltage (the per unit base voltage).



