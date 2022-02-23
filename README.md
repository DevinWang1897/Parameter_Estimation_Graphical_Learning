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
5. partition_table: The lines that seperate the network into smaller sub-networks for faster parallel estimaiton. Each row is a line, and the two columns are the two node IDs of each line. The lines in this table have 3-phase power flow measurements on each line. The first column is also the point where the PMU measures three-phase voltage magnitude and anlges.
6. global_source_node: The node of the whole network's head.
7. app_P: A matrix containing the real power measurement from smart meters. Each row is a smart meter, and rows corresponds to the load_id_phase's rows. Each column is a timestamp.

Note about the data:
1. All the voltage magnitude data are stored in per unit. 
2. All the voltage angle data are stored as radian.
3. All the power measuresent data are transformed as follows, the power in watt or var is devided by the square of nominal voltage (the per unit base voltage).



