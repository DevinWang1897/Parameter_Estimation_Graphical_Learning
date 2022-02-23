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

## Perform basic or constrained line parameter estimation.
Run "NLGNNdiff_par_basic.m" to perform the basic graphical learning algorithm to estiamte the line parameters. Constraints can also be added in this code. The following parameters or variables in the code can be adjusted or defined:
1. Max_para_num: maximum number of parallel threads to run.
2. In line 18, the directory of input data ("Input_struct") should be defined.
3. alg_seed_list: A list of random seeds to control the stochastic gradient descent (SGD). A use can use one seed to get one result, or a list of seeds to get multiple results.
4. error_ratio: The assumed ratio of error of the initial parameters. For example, if we assume the initial parameters are within 50% of the true values, then we can set error_ratio to 0.5.
5. batch_size: The size of the mini batches in SGD.
6. max_iter: The maximum iteration numbers of the SGD.
7. threshold_ratio: The ratio controlling how well the forward function converges. Smaller ratio means better convergence, but also means longer running time.
8. early_stop_patience: Early stop patience.
9. early_stop_threshold_ratio: A ration defining how much improvement is needed to becaled an improvement in early stopping.
10. initial_step_size: Standard step size for each back tracking line search.
11. step_dynamic_beta: variables to control the dynamic initial step size.
12. min_step_threshold: minimum step size of the dynamic initial step size.
13. max_step_threshold: maximum step size of the dynamic initial step size.
14. alpha: The alpha of back tracking line search.
15. beta: The beta of back tracking line search.
16. folder_dir: The folder directory to save the parameter estimation results.
17. description_txt: A text describing the simulation.
18. limit_range_txt: If we use range constraints, set it to 'yes', otherwise 'no'.
19. prior_adjust_ratio: must be 0 because this is a basic or constrained run, but no MAP prior is used.

The output of the code is saved in the "folder_dir" directory, which contains two kinds of files:
1. File "global_setup_save.mat", which contains the simulation parameter setups. 
2. The rest of the output files are named by the subnet number and alg_seed (SGD random seed). Suppose there are n subnetworks, then for each SGD random seed, there should be n files, one for each subnet. Each such file contains "para_set" that contains all the parameter setups of the simulation. The "save_result" contains the parameter estimation history and result. The estimated line parameters are the "w" structure in save_result.w_history_record that corresponds to lowest loss function values in the save_result.loss_fun_history.

## Perform constrained or MAP line parameter estimation.
Run "NLGNNdiff_par_prior.m" to perform the constrained or MAP line parameter estimation. The parameter setups are similar to the code of "NLGNNdiff_par_basic.m", but pay attention to the follows:
1. prior_adjust_ratio: This needs to be set to 1.
2. dir_basic_result: A user must define it correctly. This is the directory folder containing the results of the basic method. This will be used to estimate the prior distribution parameters.
3. alg_seed_list: This must be the same as the alg_seed_list with with the basic method results are collected.

The output files have similar format as "NLGNNdiff_par_basic.m".

## Summarize the output of the parameter estimation
Run "par_result_summary.m" to get a summary of the parameter estimation result. This code combines the results of each subnet and show the result of the full network. Some parameter should be defined by the user:
1. folder_dir: The directory which contains the output of parameter estimation that a user want to look at.
2. open_alg_seed_list: The list of SGD random seeds. This should match the folder_dir's files.
3. subnet_num: The number of subnest. This should match the folder_dir's files.

Output: (in the workspace after running the code)
1. MADR_improve_end_table: A table of two columns. Colum 1 is the SGD seed, column 2 is the MADR improvement (%) of using this seed in the SGD. This table is meaningless if a user do not provide the true parameter values in the "Input_struct" when doing the parameter estimation.
2. The global_w_end_full: This stores the estiamted line parameters (full network) of using different SGD seeds. A user can use this as the parameter estimation result.

