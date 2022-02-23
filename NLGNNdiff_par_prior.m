% 2022-02-21: This code will do parameter estimation over each partitioned
% sub-network.

% Before running, check:
% (1) The setting of parameters.



clear all

%% Set up parallel compuation for matlab
Max_para_num=3; % maximum number of parallel threads.
delete(gcp('nocreate'));%delete existing pool.
parpool('local',Max_para_num); %initialize parallel pool.

%% Load input data
Input_struct=[];
load('SaveData/z20220220_input_for_para_est')

w_start=Input_struct.w_start;
w_z_correct=Input_struct.w_z_correct;
line_node_list=Input_struct.line_node_list;
main_line_config=Input_struct.main_line_config;
partition_table=Input_struct.partition_table;
% partition_table=[];
global_source_node=Input_struct.global_source_node;
app_P=Input_struct.app_P;
app_Q=Input_struct.app_Q;
app_V_mag=Input_struct.app_V_mag;
app_PMU_P=Input_struct.app_PMU_P;
app_PMU_Q=Input_struct.app_PMU_Q;
load_id_phase=Input_struct.load_id_phase;
load_to_main_node=Input_struct.load_to_main_node;
source_V_mag=Input_struct.source_V_mag;
source_V_rad=Input_struct.source_V_rad;
source_P=Input_struct.source_P;
source_Q=Input_struct.source_Q;
PMU_V_mag=Input_struct.PMU_V_mag;
PMU_V_rad=Input_struct.PMU_V_rad;

%% parameter definition
% alg_seed_list=[79,83,89,97,101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193,197,199];
alg_seed_list=[1]; % seed for SGD algorithm random number.

error_ratio=0.5; % if we assume the initial parameters are within 50% of the correct value, use 0.5
batch_size=10;
max_iter=2e3; % 2e3 by default

% end_threshold_ratio=1e-20;
threshold_ratio=1e-20; % threshold to stop iteration. It is a ratio indicating how much
% is changed compared with the previous iteration.
early_stop_patience=10; % number of epochs of no improvement, after which stop the iteration.
early_stop_threshold_ratio=0.01; % decide improvement: at least early_stop_threshold_ratio*initial obj function value.

% Backtracking linesearch parameter
initial_step_size=1e3; % standard step size
alpha=0.3; % Backtracking line search parameter.
beta=0.5; % Backtracking line search parameter.

step_dynamic_beta=2; % parameter to decide dynamic initial step size
min_step_threshold=1e-20;% minimum initial step size when using dynamic initial step size, default=1e-20.
max_step_threshold=1e3;

folder_dir='SaveResult/20220222_37b_GitHub_prior';
description_txt='3-parition NLGNNdiff, 37 bus balanced. No noise. Use basic results as priror.';

limit_range_txt='yes'; % yes or no, if range limit is used.
prior_adjust_ratio=1; % a ratio to adjust the regularization factor.
dir_basic_result='SaveResult/20220222_37b_GitHub';% dir to load basic NLGNNdiff estiamtion result, for primal_var estimation. (if no such folder, comment out this line for basic tests.)

if isfolder(folder_dir) % if the folder already exists
    disp('SAVE FOLDER ALREADY EXISTS!!')
    return
else
    mkdir(folder_dir);%else, create the folder
end


%% load and prepare data
% TUNABLE HERE

if ~isempty(partition_table)
    all_source_node_list=[global_source_node;partition_table(:,1)]; % this is the full source and pseudo source node
else
    all_source_node_list=global_source_node;
end


% nodal_data_clean=nodal_data; %remember to change back!!!!


% prepare intermediate node data (here add PMU 3-phase meter data)
% Note that not all such inter node volt will be used in the subnets.
inter_node=all_source_node_list;
inter_V_mag=[source_V_mag;PMU_V_mag];


%% build weight strucutre (line parameter) from the primitive Y of each line.

% The main_node_list is also the order for constructing the MLE model.
main_node_list=unique([main_line_config(:,1);main_line_config(:,2)]); %this is the node list for the reduced feeder model
% main_line_config=line3p_config_matrix(:,1:3);% a table used for MLE code. Col 1--node A, col 2--node B, col 3--line length in ft.
main_node_num=length(main_node_list);
num_L=size(main_line_config,1); % number of lines


%% Prepare label (nodal real and reactive power)
% source_node=650;
source_idx=find(main_node_list==global_source_node);%find the index of the source node
main_nonsource_node_list=main_node_list;
main_nonsource_node_list(source_idx)=[];
main_nonsource_node_num=length(main_nonsource_node_list);

% remove the unmeasured point
load_id_list=load_id_phase(:,1);
% load_id_phase=nonsource_id_phase;
I1=ismember(load_id_list,load_to_main_node(:,1));
temp_idx=find(~I1);
load_id_list(temp_idx)=[];
load_id_phase(temp_idx,:)=[];
load_num=length(load_id_list);
temp_idx_3=temp_idx*3-3+(1:3);
app_P(temp_idx,:)=[];
app_Q(temp_idx,:)=[];
app_V_mag(temp_idx,:)=[];


% the linear model, U_hats correspond to main node's nodal power input, U1 and U2
% correspond to nonsource main node nodal volt, X correspond to matching
% phase to input load and output volt measurement. In the GNN model, we
% modify to include source node in the U_hats, for easier coding.
X=[];
U1=zeros(3*load_num,3*main_nonsource_node_num); % U1 correspond to output, not changed for label calculation.
U2=zeros(3*load_num,3*main_nonsource_node_num);
U_hat_1=zeros(3*main_node_num,3*load_num); % change in order to properly calcualte label
U_hat_2=zeros(3*main_node_num,3*load_num);
U_hat_3=zeros(3*main_node_num,3*load_num);
U_hat_4=zeros(3*main_node_num,3*load_num);

for i=1:load_num % loop through every cust node (cust_id_list is the order of the node in the data tables)
    temp_id=load_id_list(i);
    I1=find(load_id_phase(:,1)==temp_id);
    temp_phase=load_id_phase(I1,2);
    temp_3p_phase=load_id_phase(I1,3); % if it is 3-phase, which phase does it measure volt.
%     temp_3p_phase=1; % remember to change back!!!!
    u_2p=0.5*sqrt(3)*[1 1 0; 0 1 1; 1 0 1];
    u_hat_1=0.5*[1 0 1;1 1 0;0 1 1];
    u_hat_2=(1/2/sqrt(3))*[1 0 -1;-1 1 0;0 -1 1];
    u_hat_3=(1/2/sqrt(3))*[-1 0 1;1 -1 0;0 1 -1];
    u_hat_4=0.5*[1 0 1;1 1 0;0 1 1];
    
    temp_u2_2p=0.5*[1 -1 0; 0 1 -1; -1 0 1];
    
    switch temp_phase % each phaes connection case
        case 1 %phase a
            temp_x=[1 0 0];
            temp_u1=eye(3);
            temp_u2=zeros(3);
            temp_u_hat_1=eye(3);
            temp_u_hat_2=zeros(3);
            temp_u_hat_3=zeros(3);
            temp_u_hat_4=eye(3);
        case 2 %phase b
            temp_x=[0 1 0];
            temp_u1=eye(3);
            temp_u2=zeros(3);
            temp_u_hat_1=eye(3);
            temp_u_hat_2=zeros(3);
            temp_u_hat_3=zeros(3);
            temp_u_hat_4=eye(3);
        case 3 % phase c
            temp_x=[0 0 1];
            temp_u1=eye(3);
            temp_u2=zeros(3);
            temp_u_hat_1=eye(3);
            temp_u_hat_2=zeros(3);
            temp_u_hat_3=zeros(3);
            temp_u_hat_4=eye(3);
        case 12 %phase  ab
            temp_x=[1 0 0];
            temp_u1=u_2p;
            temp_u2=temp_u2_2p;
            temp_u_hat_1=u_hat_1;
            temp_u_hat_2=u_hat_2;
            temp_u_hat_3=u_hat_3;
            temp_u_hat_4=u_hat_4;
        case 23 %phase bc
            temp_x=[0 1 0];
            temp_u1=u_2p;
            temp_u2=temp_u2_2p;
            temp_u_hat_1=u_hat_1;
            temp_u_hat_2=u_hat_2;
            temp_u_hat_3=u_hat_3;
            temp_u_hat_4=u_hat_4;
        case 31 % phase ca
            temp_x=[0 0 1];
            temp_u1=u_2p;
            temp_u2=temp_u2_2p;
            temp_u_hat_1=u_hat_1;
            temp_u_hat_2=u_hat_2;
            temp_u_hat_3=u_hat_3;
            temp_u_hat_4=u_hat_4;
        case 123 % phase abc, assume measuring the phase a volt
            switch temp_3p_phase
                case 1 %3 phase measure a
                    temp_x=[1 0 0];
                case 2 %3 phase measure b
                    temp_x=[0 1 0];
                case 3 % 3 phase measure c
                    temp_x=[0 0 1];
            end
            temp_u1=eye(3);
            temp_u2=zeros(3);
            temp_u_hat_1=1/3*ones(3);
            temp_u_hat_2=zeros(3);
            temp_u_hat_3=zeros(3);
            temp_u_hat_4=1/3*ones(3);
    end 
    
    temp_idx=find(load_to_main_node(:,1)==temp_id);
    temp_main_node=load_to_main_node(temp_idx,2);
    temp_j=find(main_node_list==temp_main_node);
    
    temp_select_i=i*3-3+(1:3);
    temp_select_j=temp_j*3-3+(1:3);
    
    X=blkdiag(X,temp_x);   
    U1(temp_select_i,temp_select_j)=temp_u1;  
    U2(temp_select_i,temp_select_j)=temp_u2; 
    U_hat_1(temp_select_j,temp_select_i)=temp_u_hat_1; 
    U_hat_2(temp_select_j,temp_select_i)=temp_u_hat_2;
    U_hat_3(temp_select_j,temp_select_i)=temp_u_hat_3;
    U_hat_4(temp_select_j,temp_select_i)=temp_u_hat_4; 
end

% Pack the nodal P Q injection corrsponding to main_node_list
% temp_M=[U_hat_1,U_hat_2;-U_hat_2,U_hat_1]*blkdiag(X',X')*[app_P;app_Q];
main_nodal_P=[U_hat_1,U_hat_2]*blkdiag(X',X')*[app_P;app_Q];
main_nodal_Q=[-U_hat_2,U_hat_1]*blkdiag(X',X')*[app_P;app_Q];
temp_length=size(main_nodal_P,2);
% I1=find(node_id_list==global_source_node);
% source load has no noise.
source_idx=find(main_node_list==global_source_node);%find the index of the source node in the main_node_list, nonsource_main_node_list is the same as main_node_list except it removes the source node. 
temp_idx=(source_idx-1)*3+(1:3); % index corresponding to source 3-phase power injection.
main_nodal_P(temp_idx,:)=source_P; % insert source node injection
main_nodal_Q(temp_idx,:)=source_Q; % insert source node injection

% pack the label into label structure
% Structure of label:
% label(i).node--name of node i.
% label(i).conj--a 3-by-T matrix of nodal conjugate complex power of node i of all
% time through T.
label=[];
for i=1:length(main_node_list)
    label(i).node=main_node_list(i);
    temp_idx=(i-1)*3+(1:3);
    label(i).conj=main_nodal_P(temp_idx,:)-1i*main_nodal_Q(temp_idx,:);
    label(i).P=main_nodal_P(temp_idx,:);
    label(i).Q=main_nodal_Q(temp_idx,:);
end

%% prepare state structure
% Prepare the initial state and the correct state to compare.

% Structure of in_state:
% in_state(i).node--name of node i.
% in_state(i).mag--a 3-by-T matrix of nodal volt magnitude of node i of all
% time through T.
% in_state(i).rad--a 3-by-T matrix of nodal volt angle (in rad) of node i of
% all time through T.
state_initial=[];
temp_T=size(app_P,2); % time length

for i=1:length(main_node_list)
    target_node=main_node_list(i);
    state_initial(i).node=target_node;
    
    state_initial(i).mag=ones(3,temp_T);
    state_initial(i).rad=[0;-2*pi/3;2*pi/3]*ones(1,temp_T);    
end

% will use the complex form of the state.
state_initial_complex=f_state2complex(state_initial);

%% build output model.
% Prepare output_model:
% output_model--structure containing how the state is mapped to the output.
% Structure of output_model:
% output_model(i).meter--the name of the meter i.
% output_model(i).node--the node which the meter is connected to.
% output_model(i).phase--the phase of the smart meter i.
% 1,2,3,12,23,31,301,302,303 represents phase A, B, C, AB, BC, CA, 3-phase
% A, 3-phase B, and 3-phase C.
% output_model(i).f_v--the vector model of meter i. f_v is 1-by-3, a
% selection vetor of 1,-1, and 0 indicating which phase has which sign. f_v
% is the same for both real and imaginary part of a state of one particular
% meter.
% 2020-05-15: add the measurement at the intermediate node.

output_model=[];
u_2p=0.5*sqrt(3)*[1 1 0; 0 1 1; 1 0 1];
temp_u2_2p=0.5*[1 -1 0; 0 1 -1; -1 0 1];

for i_meter=1:length(load_id_list)
    target_meter=load_id_list(i_meter);
    output_model(i_meter).meter=target_meter;
    
    I1=find(load_to_main_node(:,1)==target_meter);
    target_node=load_to_main_node(I1,2);
    output_model(i_meter).node=target_node;
    
    I1=find(load_id_phase(:,1)==target_meter);
    target_phase=load_id_phase(I1,2);
    temp_3p_phase=load_id_phase(I1,3);
%     temp_3p_phase=1;%remember to change back!!!!
    switch target_phase % each phaes connection case
        case 1 %phase a
            output_model(i_meter).phase=target_phase;
            output_model(i_meter).f_v=[1 0 0];
            
        case 2 %phase b
            output_model(i_meter).phase=target_phase;
            output_model(i_meter).f_v=[0 1 0];
        case 3 %phase c
            output_model(i_meter).phase=target_phase;
            output_model(i_meter).f_v=[0 0 1];
         
        case 12 %phase ab
            output_model(i_meter).phase=target_phase;
            output_model(i_meter).f_v=[1 -1 0];
        
        case 23 %phase bc
            output_model(i_meter).phase=target_phase;
            output_model(i_meter).f_v=[0 1 -1];
            
        case 31 %phase ca
            output_model(i_meter).phase=target_phase;
            output_model(i_meter).f_v=[-1 0 1];
        case 123 % phase abc, assume measuring the phase a volt
            switch temp_3p_phase
                case 1 %3 phase measure a
                    output_model(i_meter).phase=301;
                    output_model(i_meter).f_v=[1 0 0];
                case 2 %3 phase measure b
                    output_model(i_meter).phase=302;
                    output_model(i_meter).f_v=[0 1 0];
                case 3 % 3 phase measure c
                    output_model(i_meter).phase=303;
                    output_model(i_meter).f_v=[0 0 1];
            end
    end
end

% add the measurement of intermediate node
i_meter=length(load_id_list); % index
for i_inter_node=1:length(inter_node)
    target_meter=inter_node(i_inter_node);
    
    for i_phase=1:3 % each inter node has 3 phase meter
        i_meter=i_meter+1;
        output_model(i_meter).meter=target_meter*10+i_phase; % define meter id as node id+phase number
        output_model(i_meter).node=target_meter;
        output_model(i_meter).phase=i_phase;
        temp_v=zeros(1,3);
        temp_v(i_phase)=1;
        output_model(i_meter).f_v=temp_v;
    end
end

% the correct output
% output--a structure of output. output(t).meter_list is a vector containing the
% list of meter names (same for all t). output(t).value is a vector of
% output value (in this case the volt measurement) matching to each meter
% at time t.
% app_V_mag_diff=diff(app_V_mag,1,2); % use time difference of meter volt measures
inter_V_mag=inter_V_mag;
T=size(app_V_mag,2); % time length of measures.
output_correct=[];

temp_meter_list=[output_model(:).meter]';
for i_t=1:T
    output_correct(i_t).meter_list=temp_meter_list;
    if ~isempty(inter_node)
        output_correct(i_t).value=[app_V_mag(:,i_t);inter_V_mag(:,i_t)];
    else
        output_correct(i_t).value=[app_V_mag(:,i_t)];
    end
end

T_length=length(output_correct); % the length in time of actual smart meter data (before taking time difference).
T_diff_mat_full=[(1:T_length-1)',(2:T_length)']; % time difference index table full: time difference is column 2's time data minus column 1's. 
output_correct_diff=f_output2diff(output_correct,T_diff_mat_full);

%% Build d_GB_d_w
% d_GB_d_w--a structure storing the fixed derivative matrices for line
% parameter.d_GB_d_w(i).M is a 3-by-3 matrix. i=1~6, representing 6 phase
% types: aa, ab, ac, bb, bc, cc. G and B share the same patter, so we only 
% i=1~6 to represent these cases.
d_GB_d_w=[];
for i=1:6
    temp_M=zeros(3);
    switch i
        case 1 % phase aa
            temp_M(1,1)=1;
        case 2 % phase ab
            temp_M(1,2)=1;
            temp_M(2,1)=1;
        case 3 % phase ac
            temp_M(1,3)=1;
            temp_M(3,1)=1;
        case 4 % phase bb
            temp_M(2,2)=1;
        case 5 % phase bc
            temp_M(2,3)=1;
            temp_M(3,2)=1;
        case 6 % phase cc
            temp_M(3,3)=1;
    end
    d_GB_d_w(i).M=temp_M;
end

%% prepare data for global estimation setup

all_source_state(1).node=global_source_node;
all_source_state(1).mag=source_V_mag;
all_source_state(1).rad=source_V_rad;
for i=2:length(all_source_node_list) % accurate data of all feeder head and PMU
    temp_idx=(1:3)+(i-2)*3;
    all_source_state(i).node=all_source_node_list(i);
    all_source_state(i).mag=PMU_V_mag(temp_idx,:);
    all_source_state(i).rad=PMU_V_rad(temp_idx,:);
end
all_source_state_complex=f_state2complex(all_source_state);



% TUNABLE HERE.
w_start=Input_struct.w_start;
w_iter=w_start;


%%% prepare the range limit
% TUNABLE HERE.
if strcmp(limit_range_txt,'yes')
    [w_min,w_max]=f_w_generate_limit(w_start,error_ratio);
elseif strcmp(limit_range_txt,'no')
    [w_min,w_max]=f_w_generate_inf_limit(w_start);
else
    disp('RANGE LIMIT USAGE NOT CORRECTLY DFINED')
    return
end

%%% Generate structures that store the variance and mean value of each 
%%% parameter. The mean should be the initial parameter value, the sigma 
%%% should be 1/3 of the parameter’s error range value.
prior_dist=f_generate_prior_dist(w_start,error_ratio);

%% Separate the netowrk into sub-networks.
%%% use matlab's graph functions, first use index number to indicate each
%%% node.
line_node_list=unique([main_line_config(:,1);main_line_config(:,2)]);
% start building the line configuration of break-up network
[Lia,main_line_sep_idx_table]=ismember(main_line_config(:,1:2),line_node_list); % represent config table in terms of indices.
[Lia,partition_table_idx]=ismember(partition_table,line_node_list); % represent partition table in indices
for i=1:size(partition_table_idx,1) %remove cutting edge from the full connection table
    temp_idx_1=partition_table_idx(i,1);
    temp_idx_2=partition_table_idx(i,2);
    I1=find(main_line_sep_idx_table(:,1)==temp_idx_1 & main_line_sep_idx_table(:,2)==temp_idx_2);
    I2=find(main_line_sep_idx_table(:,1)==temp_idx_2 & main_line_sep_idx_table(:,2)==temp_idx_1); % consider the swaping of indices
    main_line_sep_idx_table([I1;I2],:)=[]; % remove i-th cutting edge.
end

temp_G=graph(main_line_sep_idx_table(:,1),main_line_sep_idx_table(:,2),[],length(line_node_list));
bins=conncomp(temp_G); % the bins number represents different groups of connected nodes

%%% Now build sub-networks, each cell contains the line config table of a
%%% sub-net.
sub_net_struct=[];
for i=1:max(bins) % each loop organize the sub-net of the i-th bin
    I1=find(bins==i); 
    Lia=ismember(main_line_sep_idx_table,I1);
    temp_net_idx=main_line_sep_idx_table(Lia(:,1)&Lia(:,2),:);
    sub_net_struct(i).line_config=line_node_list(temp_net_idx); % store the sub-network in terms actual node number.
end

% Add pesudo-substation to the downstream subnetwork
for i=1:length(sub_net_struct)
    for i2=1:size(partition_table,1) % check if the cutting edge's downstream connects the sub-net
        temp_line=partition_table(i2,:);
        Lia=ismember(temp_line(2),sub_net_struct(i).line_config);
        if Lia % if the downstream node is in the sub-net
            sub_net_struct(i).line_config=[sub_net_struct(i).line_config;temp_line];
        end
    end
end

%% Prepare the data and parameters for each sub-network.

%%% define parameter set first, some values are the same for all sub-nets,
%%% define here first:
% para_set.weight_seed=weight_seed;
para_set.batch_size=batch_size;
para_set.max_iter=max_iter;
para_set.threshold_ratio=threshold_ratio;
para_set.early_stop_patience=early_stop_patience; % number of epochs of no improvement, after which stop the iteration.
para_set.early_stop_threshold_ratio=early_stop_threshold_ratio; % decide improvement: at least early_stop_threshold_ratio*initial obj function value.
para_set.prior_adjust_ratio=prior_adjust_ratio; % a ratio to adjust the regularization factor.
% Backtracking linesearch parameter
para_set.initial_step_size=initial_step_size; % standard step size
para_set.alpha=alpha; % Backtracking line search parameter.
para_set.beta=beta; % Backtracking line search parameter.
% Step size adjusting parameter
para_set.step_dynamic_beta=step_dynamic_beta; % parameter to decide dynamic initial step size
para_set.min_step_threshold=min_step_threshold;% minimum initial step size when using dynamic initial step size
para_set.max_step_threshold=max_step_threshold;
% para_set.primal_var=primal_var;
para_set.folder_dir=folder_dir; % folder name to save the result
para_set.description=description_txt;
para_set.limit_range_txt=limit_range_txt;
if exist('dir_basic_result','var')% if the dir_basic_result exists
    para_set.dir_basic_result=dir_basic_result;
end

% global parameter setup define here
global_para_set=para_set;
global_para_set.w_start=w_start;
global_para_set.w_z_correct=w_z_correct;
global_para_set.prior_dist=prior_dist;
global_para_set.limit_range.w_max=w_max;
global_para_set.limit_range.w_min=w_min;
global_para_set.sub_net_struct=sub_net_struct;

% para_set_array_by_subnet=[]; % each subnet has the same para_set
% data_set_array_by_subnet=struct;
for i_net=1:length(sub_net_struct)
    temp_line_config=sub_net_struct(i_net).line_config;
    temp_subnet_node_list=unique([temp_line_config(:,1);temp_line_config(:,2)]);
    temp_para_set=para_set;
    temp_data_set=[];
    
    %%% continue building sub-net para_set
    temp_para_set.line_config=temp_line_config;
    temp_para_set.subnet_id=i_net;
    temp_para_set.w_start=f_build_subnet_w(w_start,temp_line_config);
    temp_para_set.w_correct=f_build_subnet_w(w_z_correct,temp_line_config);
    temp_para_set.w_min=f_build_subnet_w(w_min,temp_line_config);
    temp_para_set.w_max=f_build_subnet_w(w_max,temp_line_config);
    % the para set of each subnet
    para_set_array_by_subnet(i_net)=temp_para_set;
    
    %%% build sub-net data_set's label
    % build label
    temp_label=f_filter_subnet_label(label,temp_subnet_node_list); % filter to keep only the label of nodes in the subnet
    temp_label_node_list=[temp_label.node]';
    Lia=ismember(partition_table,temp_line_config);
%     I1=find(Lia(:,1)&Lia(:,2)); % I1 indicates the cutting edge's both nodes are in this subnet
    if ~isempty(Lia)
        I1=find(Lia(:,1)&Lia(:,2)); % I1 indicates the cutting edge's both nodes are in this subnet
    else
        I1=[];
    end
    temp_partition=partition_table(I1,:);
    for i_cut=1:size(temp_partition,1) % adjust label for each pseudo substation for this downstream subnet
        temp_node_1=temp_partition(i_cut,1); 
        temp_node_2=temp_partition(i_cut,2);
        temp_txt=[num2str(temp_node_1),'_',num2str(temp_node_2)]; %the corresponding PMU id
%         I1=find(cell2mat(cellfun(@(x) strcmp(x,temp_txt),PMU_id_list,'UniformOutput',false))); % find the pseudo substation's corresponding source load
        I1=find(partition_table(:,1)==temp_node_1 & partition_table(:,2)==temp_node_2); % find the pseudo substation's corresponding source load
        temp_idx=(I1-1)*3+(1:3);
        temp_P=app_PMU_P(temp_idx,:);
        temp_Q=app_PMU_Q(temp_idx,:);
        I2=find(temp_label_node_list==temp_node_1); % find the label index of the pseudo substation
        temp_label(I2).P=temp_P; % adjust the label value (the pesodo-source's power injection)
        temp_label(I2).Q=temp_Q;
        temp_label(I2).conj=temp_P-1i*temp_Q;
    end
    if ~isempty(Lia)
        I1=find(Lia(:,1)&~Lia(:,2)); % I1 indicates the cutting edges has only upstream node in this subnet
    else
        I1=[];
    end
%     I1=find(Lia(:,1)&~Lia(:,2)); % I1 indicates the cutting edges has only upstream node in this subnet
    temp_partition=partition_table(I1,:);
    for i_cut=1:size(temp_partition,1) % adjust label for each pseudo substation for this upstream subnet
        temp_node_1=temp_partition(i_cut,1); 
        temp_node_2=temp_partition(i_cut,2);
        temp_txt=[num2str(temp_node_1),'_',num2str(temp_node_2)]; %the corresponding PMU id
%         I1=find(cell2mat(cellfun(@(x) strcmp(x,temp_txt),PMU_id_list,'UniformOutput',false))); % find the pseudo substation's corresponding source load
        I1=find(partition_table(:,1)==temp_node_1 & partition_table(:,2)==temp_node_2); % find the pseudo substation's corresponding source load
        temp_idx=(I1-1)*3+(1:3);
        temp_P=app_PMU_P(temp_idx,:);
        temp_Q=app_PMU_Q(temp_idx,:);
        I2=find(temp_label_node_list==temp_node_1); % find the label index of the pseudo substation
        % pay attention to the sign below, for upstream subnet, the
        % substation absorbs the power.
        temp_label(I2).P=temp_label(I2).P-temp_P; % adjust the label value (the pesodo-source's power injection)
        temp_label(I2).Q=temp_label(I2).Q-temp_Q;
        temp_label(I2).conj=temp_label(I2).conj-temp_P+1i*temp_Q;
    end
    temp_data_set.label=temp_label; % label is build.
    
    % build subnet's state
    temp_data_set.state_initial_complex=f_build_subnet_complex_state(state_initial_complex,temp_subnet_node_list);
    % build source node
    Lia=ismember(all_source_node_list,temp_subnet_node_list);
    temp_node_list=all_source_node_list(Lia); % the souce nodes in this subnet
    % only keep one node as source.
    I1=find(temp_node_list==global_source_node);
    temp_PMU_list=temp_node_list;
    if ~isempty(I1)
        temp_source_node=global_source_node;
        temp_PMU_list(I1)=[];
    else
        temp_source_node=temp_node_list(1);
        temp_PMU_list(1)=[];
    end
%     [Lia,Locb]=ismember(temp_node_list,[state_measure_complex.node]');
%     temp_data_set.source_state_complex=state_measure_complex(Locb);
    [Lia,Locb]=ismember(temp_source_node,[all_source_state_complex.node]');
    temp_data_set.source_state_complex=all_source_state_complex(Locb);
    % build ouput model and correct output
    temp_data_set.output_model=f_build_subnet_output_model(output_model,temp_subnet_node_list,temp_source_node);
    temp_data_set.T_diff_mat_full=T_diff_mat_full;
    temp_data_set.output_correct_diff=f_build_subnet_output(output_correct_diff,temp_data_set.output_model);
    temp_data_set.output_correct=f_build_subnet_output(output_correct,temp_data_set.output_model);
    % build prior
    temp_data_set.prior_dist=f_build_subnet_prior_dist(prior_dist,temp_line_config);
    % other data set values
    temp_data_set.d_GB_d_w=d_GB_d_w;
    
    % The data set of each subnet
    data_set_array_by_subnet(i_net)=temp_data_set;
    
end



%% Prepare the data for each sub-network's case of one algorithm seed.
% This is different from data/parameter set by subnet, because each subnet
% may have more than one algorithm seed, and each case will have a
% different save file name.
subnet_list=1:length(sub_net_struct);
Test_case=[];% this is the test case, containing the circuit id and alg_seed
for i_alg=1:length(alg_seed_list)
    for i_net=1:length(subnet_list)
        Test_case=[Test_case;i_net,alg_seed_list(i_alg)]; % col 1--net id, col 2--alg_seed
    end
end

global_para_set.alg_seed_list=alg_seed_list;

save([folder_dir,'/global_setup_save'],'global_para_set');

%% (parallel) simulation of parameter estimation in each case of subnet-alg_seed combination 
% To do parallel computing, use parfor.
parfor (i_sim=1:size(Test_case,1),Max_para_num)
% for i_sim=3:3
    i_net=Test_case(i_sim,1);
    alg_seed=Test_case(i_sim,2);
    case_para_set=para_set_array_by_subnet(i_net);
    case_para_set.alg_seed=alg_seed;
    case_para_set.file_dir=['subnet-',num2str(i_net),'_algseed-',num2str(alg_seed)];
    [save_result]=f_subnet_est_NL_GNNdiff_prlm(case_para_set,data_set_array_by_subnet(i_net));
end

%% This is the end of the main code.





%% Define functions for the gradient method
%%% function that generate a state only on the batched time stamps.
function out_state_batch=f_state_batch(in_state,idx)
% Input:
% in_state--input state structure containing the nodal volt magnitude and
% angle. In this problem, it is the initialized state.
% idx--index of selected time stamps.
% Output:
% out_state_batch--same kind of structure as in_state, but only contains
% time of indexed timestampls.

%%% Details of the structures---------------------------------
% Structure of in_state:
% in_state(i).node--name of node i.
% in_state(i).mag--a 3-by-T matrix of nodal volt magnitude of node i of all
% time through T.
% in_state(i).rad--a 3-by-T matrix of nodal volt angle (in rad) of node i of
% all time through T.

out_state_batch=in_state;
for i=1:length(in_state)
    out_state_batch(i).real=out_state_batch(i).real(:,idx);
    out_state_batch(i).imag=out_state_batch(i).imag(:,idx);
end

end

%%% function that generate a label only on the batched time stamps.
function out_label_batch=f_label_batch(in_label,idx)
% Input:
% label--label structure containing the nodal real and reactive power
% injection.
% idx--index of selected time stamps.
% Output:
% out_label_batch--same kind of structure as label, but only contains
% time of indexed timestampls.

%%% Details of the structures---------------------------------
% Structure of label:
% label(i).node--name of node i.
% label(i).conj--a 3-by-T matrix of nodal conjugate complex power of node i of all
% time through T.

% out_label_batch=in_label;
for i=1:length(in_label)
    out_label_batch(i).node=in_label(i).node;
    out_label_batch(i).conj=in_label(i).conj(:,idx);
    out_label_batch(i).P=in_label(i).P(:,idx);
    out_label_batch(i).Q=in_label(i).Q(:,idx);
end

end

%%% function to update parameter value
function w_out=f_update_w(w_in,w_gradient,step_size)
% Input:
% w_in--the weight/parameter. In our problem this is the set of line
% parameters. The structure is the same as w.
% w_gradient--the gradient structure representing the gradient of the
% weight, has the same structure as w.
% step_size--the step size. Note that in gradient descent, the step size
% input here should be a negative value. The step size is a multiplier to
% multiply the gradient.

% Output:
% w_out--the new weight structure updated by the step size.

% Structure of w:
% w(i).line--a 2-element vector containing the vertices of the line. Length
% of w is the number of lines.
% w(i).g--a vector of conductance of line i, contain 6 elements.
% w(i).b--a vector of susceptance of line i, contain 6 elements.
% The 6 elements are in the order of phase aa, ab, ac, bb, bc, cc.

w_out=w_in;
for i=1:length(w_in)
    w_out(i).g=w_out(i).g+w_gradient(i).g*step_size;
    w_out(i).b=w_out(i).b+w_gradient(i).b*step_size;
end

end

%%% function to reorganize a weight strucutre to a vector
function w_vec=f_w_struct2vec(w_in)
% Input:
% w_in--the weight/parameter. In our problem this is the set of line
% parameters. The structure is the same as w.

% Output:
% w_vec--a vector that packs the structure values.

% Structure of w:
% w(i).line--a 2-element vector containing the vertices of the line. Length
% of w is the number of lines.
% w(i).g--a vector of conductance of line i, contain 6 elements.
% w(i).b--a vector of susceptance of line i, contain 6 elements.
% The 6 elements are in the order of phase aa, ab, ac, bb, bc, cc.

w_vec=[];
for i=1:length(w_in) % for each line, pack g 1st, b 2nd.
    w_vec=[w_vec;w_in(i).g];
    w_vec=[w_vec;w_in(i).b];
end
end

%%% function to calculate MADR of two sets of parameters
function MADR=f_MADR(vec_est,vec_true)
vec_diff=vec_est-vec_true;
MADR=sum(abs(vec_diff))/sum(abs(vec_true))*100;
end

%%% function to calculate the sign of weight or weight gradient structure
function w_sign=f_weight_sign(w_in)
% Input:
% w_in--the weight/parameter. In our problem this is the set of line
% parameters. The structure is the same as w.

% Output:
% w_sign--a structure same as w_in, but only store the sign of the w_in.

% Structure of w:
% w(i).line--a 2-element vector containing the vertices of the line. Length
% of w is the number of lines.
% w(i).g--a vector of conductance of line i, contain 6 elements.
% w(i).b--a vector of susceptance of line i, contain 6 elements.
% The 6 elements are in the order of phase aa, ab, ac, bb, bc, cc.

w_sign=w_in;
temp_x=1e-16; % a term for equilibrium values
for i=1:length(w_sign)
    w_sign(i).g=sign(w_in(i).g)+temp_x;
    w_sign(i).b=sign(w_in(i).b)+temp_x;
end
end

%%% function to update the step size based on Rprop rule
function step_new=f_Rprop_update_step(step_old,grad_sign_old,grad_sign_new,eta_plus,eta_minus)
% Input:
% step_old--old step structure, structure same as w.
% grad_sign_old--old sign structure. structure same as w.
% grad_sign_new--new sign structure. structure same as w.
% eta_plus--parameter for Rprop.
% eta_minus--parameter for Rprop.

% Output:
% step new--new step structure. structure same as w.

% Structure of w:
% w(i).line--a 2-element vector containing the vertices of the line. Length
% of w is the number of lines.
% w(i).g--a vector of conductance of line i, contain 6 elements.
% w(i).b--a vector of susceptance of line i, contain 6 elements.
% The 6 elements are in the order of phase aa, ab, ac, bb, bc, cc.

step_new=step_old;
for i=1:length(step_old)
    sign_g=(grad_sign_old(i).g).*(grad_sign_new(i).g);
    sign_b=(grad_sign_old(i).b).*(grad_sign_new(i).b);
    
    for i_2=1:length(sign_g)
        if sign_g(i_2)>=0
            step_new(i).g(i_2)=step_old(i).g(i_2)*eta_plus;
        else
            step_new(i).g(i_2)=step_old(i).g(i_2)*eta_minus;
        end
    end
    
    for i_2=1:length(sign_b)
        if sign_b(i_2)>=0
            step_new(i).b(i_2)=step_old(i).b(i_2)*eta_plus;
        else
            step_new(i).b(i_2)=step_old(i).b(i_2)*eta_minus;
        end
    end
    
end
end

%%% function to update weight using Rprop
function w_new=f_Rprop_update_weight(w_old,step_w,grad_sign,direction)
% Input:
% w_old--old weight, same structure as w.
% step_w--step size, same structure as w.
% grad_sign--sign of gradient, same structure as w.
% direction--+1 for gradient ascending, -1 for descending.

% Output:
% w_new--updated weight, same structure as w.

% Structure of w:
% w(i).line--a 2-element vector containing the vertices of the line. Length
% of w is the number of lines.
% w(i).g--a vector of conductance of line i, contain 6 elements.
% w(i).b--a vector of susceptance of line i, contain 6 elements.
% The 6 elements are in the order of phase aa, ab, ac, bb, bc, cc.

w_new=w_old;
for i=1:length(w_new)
    w_new(i).g=w_old(i).g+(grad_sign(i).g).*(step_w(i).g)*direction;
    w_new(i).b=w_old(i).b+(grad_sign(i).b).*(step_w(i).b)*direction;
end

end

%% Define functions for forward and other functions.

%%% calculate the primitive admittance matrix from a line's paramters
function Y=f_Ynk_Matrix(g,b)
% Input:
% g--a vector of conductance of a line, contain 6 elements.
% b--a vector of conductance of a line, contain 6 elements.
% the elements are in the order of phase aa, ab, ac, bb, bc, cc

% Output:
% Y-- a complex primitive Y matrix, it is 3-by-3.

m=g;
Lambda_g=[m(1),m(2),m(3); ...
    m(2),m(4),m(5); ...
    m(3),m(5),m(6)];

m=b;
Lambda_b=[m(1),m(2),m(3); ...
    m(2),m(4),m(5); ...
    m(3),m(5),m(6)];

Y=-Lambda_g-1i*Lambda_b;
end

%%% function to calculate rotation matrix for L_num lines (designed 2020-03-30)
function R=Rotate_phi_fun(L_num,power_sign)
% input: 
% L_num: number of lines.
% power_sign: +1 or -1
% output:
% R: rotation matrix used to transform the G,B to linearized power flow
% model.

if power_sign==1
    v1=zeros(1,L_num);
    v2=-2*pi/3*ones(1,L_num);
    v3=2*pi/3*ones(1,L_num);
elseif power_sign==-1
    v1=zeros(1,L_num);
    v2=2*pi/3*ones(1,L_num);
    v3=-2*pi/3*ones(1,L_num);
else
    disp('wrong power_sign')
    return
end
angle_v=[v1,v2,v3];
M_cos=diag(cos(angle_v));
M_sin=diag(sin(angle_v));
R=[M_cos,M_sin;-M_sin,M_cos];
end

%%% function to calculate the MSE of the differences between two state
%%% structures.
function MSE=f_state_update_MSE(new_state,old_state)
% Input:
% new_state--input state structure containing the nodal volt magnitude and
% angle.
% old_state--another state structure to compare with, the same structure as
% new_state.

% Output:
% MSE--the mean square error.

% Structure of new_state:
% structure containing the nodal volt magnitude and
% angle. Organized in a 6N x T big matrix. Each 6 rows represent a node's
% volt magnitude and angle. It contains 2 field, the node_list, and magrad_mat.

T=size(new_state.magrad_mat,2); % number of timestamps
N_6=size(new_state.magrad_mat,1);
temp_sum=norm(new_state.magrad_mat-old_state.magrad_mat,'fro')^2;
MSE=temp_sum/(T*N_6);
end

%%% function to calculate the Mean square value of the state structures.
function MSE=f_state_MSE(in_state)
% Input:
% in_state--input state structure containing the nodal volt magnitude and
% angle.

% Output:
% MSE--the mean square value.

% Structure of in_state:
% structure containing the nodal volt magnitude and
% angle. Organized in a 6N x T big matrix. Each 6 rows represent a node's
% volt magnitude and angle. It contains 2 field, the node_list, and magrad_mat.

T=size(in_state.magrad_mat,2); % number of timestamps
N_6=size(in_state.magrad_mat,1); % number of timestamps
temp_sum=norm(in_state.magrad_mat,'fro')^2;
MSE=temp_sum/(T*N_6);
end

%%% A global transistion function, which is a stack of all local transition
%%% function
function out_state_compact=f_global_transition(in_state_compact,Model,label)
% Input:
% in_state_compact--input state structure containing the nodal volt real
% and imaginary part. Of compact state structure
% Model--the structure that stores the needed A1 matrix and A2 matrics for 
% each node's local transition function.
% label--label structure containing the nodal real and reactive power
% injection.

% Output:
% out_state_compact--output state structure containing the nodal volt real
% and imaginary part. Of compact state structure.

% Structur of the state_compact
% state_compact.node_list--list of node (N node)
% state_compact.scalor_struct= a N-by-1 structure array, 
% state_compact.scalor_struct(i).scalor is a 6-by-T matrix of nodal volt real 
% and imaginary part of node i of all time through T.
% state_compact.complex_struct= a N-by-1 structure array, 
% state_compact.conj_struct(i).conj is a 3-by-T matrix of nodal conjugate complex volt
% of node i of all time through T.

% Structure of the Model:
% Model(i).node--node name of node i.
% Model(i).neighbor--vector of node i's neighbors.
% Model(i).rZ_nn--Z_nn real matrix for node i's local transition function
% (6-by-6).
% Model(i).rY_nk--structure of Y_nk real matrix for node i's local transition function.
% Model(i).rY_nk(j).M--the Y_nk real matrix corresponding to node ji's
% neighbor node j (6-by-6).

% Structure of label:
% label(i).node--name of node i.
% label(i).P--a 3-by-T matrix of nodal real power of node i of all
% time through T.
% label(i).Q--a 3-by-T matrix of nodal reactive power of node i of
% all time through T.

% Structure of out_state:
% out_state has the same structure as in_state.
%---------------------------------------------------------------

out_state_compact=in_state_compact;
state_node_list=in_state_compact.node_list; %list of node names in in_state.
Model_node_list=[Model.node]; % list of node names in Model.
label_node_list=[label.node]; % list of node names in label.

for i=1:length(state_node_list)
    target_node=state_node_list(i);
    model_idx=find(Model_node_list==target_node); % find the local model for the target node.
    label_idx=find(label_node_list==target_node); % find the local model for the target node.
    % This is Gauss iteration.
    target_out_state=f_local_transition(in_state_compact,Model(model_idx),label(label_idx));% update the target node's state.
    % below try using Gauss-Seidel iteration (result not good, abandon)
%     target_out_state=f_local_transition(out_state_compact,Model(model_idx),label(label_idx));% update the target node's state.
    out_state_compact.scalor_struct(i).scalor=target_out_state.scalor; % update the state of node i.
    out_state_compact.conj_struct(i).conj=target_out_state.conj; % update the state of node i.
end

end

%%% A local transistion function
function out_state_local_complex=f_local_transition(in_state_compact,Model_local,label_local)
% Input:
% in_state_compact--input state structure containing the nodal volt real
% and imaginary part. Of compact structure.
% Model_local--the structure that stores the needed rZ_nn matrix and rY_nk
% matrices for the one target node.
% label_local--label of the local node, i.e., the nodal real and reactive
% power.

% Output:
% out_state_local_complex--output state structure containing the nodal volt 
% real and reactive part of the target node.

% Strucutre of in_state_complex:
% complex_state.node--name of node.
% complex_state.real--a 3-by-T matrix of nodal volt real part of node of all
% time through T.
% complex_state.imag-a 3-by-T matrix of nodal volt imaginary part of node of
% all time through T.

% Structure of the Model_local:
% Model_local.node--node name of node.
% Model_local.neighbor--vector of node's neighbors.
% Model_local.rZ_nn--Z_nn real matrix for node's local transition
% function (6-by-6).
% Model_local.rY_nk--structure of real Y_nk matrix for node's local transition function.
% Model_local.rY_nk(j).M--the rY_nk matrix corresponding to node's neighbor node
% j (matching j-th neighbor in Model_local.neighbor.)

% Structure of label_local:
% label_local.node--name of a node.
% label_local.P--a 3-by-T matrix of nodal real power of a node of all
% time through T.
% label_local.Q--a 3-by-T matrix of nodal reactive power of a node of
% all time through T.

% Structure of out_state:
% out_state has the same structure as in_state, but contains only one node.

% Structur of the state_compact
% state_compact.node_list--list of node (N node)
% state_compact.scalor_struct= a N-by-1 structure array, 
% state_compact.scalor_struct(i).scalor is a 6-by-T matrix of nodal volt real 
% and imaginary part of node i of all time through T.
% state_compact.complex_struct= a N-by-1 structure array, 
% state_compact.conj_struct(i).conj is a 3-by-T matrix of nodal conjugate complex volt
% of node i of all time through T.
%--------------------------------------------------------------

state_node_list=in_state_compact.node_list; %list of node names in in_state.
local_node=label_local.node;
I1=find(state_node_list==local_node);
temp_V=in_state_compact.conj_struct(I1).conj;%conjugate of local node volt

temp_S=label_local.conj; % conjugate of power injection
temp_M=temp_S./temp_V;
temp_output=[real(temp_M);imag(temp_M)];

for i=1:length(Model_local.neighbor)
    temp_neighbor=Model_local.neighbor(i);% current neighbor
%     rY_nk=Model_local.rY_nk(i).M;
    I1=find(state_node_list==temp_neighbor);
%     temp_v=[in_state_compact.real_struct(I1).real;in_state_compact.imag_struct(I1).imag];
    temp_output=temp_output-Model_local.rY_nk(i).M*in_state_compact.scalor_struct(I1).scalor; % update neighbor's contribution
end
% rZ_nn=Model_local.rZ_nn;
temp_output=Model_local.rZ_nn*temp_output;

out_state_local_complex.node=Model_local.node;
out_state_local_complex.scalor=temp_output;
out_state_local_complex.conj=temp_output(1:3,:)-1i*temp_output(4:6,:);
end

%%% function to transfer complex matrix to real multiplication matrix. This
%%% real multiplication matrix represent a complex multiplication in forms
%%% of real multiplication
function real_M=f_complex2realmulti(complex_M)
% Input:
% complex_M--a complex matrix of size M-by-N.
% Output:
% real_M--a real matrxi of szie 2M-by-2N.
r_M=real(complex_M);
i_M=imag(complex_M);
real_M=[r_M,-i_M;i_M,r_M];
end


%%% function to transfer standard state structure (volt magnitude and
%%% angle) to complex state structure (real and reactive part)
function complex_state=f_state2complex(in_state)
% Input:
% in_state--input state structure containing the nodal volt magnitude and
% angle.

% Output:
% complex_state--output state structure containing the nodal volt in
% complex form--real and imaginary part.

% Structure of in_state:
% in_state(i).node--name of node i.
% in_state(i).mag--a 3-by-T matrix of nodal volt magnitude of node i of all
% time through T.
% in_state(i).rad--a 3-by-T matrix of nodal volt angle (in rad) of node i of
% all time through T.

% Strucutre of complex state
% complex_state(i).node--name of node i.
% complex_state(i).real--a 3-by-T matrix of nodal volt real part of node i of all
% time through T.
% complex_state(i).imag-a 3-by-T matrix of nodal volt imaginary part of node i of
% all time through T.

complex_state=[];
for i=1:length(in_state)
    complex_state(i).node=in_state(i).node;
    temp_x=in_state(i).mag.*exp(1i*in_state(i).rad);
    complex_state(i).real=real(temp_x);
    complex_state(i).imag=imag(temp_x);
end

end

%%% function to transfer complex state structure (real and reactive part) to
%%% standard state structure (volt magnitude and angle).
function out_state=f_complex2state(complex_state)
% Input:
% complex_state--input state structure containing the nodal volt in
% complex form--real and imaginary part.

% Output:
% out_state--output state structure containing the nodal volt magnitude and
% angle.

% Structure of out_state:
% in_state(i).node--name of node i.
% in_state(i).mag--a 3-by-T matrix of nodal volt magnitude of node i of all
% time through T.
% in_state(i).rad--a 3-by-T matrix of nodal volt angle (in rad) of node i of
% all time through T.

% Strucutre of complex state
% complex_state(i).node--name of node i.
% complex_state(i).real--a 3-by-T matrix of nodal volt real part of node i of all
% time through T.
% complex_state(i).imag-a 3-by-T matrix of nodal volt imaginary part of node i of
% all time through T.

out_state=[];
for i=1:length(complex_state)
    out_state(i).node=complex_state(i).node;
    temp_x=complex_state(i).real+1i*complex_state(i).imag;
    out_state(i).mag=abs(temp_x);
    out_state(i).rad=angle(temp_x);
end

end

%% Define functions that are added to calcualte gradient.

%%% Output funciton of the GNN
function output=f_GNN_output(in_state,output_model)
% Input:
% in_state--input state structure containing the nodal volt real and
% reactive part.
% output_model--structure containing how the state is mapped to the output.

% Output:
% output--a structure of output. output(t).meter_list is a vector containing the
% list of meter names (same for all t). output(t).value is a vector of
% output value (in this case the volt measurement) matching to each meter
% at time t.


% Strucutre of in_state or complex state
% complex_state(i).node--name of node i.
% complex_state(i).real--a 3-by-T matrix of nodal volt real part of node i of all
% time through T.
% complex_state(i).imag-a 3-by-T matrix of nodal volt imaginary part of node i of
% all time through T.

% Structure of output_model:
% output_model(i).meter--the name of the meter i.
% output_model(i).node--the node which the meter is connected to.
% output_model(i).phase--the phase of the smart meter i.
% 1,2,3,12,23,31,301,302,303 represents phase A, B, C, AB, BC, CA, 3-phase
% A, 3-phase B, and 3-phase C.
% output_model(i).f_v--the vector model of meter i. f_v is 1-by-3, a
% selection vetor of 1,-1, and 0 indicating which phase has which sign. f_v
% is the same for both real and imaginary part of a state of one particular
% meter.
%---------------------------------------------------------

node_list_in_state=[in_state.node]; % list of nodes in the in_state.
T=size(in_state(1).real,2); % time length of the states.
output_meter_list=[output_model.meter]'; % meter list in the output_model

temp_data_store=[];% a temp matrix to store data

for i_meter=1:length(output_model)
%     target_meter=output_model(i_meter).meter;
    target_node=output_model(i_meter).node;
    target_f_v=output_model(i_meter).f_v;
    
    node_idx=find(node_list_in_state==target_node);
    temp_v1=target_f_v*in_state(node_idx).real;
    temp_v2=target_f_v*in_state(node_idx).imag;
    temp_v=sqrt(temp_v1.^2+temp_v2.^2); % the reading time series.   
    temp_data_store=[temp_data_store;temp_v];
end

for i_t=1:T
    output(i_t).meter_list=output_meter_list;
    output(i_t).value=temp_data_store(:,i_t);
end

end

%%% Old Output funciton of the GNN for the previous linear power flow
%%% model.
function output=f_GNN_linear_output(in_state,output_model)
% Input:
% in_state--input state structure containing the nodal volt magnitude and
% angle.
% output_model--structure containing how the state is mapped to the output.

% Output:
% output--a structure of output. output(t).meter_list is a vector containing the
% list of meter names (same for all t). output(t).value is a vector of
% output value (in this case the volt measurement) matching to each meter
% at time t.


% Structure of in_state:
% in_state(i).node--name of node i.
% in_state(i).mag--a 3-by-T matrix of nodal volt magnitude of node i of all
% time through T.
% in_state(i).rad--a 3-by-T matrix of nodal volt angle (in rad) of node i of
% all time through T.

% Structure of output_model:
% output_model(i).meter--the name of the meter i.
% output_model(i).node--the node which the meter is connected to.
% output_model(i).phase--the phase of the smart meter i.
% 1,2,3,12,23,31,301,302,303 represents phase A, B, C, AB, BC, CA, 3-phase
% A, 3-phase B, and 3-phase C.
% output_model(i).f_v--the vector model of meter i. f_v times the state of
% the corresponding node is equal to the output of meter i.
%---------------------------------------------------------

node_list_in_state=[in_state.node]; % list of nodes in the in_state.
T=size(in_state(1).mag,2); % time length of the states.
output_meter_list=[output_model.meter]'; % meter list in the output_model

temp_data_store=[];% a temp matrix to store data

for i_meter=1:length(output_model)
%     target_meter=output_model(i_meter).meter;
    target_node=output_model(i_meter).node;
    target_f_v=output_model(i_meter).f_v;
    
    node_idx=find(node_list_in_state==target_node);
    temp_v=target_f_v*[in_state(node_idx).mag;in_state(node_idx).rad]; % the reading time series.   
    temp_data_store=[temp_data_store;temp_v];
end

for i_t=1:T
    output(i_t).meter_list=output_meter_list;
    output(i_t).value=temp_data_store(:,i_t);
end

end

%%% function to calculate A(t) for gradient
function grad_A=f_grad_A(in_state,label,Model,source_node)
% Input
% in_state--input state structure containing the nodal volt real and
% imaginary part.
% label--label structure containing the nodal real and reactive power
% injection.
% Model--the structure that stores the needed Z_nn matrix and Y_nk matrics for 
% each node's local transition function.
% source_node--the source node

% Output:
% grad_A--a structure containg (d F_w(x(t),l(t)))/d x(t).
% grad_A(t).node_list is the list of nodes (same for all t). grad_A(t).M is
% a 6N-by-6N matrix (N the number of nonsource nodes) at time t.

% Strucutre of in_state or complex state
% complex_state(i).node--name of node i.
% complex_state(i).real--a 3-by-T matrix of nodal volt real part of node i of all
% time through T.
% complex_state(i).imag-a 3-by-T matrix of nodal volt imaginary part of node i of
% all time through T.

% Structure of label:
% label(i).node--name of node i.
% label(i).P--a 3-by-T matrix of nodal real power of node i of all
% time through T.
% label(i).Q--a 3-by-T matrix of nodal reactive power of node i of
% all time through T.

% Structure of the Model:
% Model(i).node--node name of node i.
% Model(i).neighbor--vector of node i's neighbors.
% Model(i).rZ_nn--Z_nn real matrix for node i's local transition function
% (6-by-6).
% Model(i).rY_nk--structure of Y_nk real matrix for node i's local transition function.
% Model(i).rY_nk(j).M--the Y_nk real matrix corresponding to node i's
% neighbor node j (6-by-6).

% The columns and rows of grad_A(t).M:
% The index of columns and rows are the same. 1~6 represent 3 nodal real
% and 3 nodal imaginary part of node 1; then the next 6 indices represent the states
% of node 2. And so on.
%-------------------------------------------------

Model_node_list=[Model.node]; % list of node in the Model structure
label_node_list=[label.node]; % list of node in the label structure
in_state_node_list=[in_state.node]; % list of node
I1=find(in_state_node_list==source_node);
nonsource_node_list=in_state_node_list;
nonsource_node_list(I1)=[];% nonsource node list
N=length(nonsource_node_list); % number of nonsource node.
T=size(in_state(1).real,2);

% Speed up the code, initialize grad_A matrix into a structure by time
temp_A=zeros(6*N);
grad_A=[];
for i_t=1:T
    grad_A(i_t).node_list=nonsource_node_list;
    grad_A(i_t).M=temp_A;
end


% Then filling the values for grad_A's matrices of each time section
for i_x=1:N
    temp_x_idx=(i_x-1)*6+[1:6];
    
    target_node=nonsource_node_list(i_x);
    I1=find(Model_node_list==target_node);
    Model_local=Model(I1);
    rZ_nn=Model_local.rZ_nn;
    
    for i_y=1:N
        dev_node=nonsource_node_list(i_y);
        neighbor_idx=find(Model_local.neighbor==dev_node);
        
        if isempty(neighbor_idx) && dev_node~=target_node % if the dev_node is neither the target node nor the neighbor
            continue;
            
        elseif dev_node==target_node % if the dev_node is the target node
            I1=find(label_node_list==dev_node); % take the node's data
            local_P=label(I1).P; % P and Q data, 3-by-T
            local_Q=label(I1).Q;
            I1=find(in_state_node_list==dev_node);
            local_real=in_state(I1).real;% State real and imaginary data, 3-by-T
            local_imag=in_state(I1).imag;
            
            dR_R=[];
            dR_I=[];
            dI_R=[];
            dI_I=[];
            
            for i_phase=1:3 % phase a b c
                temp_P=local_P(i_phase,:); % take that phase's data
                temp_Q=local_Q(i_phase,:);
                temp_real=local_real(i_phase,:);
                temp_imag=local_imag(i_phase,:);
                
                denominator=temp_real.^2+temp_imag.^2;
                numerator_real=temp_P.*temp_real+temp_Q.*temp_imag;
                numerator_imag=temp_P.*temp_imag-temp_Q.*temp_real;
                divider=denominator.^2;
                
                dR_R(i_phase).v=(temp_P.*denominator-numerator_real.*temp_real*2)./divider; % derivative real to real.
                dR_I(i_phase).v=(temp_Q.*denominator-numerator_real.*temp_imag*2)./divider; % derivative real to imag.
                dI_R(i_phase).v=dR_I(i_phase).v; % derivative imag to real.
                dI_I(i_phase).v=-dR_R(i_phase).v; % derivative imag to imag.
            end
                        
            temp_y_idx=(i_y-1)*6+[1:6];
            for i_t=1:T
                temp_RR=diag([dR_R(1).v(i_t),dR_R(2).v(i_t),dR_R(3).v(i_t)]);
                temp_RI=diag([dR_I(1).v(i_t),dR_I(2).v(i_t),dR_I(3).v(i_t)]);
                temp_IR=diag([dI_R(1).v(i_t),dI_R(2).v(i_t),dI_R(3).v(i_t)]);
                temp_II=diag([dI_I(1).v(i_t),dI_I(2).v(i_t),dI_I(3).v(i_t)]);
                temp_M=[temp_RR,temp_RI;temp_IR,temp_II];
                
                grad_A(i_t).M(temp_x_idx,temp_y_idx)=rZ_nn*temp_M;
            end
            
        else % if the dev node is a neighbor
            rY_nk=Model_local.rY_nk(neighbor_idx).M;
            temp_M=-rZ_nn*rY_nk;
            temp_y_idx=(i_y-1)*6+[1:6];
            for i_t=1:T
                grad_A(i_t).M(temp_x_idx,temp_y_idx)=temp_M;
            end
        end
        
    end
end

end


%%% function to calculate b(t) for gradient
function grad_b=f_grad_b(output,output_correct,output_model,in_state,source_node)
% Input:
% output--a structure of output. output(t).meter_list is a vector containing the
% list of meter names (same for all t). output(t).value is a vector of
% output value (in this case the volt measurement) matching to each meter
% at time t.
% output_correct--the same structure as output, but contains the actual
% output measurements.
% output_model--structure containing how the state is mapped to the output.
% in_state--input state structure containing the nodal volt real and imaginary part.
% source_node--the source node.

% Output:
% grad_b--a structure containg d e_w(t)/d o(t) * d G(x(t)/d x(t).
% grad_b(t).node_list is the list of nodes. grad_b(t).vec is
% a 1-by-6N vector (N the number of nonsource nodes) at time t.


% Structure of output_model:
% output_model(i).meter--the name of the meter i.
% output_model(i).node--the node which the meter is connected to.
% output_model(i).phase--the phase of the smart meter i.
% 1,2,3,12,23,31,301,302,303 represents phase A, B, C, AB, BC, CA, 3-phase
% A, 3-phase B, and 3-phase C.
% output_model(i).f_v--the vector model of meter i. f_v is 1-by-3, a
% selection vetor of 1,-1, and 0 indicating which phase has which sign. f_v
% is the same for both real and imaginary part of a state of one particular
% meter.

% Strucutre of in_state or complex state
% complex_state(i).node--name of node i.
% complex_state(i).real--a 3-by-T matrix of nodal volt real part of node i of all
% time through T.
% complex_state(i).imag-a 3-by-T matrix of nodal volt imaginary part of node i of
% all time through T.
%----------------------------------------------------------

node_list=[in_state.node]; % list of node
I1=find(node_list==source_node);
nonsource_node_list=node_list;
nonsource_node_list(I1)=[];% nonsource node list
N=length(nonsource_node_list); % number of nonsource node.
T=size(in_state(1).real,2);
meter_list=[output(1).meter_list];
output_model_meter_list=[output_model.meter];
M=length(output(1).meter_list);

de_do=zeros(T,M);
for i_t=1:T % initialize grad_b, and prepare de/do.
    grad_b(i_t).node_list=nonsource_node_list;
    grad_b(i_t).vec=zeros(1,6*N);
    temp_v=2/M*(output(i_t).value-output_correct(i_t).value);
    de_do(i_t,:)=temp_v';% This is 1-by-M.
end


% filling the actual grad_b
for i_meter=1:M % for each meter
    target_meter=meter_list(i_meter);
    I1=find(output_model_meter_list==target_meter);
    target_node=output_model(I1).node; % the node connected to the target_meter
    f_v=output_model(I1).f_v;
    phase_type=output_model(I1).phase;
    
    I1=find(node_list==target_node);
    temp_real=f_v*in_state(I1).real; % 1-by-T
    temp_imag=f_v*in_state(I1).imag;
    
    denomenator=sqrt(temp_real.^2+temp_imag.^2);
    
    idx_node=find(nonsource_node_list==target_node);
    %%% below the calculation is based on
    %%% d_e/d_x_k=sum_over_m(d_e/d_o_m*d_g_m/d_x_r). More code but less
    %%% calculation.
    switch phase_type % filling grad_b based on phase type
        case 1 % phase A
            idx1=1;
            
            real_grad=temp_real./denomenator; % d_gm/d_real
            imag_grad=temp_imag./denomenator; % d_gm/d_imag
            for i_t=1:T % update each time
                temp_idx=(idx_node-1)*6+idx1; % idx of real of target node
                grad_b(i_t).vec(temp_idx)=grad_b(i_t).vec(temp_idx)+de_do(i_t,i_meter)*real_grad(i_t);
                temp_idx=(idx_node-1)*6+idx1+3; % idx of imag of target node
                grad_b(i_t).vec(temp_idx)=grad_b(i_t).vec(temp_idx)+de_do(i_t,i_meter)*imag_grad(i_t);
            end
            
        case 2 % phase B
            idx1=2;
            
            real_grad=temp_real./denomenator; % d_gm/d_real
            imag_grad=temp_imag./denomenator; % d_gm/d_imag
            for i_t=1:T % update each time
                temp_idx=(idx_node-1)*6+idx1; % idx of real of target node
                grad_b(i_t).vec(temp_idx)=grad_b(i_t).vec(temp_idx)+de_do(i_t,i_meter)*real_grad(i_t);
                temp_idx=(idx_node-1)*6+idx1+3; % idx of imag of target node
                grad_b(i_t).vec(temp_idx)=grad_b(i_t).vec(temp_idx)+de_do(i_t,i_meter)*imag_grad(i_t);
            end
            
        case 3 % phase C
            idx1=3;
            
            real_grad=temp_real./denomenator; % d_gm/d_real
            imag_grad=temp_imag./denomenator; % d_gm/d_imagg
            for i_t=1:T % update each time
                temp_idx=(idx_node-1)*6+idx1; % idx of real of target node
                grad_b(i_t).vec(temp_idx)=grad_b(i_t).vec(temp_idx)+de_do(i_t,i_meter)*real_grad(i_t);
                temp_idx=(idx_node-1)*6+idx1+3; % idx of imag of target node
                grad_b(i_t).vec(temp_idx)=grad_b(i_t).vec(temp_idx)+de_do(i_t,i_meter)*imag_grad(i_t);
            end
            
        case 12 % phase AB
            idx1=1; 
            idx2=2;
            
            real_grad=temp_real./denomenator; % d_gm/d_real
            imag_grad=temp_imag./denomenator; % d_gm/d_imag
            for i_t=1:T % update each time
                temp_idx=(idx_node-1)*6+idx1; % idx of real of target node (first phase)
                grad_b(i_t).vec(temp_idx)=grad_b(i_t).vec(temp_idx)+de_do(i_t,i_meter)*real_grad(i_t);
                temp_idx=(idx_node-1)*6+idx1+3; % idx of imag of target node
                grad_b(i_t).vec(temp_idx)=grad_b(i_t).vec(temp_idx)+de_do(i_t,i_meter)*imag_grad(i_t);
                
                % below pay attention to idx and sign before de_do. (second pahse)
                temp_idx=(idx_node-1)*6+idx2; % idx of real of target node
                grad_b(i_t).vec(temp_idx)=grad_b(i_t).vec(temp_idx)-de_do(i_t,i_meter)*real_grad(i_t);
                temp_idx=(idx_node-1)*6+idx2+3; % idx of imag of target node
                grad_b(i_t).vec(temp_idx)=grad_b(i_t).vec(temp_idx)-de_do(i_t,i_meter)*imag_grad(i_t);
            end
            
        case 23 % phase BC
            idx1=2; 
            idx2=3;
            
            real_grad=temp_real./denomenator; % d_gm/d_real
            imag_grad=temp_imag./denomenator; % d_gm/d_imag
            for i_t=1:T % update each time
                temp_idx=(idx_node-1)*6+idx1; % idx of real of target node (first phase)
                grad_b(i_t).vec(temp_idx)=grad_b(i_t).vec(temp_idx)+de_do(i_t,i_meter)*real_grad(i_t);
                temp_idx=(idx_node-1)*6+idx1+3; % idx of imag of target node
                grad_b(i_t).vec(temp_idx)=grad_b(i_t).vec(temp_idx)+de_do(i_t,i_meter)*imag_grad(i_t);
                
                % below pay attention to idx and sign before de_do. (second pahse)
                temp_idx=(idx_node-1)*6+idx2; % idx of real of target node
                grad_b(i_t).vec(temp_idx)=grad_b(i_t).vec(temp_idx)-de_do(i_t,i_meter)*real_grad(i_t);
                temp_idx=(idx_node-1)*6+idx2+3; % idx of imag of target node
                grad_b(i_t).vec(temp_idx)=grad_b(i_t).vec(temp_idx)-de_do(i_t,i_meter)*imag_grad(i_t);
            end
                 
        case 31 % phase CA
            idx1=3;
            idx2=1;
            
            real_grad=temp_real./denomenator; % d_gm/d_real
            imag_grad=temp_imag./denomenator; % d_gm/d_imag
            for i_t=1:T % update each time
                temp_idx=(idx_node-1)*6+idx1; % idx of real of target node (first phase)
                grad_b(i_t).vec(temp_idx)=grad_b(i_t).vec(temp_idx)+de_do(i_t,i_meter)*real_grad(i_t);
                temp_idx=(idx_node-1)*6+idx1+3; % idx of imag of target node
                grad_b(i_t).vec(temp_idx)=grad_b(i_t).vec(temp_idx)+de_do(i_t,i_meter)*imag_grad(i_t);
                
                % below pay attention to idx and sign before de_do. (second pahse)
                temp_idx=(idx_node-1)*6+idx2; % idx of real of target node
                grad_b(i_t).vec(temp_idx)=grad_b(i_t).vec(temp_idx)-de_do(i_t,i_meter)*real_grad(i_t);
                temp_idx=(idx_node-1)*6+idx2+3; % idx of imag of target node
                grad_b(i_t).vec(temp_idx)=grad_b(i_t).vec(temp_idx)-de_do(i_t,i_meter)*imag_grad(i_t);
            end
            
        case 301 % phase ABC-A
            idx1=1;
            
            real_grad=temp_real./denomenator; % d_gm/d_real
            imag_grad=temp_imag./denomenator; % d_gm/d_imag
            for i_t=1:T % update each time
                temp_idx=(idx_node-1)*6+idx1; % idx of real of target node
                grad_b(i_t).vec(temp_idx)=grad_b(i_t).vec(temp_idx)+de_do(i_t,i_meter)*real_grad(i_t);
                temp_idx=(idx_node-1)*6+idx1+3; % idx of imag of target node
                grad_b(i_t).vec(temp_idx)=grad_b(i_t).vec(temp_idx)+de_do(i_t,i_meter)*imag_grad(i_t);
            end
            
        case 302 % phase ABC-B
            idx1=2;
            
            real_grad=temp_real./denomenator; % d_gm/d_real
            imag_grad=temp_imag./denomenator; % d_gm/d_imag
            for i_t=1:T % update each time
                temp_idx=(idx_node-1)*6+idx1; % idx of real of target node
                grad_b(i_t).vec(temp_idx)=grad_b(i_t).vec(temp_idx)+de_do(i_t,i_meter)*real_grad(i_t);
                temp_idx=(idx_node-1)*6+idx1+3; % idx of imag of target node
                grad_b(i_t).vec(temp_idx)=grad_b(i_t).vec(temp_idx)+de_do(i_t,i_meter)*imag_grad(i_t);
            end
            
        case 303 % phase ABC-C
            idx1=3;
            
            real_grad=temp_real./denomenator; % d_gm/d_real
            imag_grad=temp_imag./denomenator; % d_gm/d_imag
            for i_t=1:T % update each time
                temp_idx=(idx_node-1)*6+idx1; % idx of real of target node
                grad_b(i_t).vec(temp_idx)=grad_b(i_t).vec(temp_idx)+de_do(i_t,i_meter)*real_grad(i_t);
                temp_idx=(idx_node-1)*6+idx1+3; % idx of imag of target node
                grad_b(i_t).vec(temp_idx)=grad_b(i_t).vec(temp_idx)+de_do(i_t,i_meter)*imag_grad(i_t);
            end
    end
end

end

%%% function to calculate d F/d w for gradient
function grad_dF_dw=f_grad_dF_dw(in_state,Model,label,w,d_GB_d_w,source_node)
% Input:
% in_state--input state structure containing the nodal volt real and
% reactive part.
% Model--the structure that stores the needed A1 matrix and A2 matrics for 
% each node's local transition function.
% label--label structure containing the nodal real and reactive power
% injection.
% w--the weight/parameter. In our problem this is the set of line
% parameters.
% d_GB_d_w--a structure storing the fixed derivative matrices for line
% parameter.d_GB_d_w(i).M is a 3-by-3 matrix. i=1~6, representing 6 phase
% types: aa, ab, ac, bb, bc, cc. G and B share the same patter, so we only 
% i=1~6 to represent these cases.

% Output:
% grad_dF_dw--a structure containg d F_w(x(t),l(t))/d w.
% grad_dF_dw(t).line_list is the list of lines, which is a 2-column matrix, in
% which each row is a line and each column is a vertex. 
% grad_dF_dw(t).node_list is the list of nodes (except source node). grad_dF_dw(t).M is
% a 6N-by-|w| vector (|w| the number of line parameters, i.e., 12*number of lines) at time t.


% Strucutre of in_state or complex state
% complex_state(i).node--name of node i.
% complex_state(i).real--a 3-by-T matrix of nodal volt real part of node i of all
% time through T.
% complex_state(i).imag-a 3-by-T matrix of nodal volt imaginary part of node i of
% all time through T.

% Structure of the Model:
% Model(i).node--node name of node i.
% Model(i).neighbor--vector of node i's neighbors.
% Model(i).rZ_nn--Z_nn real matrix for node i's local transition function
% (6-by-6).
% Model(i).rY_nk--structure of Y_nk real matrix for node i's local transition function.
% Model(i).rY_nk(j).M--the Y_nk real matrix corresponding to node i's
% neighbor node j (6-by-6).

% Structure of label:
% label(i).node--name of node i.
% label(i).P--a 3-by-T matrix of nodal real power of node i of all
% time through T.
% label(i).Q--a 3-by-T matrix of nodal reactive power of node i of
% all time through T.

% Structure of w:
% w(i).line--a 2-element vector containing the vertices of the line. Length
% of w is the number of lines.
% w(i).g--a vector of conductance of line i, contain 6 elements.
% w(i).b--a vector of susceptance of line i, contain 6 elements.
% The 6 elements are in the order of phase aa, ab, ac, bb, bc, cc.
%-----------------------------------------------------------------

state_node_list=[in_state.node]; % list of node
I1=find(state_node_list==source_node);
nonsource_node_list=state_node_list;
nonsource_node_list(I1)=[];% nonsource node list
N=length(nonsource_node_list); % number of nonsource node.
T=size(in_state(1).real,2);

% state_node_list=[in_state.node]; %list of node names in in_state.
Model_node_list=[Model.node]; % list of node names in Model.
label_node_list=[label.node]; % list of node names in label.

temp_store=[];
for i=1:N
    target_node=nonsource_node_list(i);
    model_idx=find(Model_node_list==target_node); % find the local model for the target node.
    label_idx=find(label_node_list==target_node); % find the local model for the target node.
    grad_local_df_dw=f_grad_local_df_dw(in_state,Model(model_idx),label(label_idx),w,d_GB_d_w); % gradient of a local node
    % grad_local_df_dw--a structure containg d f_n_w(x_ne(n)(t),l_n(t))/d w.
    % grad_local_df_dw(t).line_list is the list of lines, which is a 2-column matrix, in
    % which each row is a line and each column is a vertex. grad_local_df_dw(t).M is
    % a 6-by-|w| vector (|w| the number of line parameters, i.e., 12*number of lines) at time t.
    
    temp_store(i).struct=grad_local_df_dw;
end

grad_dF_dw=[];
one_temp_store=temp_store(1).struct;
for i_t=1:T
    grad_dF_dw(i_t).line_list=one_temp_store(1).line_list;
    grad_dF_dw(i_t).node_list=nonsource_node_list;
    
    temp_M=[];
    for i_node=1:N
        pick_temp_store=temp_store(i_node).struct;
        temp_M=[temp_M;pick_temp_store(i_t).M];
    end
    
    grad_dF_dw(i_t).M=temp_M;
end

end

%%% function to calculate local df/dw for gradient
function grad_local_df_dw=f_grad_local_df_dw(in_state,Model_local,label_local,w,d_GB_d_w)
% Input:
% in_state--input state structure containing the nodal volt real and
% imaginary part.
% Model_local--the structure that stores the needed rZ_nn matrix and rY_nk
% matrices for the one target node.
% label_local--label of the local node, i.e., the nodal real and reactive
% power.
% w--the weight/parameter. In our problem this is the set of line
% parameters.
% d_GB_d_w--a structure storing the fixed derivative matrices for line
% parameter.d_GB_d_w(i).M is a 3-by-3 matrix. i=1~6, representing 6 phase
% types: aa, ab, ac, bb, bc, cc. G and B share the same patter, so we only 
% i=1~6 to represent these cases.

% Output:
% grad_local_df_dw--a structure containg d f_n_w(x_ne(n)(t),l_n(t))/d w.
% grad_local_df_dw(t).line_list is the list of lines, which is a 2-column matrix, in
% which each row is a line and each column is a vertex. grad_local_df_dw(t).M is
% a 6-by-|w| vector (|w| the number of line parameters, i.e., 12*number of lines) at time t.


% Strucutre of in_state or complex state (remove(i))
% complex_state(i).node--name of node i.
% complex_state(i).real--a 3-by-T matrix of nodal volt real part of node i of all
% time through T.
% complex_state(i).imag-a 3-by-T matrix of nodal volt imaginary part of node i of
% all time through T.

% Structure of the Model_local: (remove(i))
% Model(i).node--node name of node i.
% Model(i).neighbor--vector of node i's neighbors.
% Model(i).rZ_nn--Z_nn real matrix for node i's local transition function
% (6-by-6).
% Model(i).rY_nk--structure of Y_nk real matrix for node i's local transition function.
% Model(i).rY_nk(j).M--the Y_nk real matrix corresponding to node i's
% neighbor node j (6-by-6).

% Structure of label_local:
% label_local.node--name of a node.
% label_local.P--a 3-by-T matrix of nodal real power of a node of all
% time through T.
% label_local.Q--a 3-by-T matrix of nodal reactive power of a node of
% all time through T.

% Structure of w:
% w(i).line--a 2-element vector containing the vertices of the line. Length
% of w is the number of lines.
% w(i).g--a vector of conductance of line i, contain 6 elements.
% w(i).b--a vector of susceptance of line i, contain 6 elements.
% The 6 elements are in the order of phase aa, ab, ac, bb, bc, cc.
%-----------------------------------------------------------------

node_list=[in_state.node]; % list of node
% I1=find(node_list==source_node);
% nonsource_node_list=node_list;
% nonsource_node_list(I1)=[];% nonsource node list
% N=length(nonsource_node_list); % number of nonsource node.
T=size(in_state(1).real,2);


target_node=Model_local.node;
rZ_nn=Model_local.rZ_nn;
rY_nk=Model_local.rY_nk;
local_neighbor=Model_local.neighbor;
Num_neighbor=length(local_neighbor);% number of neighbors
% local_label_M=[label_local.P;label_local.Q]; % a 6-by-T local label matrix

w_line_list=[];
for i_w=1:length(w)
    target_line=w(i_w).line;
    w_line_list=[w_line_list;target_line];
end
Num_line=length(w); % number of lines

% build G_nn and B_nn
temp_M=rZ_nn(1:3,1:3)+1i*rZ_nn(4:6,1:3);
Y_nn=inv(temp_M);
G_nn=real(Y_nn);
B_nn=imag(Y_nn);
G_nn_inv=inv(G_nn);

%%% calculate d_rZ_nn/d_w, d_rY_nk/d_w only on lines that connects the
%%% target node. For a line that is not connected to the target node, the
%%% gradient is 0 and is not recoreded here. For a line that is connected
%%% to the target node, for each of the 12 paramters on this line, we need
%%% to record the non-zero d_rZ_nn/d_w, one and only one non-zero 
%%% d_rY_nk/d_w for a particular k, and the corresponding neigbor node name
%%% . This is the preparation for the later caculation.
record_struct=[];
i_record=0;
for i_neighbor=1:Num_neighbor
    temp_neighbor=Model_local.neighbor(i_neighbor);% current neighbor
    I1=find(w_line_list(:,1)==target_node & w_line_list(:,2)==temp_neighbor);
    I2=find(w_line_list(:,2)==target_node & w_line_list(:,1)==temp_neighbor);
    i_w=[I1;I2];
    
    for i_p=1:12 % each line contains 12 parameters
        
        % note that the d_G_nk_d_w and d_B_nk_d_w are non-zero for only one
        % particular k that represents the neighbor line.
        if i_p<=6 % if it is g parameter
            d_G_nk_d_w=d_GB_d_w(i_p).M;
            d_B_nk_d_w=zeros(3);
        else % if 6<i_p<=12, it is b parameter
            d_G_nk_d_w=zeros(3);
            d_B_nk_d_w=d_GB_d_w(i_p-6).M;
        end
        d_G_nn_d_w=d_G_nk_d_w;
        d_B_nn_d_w=d_B_nk_d_w;
        
        % calculate d_Gnn_inv_d_w
        d_Gnn_inv_d_w=zeros(3);
        for i_1=1:3
            for i_2=1:3
                d_Gnn_inv_ij_d_Gnn=f_dA_inv(G_nn_inv,i_1,i_2);
                d_Gnn_inv_d_w(i_1,i_2)=d_Gnn_inv_ij_d_Gnn(:)'*d_G_nn_d_w(:); % this is equivalent to a trace calculation
            end
        end
        
        d_Re_Znn_inv_d_w=d_G_nn_d_w+d_B_nn_d_w*G_nn_inv*B_nn+B_nn*d_Gnn_inv_d_w*B_nn+B_nn*G_nn_inv*d_B_nn_d_w;
        
        d_Re_Znn_d_w=zeros(3);
        for i_1=1:3
            for i_2=1:3
                d_Re_Znn_ij_d_Re_Znn_inv=f_dA_inv(rZ_nn(1:3,1:3),i_1,i_2);
                d_Re_Znn_d_w(i_1,i_2)=d_Re_Znn_ij_d_Re_Znn_inv(:)'*d_Re_Znn_inv_d_w(:); % this is equivalent to a trace calculation
            end
        end
        
        d_Im_Znn_d_w=-d_Re_Znn_d_w*B_nn*G_nn_inv-rZ_nn(1:3,1:3)*d_B_nn_d_w*G_nn_inv-rZ_nn(1:3,1:3)*B_nn*d_Gnn_inv_d_w;
        
        d_rZ_nn_d_w=[d_Re_Znn_d_w,-d_Im_Znn_d_w;d_Im_Znn_d_w,d_Re_Znn_d_w];
        
        d_Ynk_d_w=[-d_G_nk_d_w,d_B_nk_d_w;-d_B_nk_d_w,-d_G_nk_d_w];

        i_record=i_record+1;
        record_struct(i_record).line_idx=i_w;
        record_struct(i_record).neighbor=temp_neighbor;
        record_struct(i_record).para_idx=i_p;
        record_struct(i_record).d_Ynk_d_w=d_Ynk_d_w;
        record_struct(i_record).d_rZ_nn_d_w=d_rZ_nn_d_w;
    end
 
end

%%% Initialize derivative
grad_local_df_dw=[];
for i_t=1:T %
    grad_local_df_dw(i_t).line_list=w_line_list;
    grad_local_df_dw(i_t).M=zeros(6,12*Num_line); % gradient, initialize as zeros.
end

%%% Now calculate derivative w.r.t each line parameter

I1=find(node_list==target_node);
target_V_conj=in_state(I1).real-1i*in_state(I1).imag; % complex conjugate of volt
target_S_conj=label_local.P-1i*label_local.Q; % complex conjugate of power
temp_M=target_S_conj./target_V_conj;
self_data=[real(temp_M);imag(temp_M)]; % first term of final gradient

neighbor_data=0;
for i=1:Num_neighbor
    temp_neighbor=local_neighbor(i);
    I1=find(node_list==temp_neighbor);
    temp_M=[in_state(I1).real;in_state(I1).imag];
    neighbor_data=neighbor_data+rY_nk(i).M*temp_M; %second term of the final gradient
end
    
record_widx_list=[record_struct.line_idx];
for i_w=1:length(w) % go through each line
    I1=find(record_widx_list==i_w);
    if isempty(I1)
        continue; % if this line is not a neighbor line, skip
    end
    
    temp_struct=record_struct(I1);
    for i_struct=1:length(temp_struct)
        
        temp_neighbor=temp_struct(i_struct).neighbor;
        i_p=temp_struct(i_struct).para_idx;
        d_Ynk_d_w=temp_struct(i_struct).d_Ynk_d_w;
        d_rZ_nn_d_w=temp_struct(i_struct).d_rZ_nn_d_w;
        
        temp_M1=d_rZ_nn_d_w*self_data;
        
        temp_M2=d_rZ_nn_d_w*neighbor_data;
        
        I1=find(node_list==temp_neighbor);
        temp_M3=rZ_nn*d_Ynk_d_w*[in_state(I1).real;in_state(I1).imag];
        
        temp_gradient=temp_M1-temp_M2-temp_M3; % final gradient w.r.t to current w, a 6-by-T
        
        temp_idx=(i_w-1)*12+i_p;
        for i_t=1:T %
            grad_local_df_dw(i_t).M(:,temp_idx)=temp_gradient(:,i_t);
        end
        
    end
    
end

end

%%% function to calculate d (A^-1)_ij/d A
function dA_inv=f_dA_inv(A_inv,i,j)
% Input:
% A_inv--input matrix, the inverse of A.
% i, j--the location of i j.
% Output:
% dA_inv=d (A^-1)_ij/d A

% %%% old code
% n=size(A_inv,1);
% E_ij=zeros(n);
% E_ij(i,j)=1;
% dA_inv=-A_inv'*E_ij*A_inv';

%%% faster code
n=size(A_inv,1);
temp_A=A_inv';
dA_inv=-temp_A(:,i)*temp_A(j,:);
end

%%% function to calculate error between two output
% plot figure to compare the output function value and the actual meter
% volt
function error=f_error(output_1,output_2)

T=length(output_1);
if length(output_2)~=T
    disp('error: two input of differenc sizes.')
    return
end
M=length(output_1(1).meter_list); % number of meters
M_1=[];
M_2=[];
for i_t=1:T
    M_1=[M_1,output_1(i_t).value];
    M_2=[M_2,output_2(i_t).value];
end

error=norm(M_1-M_2,'fro')^2/(T*M);

end

% a temp function for debug
function error=f_error_diff(output_1,output_2)

T=length(output_1);
M=length(output_1(1).meter_list); % number of meters
M_1=[];
M_2=[];
for i_t=1:T
    M_1=[M_1,output_1(i_t).value];
    M_2=[M_2,output_2(i_t).value];
end

error=norm(diff(M_1,1,2)-diff(M_2,1,2),'fro')^2/(T*M);

end

%% 2020-07-26 add functions for R X modeling

%%% FORWARD function to calculate the converging state based on current
%%% weights.

function [out_state_compact,iter_num,NaN_flag]=f_FORWARD_RX(w_z,in_state_complex,label,Line_config,threshold_ratio,source_state_complex)
% Input:
% w_z--the weight/parameter. In our problem this is the set of line
% parameters in forms of R X.
% in_state_complex--input state structure containing the nodal volt real and imaginary part. 
% In this problem, it is the initialized state.
% label--label structure containing the nodal real and reactive power
% injection.
% Line_config--a 2-column matrix showing the connection of the circuit.
% Each row is a line; each column is one vertex of the line.
% threshold_ratio --threshold to stop iteration. It is a ratio indicating how much
% is changed compared with the previous iteration.
% source_state_complex--the state of the source node, has the same structure as
% one element in the in_state_complex.

% Output:
% out_state_compact--output state structure containing the nodal volt real
% and imaginary part.
% iter_num--number of iteration to get the state converging
% NaN_flag--used to indicate iteration failure. Some wrong w may cause inf
% or NaN in this function. If so, stop the function and return NaN_flag=1.
% The very wrong w may be caused by a too large step size.

%%% Details of the structures---------------------------------
% Structure of w:
% w(i).line--a 2-element vector containing the vertices of the line. Length
% of w is the number of lines.
% w(i).g--a vector of conductance of line i, contain 6 elements.
% w(i).b--a vector of susceptance of line i, contain 6 elements.

% Strucutre of complex state
% complex_state(i).node--name of node i.
% complex_state(i).real--a 3-by-T matrix of nodal volt real part of node i of all
% time through T.
% complex_state(i).imag-a 3-by-T matrix of nodal volt imaginary part of node i of
% all time through T.

% Structure of label:
% label(i).node--name of node i.
% label(i).P--a 3-by-T matrix of nodal real power of node i of all
% time through T.
% label(i).Q--a 3-by-T matrix of nodal reactive power of node i of
% all time through T.

% Structure of out_state_complex:
% out_state_complex has the same structure as in_state_complex.

%---------------------------------------------------------------------
% some setup
max_iter=1e4; % define maximum iteration
NaN_flag=0;
in_state_compact=f_state2state_compact(in_state_complex);


% Prepare for the iteration
% T=size(label(1).P,2); % number of timestamps

% Build model based on line configuration and parameters
% Model--the structure that stores the needed A1 matrix and A2 matrics for 
% each node's local transition function.
% Structure of the Model:
% Model(i).node--node name of node i.
% Model(i).neighbor--vector of node i's neighbors.
% Model(i).A1--A1 matrix for node i's local transition function
% Model(i).A2--A2 matrix for node i's local transition function.
% Model(i).A2(j).M--the A2 matrix corresponding to node i's neighbor node
% j.

% w_y=f_RX2GB_struct(w_z); % transfer RX to GB.

Model=f_build_transition_model_RX(w_z,Line_config);
node_list_in_state=in_state_compact.node_list;
source_idx=find(node_list_in_state==source_state_complex.node);


% Iteration
iter_state=in_state_compact; %initialize state
% reset source node to the correct value;
iter_state.scalor_struct(source_idx).scalor=[source_state_complex.real;source_state_complex.imag];
iter_state.conj_struct(source_idx).conj=source_state_complex.real-1i*source_state_complex.imag;
    
iter_num=0; % count the number of iterations.
stop_flag=0;
% iter_state_magrad=f_compact2magradmat(iter_state);
iter_state_scalor=[iter_state.scalor_struct(:).scalor];
while iter_num<=max_iter && stop_flag==0
    iter_num=iter_num+1;
    
    old_state_scalor=iter_state_scalor;
    iter_state=f_global_transition(iter_state,Model,label);
    % reset source node to the correct value;
    iter_state.scalor_struct(source_idx).scalor=[source_state_complex.real;source_state_complex.imag];
    iter_state.conj_struct(source_idx).conj=source_state_complex.real-1i*source_state_complex.imag;
%     iter_state_magrad=f_compact2magradmat(iter_state);
    
    iter_state_scalor=[iter_state.scalor_struct(:).scalor];
    % debug
    xx=find(isnan(old_state_scalor),1);
    if ~isempty(xx)
        debug_flag=1;
        NaN_flag=1;
        stop_flag=1;
    end
    
    temp_improve=norm(iter_state_scalor-old_state_scalor,'fro')^2;
    old_MSE=norm(old_state_scalor,'fro')^2;
    if temp_improve<old_MSE*threshold_ratio
        stop_flag=1; % if the MSE of state update is smaller than a ratio of previous state's MSE, then stop iteration.
    end
    
    % debug use, remove when in formal use
%     state_magrad_history(iter_num).state_magrad=iter_state_magrad;
    

%     % use maximum absolute value update as threshold
%     temp_1=[[iter_state(:).mag];[iter_state(:).rad]];
%     if f_state_update_MSE(iter_state,old_state)<f_state_MSE(old_state)*threshold_ratio 
%         stop_flag=1; % if the MSE of state update is smaller than a ratio of previous state's MSE, then stop iteration.
%     end
end
out_state_compact=f_compact2state_complex(iter_state);

end

%%% Build model strucutre based on line configuration and line parameters
function Model=f_build_transition_model_RX(w_z,Line_config)
% Input:
% w_z--the weight/parameter. In our problem this is the set of line
% parameters in forms of R X.
% Line_cofig--a 2-column matrix showing the connection of the circuit.
% Each row is a line; each column is one vertex of the line.

% Output:
% Model--the structure that stores the needed Z_nn matrix and Y_nk matrics for 
% each node's local transition function.

% Structure of w:
% w(i).line--a 2-element vector containing the vertices of the line. Length
% of w is the number of lines.
% w(i).g--a vector of conductance of line i, contain 6 elements.
% w(i).b--a vector of susceptance of line i, contain 6 elements.
% The 6 elements are in the order of phase aa, ab, ac, bb, bc, cc.

% Structure of the Model:
% Model(i).node--node name of node i.
% Model(i).neighbor--vector of node i's neighbors.
% Model(i).rZ_nn--Z_nn real matrix for node i's local transition function
% (6-by-6).
% Model(i).rY_nk--structure of Y_nk real matrix for node i's local transition function.
% Model(i).rY_nk(j).M--the Y_nk real matrix corresponding to node i's
% neighbor node j (6-by-6).
%---------------------------------------------------------------

w_y=f_RX2GB_struct(w_z); % transfer RX to GB.
node_list=unique([Line_config(:,1);Line_config(:,2)]); % 'unique' will sort the numbers in ascending order
node_num=length(node_list);
w_line_config=[];
for i_w=1:length(w_y) % build a line config from w.
    w_line_config=[w_line_config;w_y(i_w).line];
end

for i_node=1:node_num % build Model structure
    Model(i_node).node=node_list(i_node);
    
    % find neighbors
    targe_node=node_list(i_node);
    I1=find(Line_config(:,1)==targe_node);
    I2=find(Line_config(:,2)==targe_node);
    I3=[I1;I2];
    neighbor_vec=[Line_config(I1,2);Line_config(I2,1)];
    [neighbor_vec,temp_I]=sort(neighbor_vec); %sort the neighbor name in ascending order.
    Model(i_node).neighbor=neighbor_vec;
    
    % build A1 and A2 matrix
    Y_nn=0; %initialize Y_nn (complex)
    rY_nk=[]; % initialize rY_nk real matrix complex
%     R=Rotate_phi_fun(1,-1);
    
    for i_neighbor=1:length(neighbor_vec)
        temp_neighbor=neighbor_vec(i_neighbor);
        I1=find(w_line_config(:,1)==targe_node & w_line_config(:,2)==temp_neighbor);
        I2=find(w_line_config(:,2)==targe_node & w_line_config(:,1)==temp_neighbor);
        w_idx=[I1;I2];
        temp_Y_nk=f_Ynk_Matrix(w_y(w_idx).g,w_y(w_idx).b); % calculate the complex primitive Y matrix
        rY_nk(i_neighbor).M=f_complex2realmulti(temp_Y_nk);
        Y_nn=Y_nn-temp_Y_nk;
    end
    Z_nn=inv(Y_nn);
    rZ_nn=f_complex2realmulti(Z_nn);
    Model(i_node).rZ_nn=rZ_nn;
    Model(i_node).rY_nk=rY_nk;
    
end

end


%%% function of the BACKWARD
function [grad_w_z,iter_num]=f_BACKWARD_RX(in_state,output_correct,output_model,Model,label,w_z,d_GB_d_w,source_node,threshold_ratio)
% Input:
% in_state--input state structure containing the nodal volt real and
% imaginary part.
% output_correct--the same structure as output, but contains the actual
% output measurements.
% output_model--structure containing how the state is mapped to the output.
% Model--the structure that stores the needed Z_nn matrix and Y_nk matrics for 
% each node's local transition function.
% label--label structure containing the nodal real and reactive power
% injection.
% w_z--the weight/parameter. In our problem this is the set of line
% parameters in forms of R X
% d_GB_d_w--a structure storing the fixed derivative matrices for line
% parameter.d_GB_d_w(i).M is a 3-by-3 matrix. i=1~6, representing 6 phase
% types: aa, ab, ac, bb, bc, cc. G and B share the same patter, so we only 
% i=1~6 to represent these cases.
% source_node--the source node

% Output
% grad_w_z: gradient structure over R X.


% Strucutre of in_state or complex state
% complex_state(i).node--name of node i.
% complex_state(i).real--a 3-by-T matrix of nodal volt real part of node i of all
% time through T.
% complex_state(i).imag-a 3-by-T matrix of nodal volt imaginary part of node i of
% all time through T.

% Structure of output_model:
% output_model(i).meter--the name of the meter i.
% output_model(i).node--the node which the meter is connected to.
% output_model(i).phase--the phase of the smart meter i.
% 1,2,3,12,23,31,301,302,303 represents phase A, B, C, AB, BC, CA, 3-phase
% A, 3-phase B, and 3-phase C.
% output_model(i).f_v--the vector model of meter i. f_v is 1-by-3, a
% selection vetor of 1,-1, and 0 indicating which phase has which sign. f_v
% is the same for both real and imaginary part of a state of one particular
% meter.

% Structure of the Model:
% Model(i).node--node name of node i.
% Model(i).neighbor--vector of node i's neighbors.
% Model(i).rZ_nn--Z_nn real matrix for node i's local transition function
% (6-by-6).
% Model(i).rY_nk--structure of Y_nk real matrix for node i's local transition function.
% Model(i).rY_nk(j).M--the Y_nk real matrix corresponding to node i's
% neighbor node j (6-by-6).

% Structure of label:
% label(i).node--name of node i.
% label(i).P--a 3-by-T matrix of nodal real power of node i of all
% time through T.
% label(i).Q--a 3-by-T matrix of nodal reactive power of node i of
% all time through T.

% Structure of w:
% w(i).line--a 2-element vector containing the vertices of the line. Length
% of w is the number of lines.
% w(i).g--a vector of conductance of line i, contain 6 elements.
% w(i).b--a vector of susceptance of line i, contain 6 elements.
% The 6 elements are in the order of phase aa, ab, ac, bb, bc, cc.

max_iter=1e4;
w_y=f_RX2GB_struct(w_z); % transfer RX to GB.

node_list=[in_state.node]; % list of node
I1=find(node_list==source_node);
nonsource_node_list=node_list;
nonsource_node_list(I1)=[];% nonsource node listgrad_b
N=length(nonsource_node_list); % number of nonsource node.

T=size(in_state(1).real,2);
output=f_GNN_output(in_state,output_model);
grad_A=f_grad_A(in_state,label,Model,source_node);
grad_b=f_grad_b(output,output_correct,output_model,in_state,source_node);

%%% calcualte z
% initialize z
z_iter=[];
for i_t=1:T
    z_iter(i_t).M=zeros(1,6*N);
end

iter_num=0; % count the number of iterations.
stop_flag=0;
while iter_num<=max_iter && stop_flag==0
    iter_num=iter_num+1;
    
    old_z=z_iter;
    
    for i_t=1:T
        z_iter(i_t).M=z_iter(i_t).M*grad_A(i_t).M+grad_b(i_t).vec;
    end
    
    temp_new=[z_iter.M];
    temp_old=[old_z.M];
    norm_update=(norm(temp_new-temp_old,'fro')^2)/(T*length(z_iter(1).M));
    norm_old=(norm(temp_old,'fro')^2)/(T*length(z_iter(1).M));
    if norm_update<norm_old*threshold_ratio
        stop_flag=1; % if the MSE of state update is smaller than a ratio of previous state's MSE, then stop iteration.
    end
end

% % or calculate z by using theoretical inverse multiplication (this method causes singular matrix in simulation, so abandon it)
% iter_num=0;
% para_number=6*N;
% z_iter=[];% theoretical converged z
% I_a=eye(para_number);
% for i_t=1:T
%     z_iter(i_t).M=grad_b(i_t).vec*inv(I_a-grad_A(i_t).M); % This is quicker than iteration, but if condition number is large, use iteration instead.
% end

% % for debug or replace code
% para_number=6*N;
% z_converge=[];% theoretical converged z
% for i_t=1:T
%     z_converge(i_t).M=grad_b(i_t).vec*inv(eye(para_number)-grad_A(i_t).M); % I guess using iteration to calculate z is to avoid inverse.
% end
% i_t=1;
% figure;plot(1:36,[(z_iter(i_t).M)',(z_converge(i_t).M)'])
% legend('iter result z','theoretical z')
% % results show that the z calculated by iteration is very close to the
% % thoretical value using matrix inverse.


grad_dF_dw=f_grad_dF_dw(in_state,Model,label,w_y,d_GB_d_w,source_node);
grad_d=[];
for i_t=1:T
    grad_d(i_t).vec=z_iter(i_t).M*grad_dF_dw(i_t).M;
end

d_e_w_d_w=grad_d;

grad_w_vec=0;
for i_t=1:T
    grad_w_vec=grad_w_vec+d_e_w_d_w(i_t).vec;
end
grad_w_vec=grad_w_vec/T;

grad_w_y=w_y;
for i_w=1:length(w_y)
    temp_idx=(i_w-1)*12;
    grad_w_y(i_w).g=grad_w_vec(temp_idx+[1:6])';
    grad_w_y(i_w).b=grad_w_vec(temp_idx+[7:12])';
end

grad_w_z=f_GBgrad2RXgrad(grad_w_y,w_z,w_y,d_GB_d_w); % chain rule to calcualte gradient over R X.

end



%%% function to transfer admittance weight structure to impedance weight
%%% structure.
function w_z=f_GB2RX_struct(w_y)
L=length(w_y);

for i_L=1:L
    w_z(i_L).line=w_y(i_L).line;
    
    temp_g=w_y(i_L).g; % g's phase order: aa, ab, ac, bb, bc. cc
    temp_b=w_y(i_L).b;
    
    gmatrix=[temp_g(1),temp_g(2),temp_g(3); ...
        temp_g(2),temp_g(4),temp_g(5); ...
        temp_g(3),temp_g(5),temp_g(6);];
    
    bmatrix=[temp_b(1),temp_b(2),temp_b(3); ...
        temp_b(2),temp_b(4),temp_b(5); ...
        temp_b(3),temp_b(5),temp_b(6);];
    
    ymatrix=gmatrix+1i*bmatrix;
    zmatrix=inv(ymatrix);
    temp_v=[zmatrix(1,1),zmatrix(1,2),zmatrix(1,3),zmatrix(2,2),zmatrix(2,3),zmatrix(3,3)].';
    
    w_z(i_L).r=real(temp_v);
    w_z(i_L).x=imag(temp_v);
end
end

%%% function to transfer impedance weight structure to admittance weight
%%% structure.
function w_y=f_RX2GB_struct(w_z)
L=length(w_z);

for i_L=1:L
    w_y(i_L).line=w_z(i_L).line;
    
    temp_r=w_z(i_L).r; % r's phase order: aa, ab, ac, bb, bc. cc
    temp_x=w_z(i_L).x;
    
    rmatrix=[temp_r(1),temp_r(2),temp_r(3); ...
        temp_r(2),temp_r(4),temp_r(5); ...
        temp_r(3),temp_r(5),temp_r(6);];
    
    xmatrix=[temp_x(1),temp_x(2),temp_x(3); ...
        temp_x(2),temp_x(4),temp_x(5); ...
        temp_x(3),temp_x(5),temp_x(6);];
    
    zmatrix=rmatrix+1i*xmatrix;
    ymatrix=inv(zmatrix);
    temp_v=[ymatrix(1,1),ymatrix(1,2),ymatrix(1,3),ymatrix(2,2),ymatrix(2,3),ymatrix(3,3)].';
    
    w_y(i_L).g=real(temp_v);
    w_y(i_L).b=imag(temp_v);
end
end

%%% function to reorganize a weight strucutre to a vector
function w_vec=f_wz_struct2vec(w_in)
% Input:
% w_in--the weight/parameter. In our problem this is the set of line
% parameters. The structure is the same as w.

% Output:
% w_vec--a vector that packs the structure values.

% Structure of w:
% w(i).line--a 2-element vector containing the vertices of the line. Length
% of w is the number of lines.
% w(i).g--a vector of conductance of line i, contain 6 elements.
% w(i).b--a vector of susceptance of line i, contain 6 elements.
% The 6 elements are in the order of phase aa, ab, ac, bb, bc, cc.

w_vec=[];
for i=1:length(w_in) % for each line, pack g 1st, b 2nd.
    w_vec=[w_vec;w_in(i).r];
    w_vec=[w_vec;w_in(i).x];
end
end

%%% function to transfer w structure of R and X to R_vec and X_vec
function [R_vec,X_vec]=f_wRX2Vec(w_in)
% Input: w_in, a structure of w (weight).
% Output: R_vec and X_vec. 2 vectors containing the line parameter values.
% The output parameter values will be organized in the sequence as follows:
% first order by g to b, then order by line, then order by phase 
% connection: aa, ab, ac, bb, bc, cc. The line order is the order in
% min_line_config.

% Structure of w:
% w(i).line--a 2-element vector containing the vertices of the line. Length
% of w is the number of lines.
% w(i).r--a vector of resistance of line i, contain 6 elements.
% w(i).x--a vector of reactance of line i, contain 6 elements.

num_L=length(w_in);
R_vec=[];
X_vec=[];

for i=1: num_L
    R_vec=[R_vec;w_in(i).r];
    X_vec=[X_vec;w_in(i).x];
end

end


%%% function to calculate gradient over RX, based on gradient over GB.
function grad_w_z=f_GBgrad2RXgrad(grad_w_y,w_z,w_y,d_GB_d_w)
% Input: 
% (1) grad_w_y: a structure similar to GB w (weight). It is the partial
% derivative over G and B.
% (2) w_z: current RX value structure.
% (3) w_y: current GB value structure.
% (4) Model_struct: stores necessary data for the function
% Output:
% grad_w_z: gradient over the R and X parameters.

num_L=length(grad_w_y);% number of lines
% d_GB_d_w=Model_struct.d_GB_d_w; % the d_GB_d_w has the same valueas d_RX_d_z

grad_w_z=w_z;  % initialzie structure of grad_w_z; it has the same structure as w_z.

% the gradient over an R or X is only related to the gradient over the
% corresponding same line's G and B.
for i_L=1:num_L
    
    %%% prepare data that will be repeatedly used
    g_matrix=f_6vec2symatrix(w_y(i_L).g);
%     b_matrix=f_6vec2symatrix(w_y(i_L).b);
    r_matrix=f_6vec2symatrix(w_z(i_L).r);
    x_matrix=f_6vec2symatrix(w_z(i_L).x);
    
    r_matrix_inv=inv(r_matrix);
    
    phase_pair_table=[1,1; ... % reference table for index of parameters
        1,2; ...
        1,3; ...
        2,2; ...
        2,3; ...
        3,3];
    num_pair=size(phase_pair_table,1);
    
    % d[g]_mn/d[g]^-1 for all mn (6 combinations)
    d_g_mn_d_g_inv=[];
    for i_pair=1:num_pair
        d_g_mn_d_g_inv(i_pair).M=f_dA_inv(g_matrix,phase_pair_table(i_pair,1),phase_pair_table(i_pair,2));
    end    
    % d[r]^-1_mn/d[r] for all mn (6 combinations)
    d_r_inv_mn_d_r=[];
    for i_pair=1:num_pair
        d_r_inv_mn_d_r(i_pair).M=f_dA_inv(r_matrix_inv,phase_pair_table(i_pair,1),phase_pair_table(i_pair,2));
    end
    
    for i_z=1:num_pair % now calcualte the gradient over R and X of each phase pair on line i_L
        
        temp_v=zeros(6,1);
        for temp_idx=1:6 % calculate d [r]^-1_mn/d an r element
            temp_v(temp_idx)=(d_r_inv_mn_d_r(temp_idx).M(:))'*d_GB_d_w(i_z).M(:);  % this is equivalent to a trace calculation
        end
        d_r_inv_d_1r=f_6vec2symatrix(temp_v);
        
        %%% First calculate gradient over R
        % d [g]^-1 / d r
        temp_part1=d_GB_d_w(i_z).M;
        temp_part2=0;
        temp_part3=x_matrix*d_r_inv_d_1r*x_matrix;
        temp_part4=0;
        d_g_inv_d_r=temp_part1+temp_part2+temp_part3+temp_part4;
        
        % d [g]/ d r over one r element
        d_g_d_r_vec=zeros(num_pair,1);
        for i_g=1:num_pair
            d_g_d_r_vec(i_g)=d_g_mn_d_g_inv(i_g).M(:)'*d_g_inv_d_r(:);
        end
        d_g_d_r_M=f_6vec2symatrix(d_g_d_r_vec);
        
        % d [b] / d r over one r element
        temp_part1=-d_g_d_r_M*x_matrix*r_matrix_inv;
        temp_part2=0;
        temp_part3=-g_matrix*x_matrix*d_r_inv_d_1r;
        d_b_d_r_M=temp_part1+temp_part2+temp_part3;
        d_b_d_r_vec=f_symatrix2vec6(d_b_d_r_M);
        
        % assign gradient value (chain rule)
        grad_w_z(i_L).r(i_z)=grad_w_y(i_L).g'*d_g_d_r_vec+grad_w_y(i_L).b'*d_b_d_r_vec;
        
        %%% Second, after R, calculate gradient over X
        % d [g]^-1 / d x over one x element
        temp_part1=0;
        temp_part2=d_GB_d_w(i_z).M*r_matrix_inv*x_matrix;
        temp_part3=0;
        temp_part4=x_matrix*r_matrix_inv*d_GB_d_w(i_z).M;
        d_g_inv_d_x=temp_part1+temp_part2+temp_part3+temp_part4;
        
        % d [g]/ d x over one x element
        d_g_d_x_vec=zeros(num_pair,1);
        for i_g=1:num_pair
            d_g_d_x_vec(i_g)=d_g_mn_d_g_inv(i_g).M(:)'*d_g_inv_d_x(:);
        end
        d_g_d_x_M=f_6vec2symatrix(d_g_d_x_vec);
        
        % d [b] / d x over one x element
        temp_part1=-d_g_d_x_M*x_matrix*r_matrix_inv;
        temp_part2=-g_matrix*d_GB_d_w(i_z).M*r_matrix_inv;
        temp_part3=0;
        d_b_d_x_M=temp_part1+temp_part2+temp_part3;
        d_b_d_x_vec=f_symatrix2vec6(d_b_d_x_M);
        
        % assign gradient value (chain rule)
        grad_w_z(i_L).x(i_z)=grad_w_y(i_L).g'*d_g_d_x_vec+grad_w_y(i_L).b'*d_b_d_x_vec;
    end
    
end


end

%%% function to transfer 6-by-1 vector to symmetric matrix
function M=f_6vec2symatrix(vec)
if length(vec)~=6
    disp('Error: vector length is not 6')
    return;
end

M=[vec(1),vec(2),vec(3); ...
    vec(2), vec(4), vec(5); ...
    vec(3), vec(5), vec(6)];
end

%%% function to transfer a symmetric matrix to 6-by-1 vector
function vec=f_symatrix2vec6(M)
if size(M,1)~=3 || size(M,2)~=3
    disp('Error: matrix size is not 3-by-3')
    return;
end

vec=[M(1,1);M(1,2);M(1,3);M(2,2);M(2,3);M(3,3)];

end


%%% function to update parameter value (for RX weight)
function w_out=f_update_wRX(w_in,w_gradient,step_size)
% Input:
% w_in--the weight/parameter. In our problem this is the set of line
% parameters. The structure is the same as w.
% w_gradient--the gradient structure representing the gradient of the
% weight, has the same structure as w.
% step_size--the step size. Note that in gradient descent, the step size
% input here should be a negative value. The step size is a multiplier to
% multiply the gradient.

% Output:
% w_out--the new weight structure updated by the step size.

% Structure of w:
% w(i).line--a 2-element vector containing the vertices of the line. Length
% of w is the number of lines.
% w(i).g--a vector of conductance of line i, contain 6 elements.
% w(i).b--a vector of susceptance of line i, contain 6 elements.
% The 6 elements are in the order of phase aa, ab, ac, bb, bc, cc.

w_out=w_in;
for i=1:length(w_in)
    w_out(i).r=w_out(i).r+w_gradient(i).r*step_size;
    w_out(i).x=w_out(i).x+w_gradient(i).x*step_size;
end

end

%%% function to reorganize a weight strucutre to a vector (for RX weight)
function w_vec=f_w_struct2vecRX(w_in)
% Input:
% w_in--the weight/parameter. In our problem this is the set of line
% parameters. The structure is the same as w.

% Output:
% w_vec--a vector that packs the structure values.

% Structure of w:
% w(i).line--a 2-element vector containing the vertices of the line. Length
% of w is the number of lines.
% w(i).g--a vector of conductance of line i, contain 6 elements.
% w(i).b--a vector of susceptance of line i, contain 6 elements.
% The 6 elements are in the order of phase aa, ab, ac, bb, bc, cc.

w_vec=[];
for i=1:length(w_in) % for each line, pack g 1st, b 2nd.
    w_vec=[w_vec;w_in(i).r];
    w_vec=[w_vec;w_in(i).x];
end
end

%%% function to calculate the gradient vector, given the current line
%%% parameter vector
function [gradient_G,gradient_B]=f_gradient_from_vecGB(G_vec,B_vec,Model_struct)
% Input: G_vec, B_vec: line parameter matrix.
% Model_struct: other parameters of the model
% Output: the gradient_G, gradient_B--the partial derivative vector to each
% line, each phase pair, of line conductance and susceptance.

Lambda_G=f_G_Vec2Matrix(G_vec);
Lambda_B=f_G_Vec2Matrix(B_vec);

[gradient_G,gradient_B]=f_gradientGB(Lambda_G,Lambda_B,Model_struct);
end


%% 2020-08-04: add function for Adam optimization
%%% function to update the parameters by Adam method
function [w_z_new,averageGrad_new,averageSqGrad_new] = f_my_Adam_update_RX(w_z,grad_w,averageGrad,averageSqGrad,iteration,parameter)
% Input: 
% (1) w_z: weight parameter in R X.
% (2) grad_w: gradient of w_z, same structure as w_z;
% (3) averageGrad: average gradient value over time. Same structure as w_z.
% (4) averageSqGrad: average squared gradient value over time. Same structure as w_z.
% (5) iteration number. First iteration number should be 1, not 0.
% (6) parameter: a structure containing parameters of step_size, beta_1,beta_2,epsilon: parameters for the Adam method.

% Output:
% Same definition as the input, but updated.


step_size=parameter.step_size;
beta_1=parameter.beta_1;
beta_2=parameter.beta_2;
epsilon=parameter.epsilon;

w_z_new=w_z;
averageGrad_new=averageGrad;
averageSqGrad_new=averageSqGrad;

for i=1:length(w_z)
    
    % deal with r first
    averageGrad_new(i).r=beta_1*averageGrad(i).r+(1-beta_1)*grad_w(i).r;
    averageSqGrad_new(i).r=beta_2*averageSqGrad(i).r+(1-beta_2)*[grad_w(i).r].^2;
    temp_v1=averageGrad_new(i).r/(1-beta_1^iteration);
    temp_v2=averageSqGrad_new(i).r/(1-beta_2^iteration);
    w_z_new(i).r=w_z(i).r-step_size*temp_v1./(sqrt(temp_v2)+epsilon);
    
    % deal with x then
    averageGrad_new(i).x=beta_1*averageGrad(i).x+(1-beta_1)*grad_w(i).x;
    averageSqGrad_new(i).x=beta_2*averageSqGrad(i).x+(1-beta_2)*[grad_w(i).x].^2;
    temp_v1=averageGrad_new(i).x/(1-beta_1^iteration);
    temp_v2=averageSqGrad_new(i).x/(1-beta_2^iteration);
    w_z_new(i).x=w_z(i).x-step_size*temp_v1./(sqrt(temp_v2)+epsilon);
end

end


%% 2020-08-11: add function, for voltage difference model
%%% function to caculate voltage difference from the actual volt value
function [output_diff]=f_output2diff(output,T_diff_mat)
%Input:
% (1) output--a structure of actual value.
% (2) T_diff_mat: two columns, column 1 and column 2 are two sets of time
% index. The value difference is column 2's time minus column 1's time.
% Output:
% (1) output_diff: a structure of time difference value

output_diff=[];
for i=1:size(T_diff_mat,1)
    output_diff(i).meter_list=output(i).meter_list;
    output_diff(i).value=output(T_diff_mat(i,2)).value-output(T_diff_mat(i,1)).value;
end
end

%%% function of the BACKWARD, RX, of volt difference nonlinear GNN model
function [grad_w_z,iter_num]=f_BACKWARD_RX_diff(T_diff_mat,in_state,output_correct,output_model,Model,label,w_z,d_GB_d_w,source_node,threshold_ratio)
% Input:
% T_diff_mat: time index table, time difference is second column's time
% minus first column's time.
% in_state--input state structure containing the nodal volt real and
% imaginary part.
% output_correct--the same structure as output, but contains the actual
% output measurements.
% output_model--structure containing how the state is mapped to the output.
% Model--the structure that stores the needed Z_nn matrix and Y_nk matrics for 
% each node's local transition function.
% label--label structure containing the nodal real and reactive power
% injection.
% w_z--the weight/parameter. In our problem this is the set of line
% parameters in forms of R X
% d_GB_d_w--a structure storing the fixed derivative matrices for line
% parameter.d_GB_d_w(i).M is a 3-by-3 matrix. i=1~6, representing 6 phase
% types: aa, ab, ac, bb, bc, cc. G and B share the same patter, so we only 
% i=1~6 to represent these cases.
% source_node--the source node

% Output
% grad_w_z: gradient structure over R X.


% Strucutre of in_state or complex state
% complex_state(i).node--name of node i.
% complex_state(i).real--a 3-by-T matrix of nodal volt real part of node i of all
% time through T.
% complex_state(i).imag-a 3-by-T matrix of nodal volt imaginary part of node i of
% all time through T.

% Structure of output_model:
% output_model(i).meter--the name of the meter i.
% output_model(i).node--the node which the meter is connected to.
% output_model(i).phase--the phase of the smart meter i.
% 1,2,3,12,23,31,301,302,303 represents phase A, B, C, AB, BC, CA, 3-phase
% A, 3-phase B, and 3-phase C.
% output_model(i).f_v--the vector model of meter i. f_v is 1-by-3, a
% selection vetor of 1,-1, and 0 indicating which phase has which sign. f_v
% is the same for both real and imaginary part of a state of one particular
% meter.

% Structure of the Model:
% Model(i).node--node name of node i.
% Model(i).neighbor--vector of node i's neighbors.
% Model(i).rZ_nn--Z_nn real matrix for node i's local transition function
% (6-by-6).
% Model(i).rY_nk--structure of Y_nk real matrix for node i's local transition function.
% Model(i).rY_nk(j).M--the Y_nk real matrix corresponding to node i's
% neighbor node j (6-by-6).

% Structure of label:
% label(i).node--name of node i.
% label(i).P--a 3-by-T matrix of nodal real power of node i of all
% time through T.
% label(i).Q--a 3-by-T matrix of nodal reactive power of node i of
% all time through T.

% Structure of w:
% w(i).line--a 2-element vector containing the vertices of the line. Length
% of w is the number of lines.
% w(i).g--a vector of conductance of line i, contain 6 elements.
% w(i).b--a vector of susceptance of line i, contain 6 elements.
% The 6 elements are in the order of phase aa, ab, ac, bb, bc, cc.

max_iter=1e4;
w_y=f_RX2GB_struct(w_z); % transfer RX to GB.

node_list=[in_state.node]; % list of node
I1=find(node_list==source_node);
nonsource_node_list=node_list;
nonsource_node_list(I1)=[];% nonsource node list
N=length(nonsource_node_list); % number of nonsource node.

% T=size(in_state(1).real,2);
T_diff=size(T_diff_mat,1);% number of time difference

output=f_GNN_output(in_state,output_model);
output_diff=f_output2diff(output,T_diff_mat);
grad_A=f_grad_A(in_state,label,Model,source_node);
grad_A_diff=f_grad_A_diff(grad_A,T_diff_mat); %calculate the grad_A of the diff model
grad_b_diff=f_grad_b_diff(T_diff_mat,output,output_correct,output_model,in_state,source_node);


%%% calcualte z
% initialize z
z_iter=[];
for i_t=1:T_diff
    z_iter(i_t).M=zeros(1,12*N);
end

iter_num=0; % count the number of iterations.
stop_flag=0;
while iter_num<=max_iter && stop_flag==0
    iter_num=iter_num+1;
    
    old_z=z_iter;
    
    for i_t=1:T_diff
        z_iter(i_t).M=z_iter(i_t).M*grad_A_diff(i_t).M+grad_b_diff(i_t).vec;
    end
    
    temp_new=[z_iter.M];
    temp_old=[old_z.M];
    norm_update=(norm(temp_new-temp_old,'fro')^2)/(T_diff*length(z_iter(1).M));
    norm_old=(norm(temp_old,'fro')^2)/(T_diff*length(z_iter(1).M));
    if norm_update<norm_old*threshold_ratio
        stop_flag=1; % if the MSE of state update is smaller than a ratio of previous state's MSE, then stop iteration.
    end
end

% % or calculate z by using theoretical inverse multiplication (this method causes singular matrix in simulation, so abandon it)
% iter_num=0;
% para_number=6*N;
% z_iter=[];% theoretical converged z
% I_a=eye(para_number);
% for i_t=1:T
%     z_iter(i_t).M=grad_b(i_t).vec*inv(I_a-grad_A(i_t).M); % This is quicker than iteration, but if condition number is large, use iteration instead.
% end

% % for debug or replace code
% para_number=6*N;
% z_converge=[];% theoretical converged z
% for i_t=1:T
%     z_converge(i_t).M=grad_b(i_t).vec*inv(eye(para_number)-grad_A(i_t).M); % I guess using iteration to calculate z is to avoid inverse.
% end
% i_t=1;
% figure;plot(1:36,[(z_iter(i_t).M)',(z_converge(i_t).M)'])
% legend('iter result z','theoretical z')
% % results show that the z calculated by iteration is very close to the
% % thoretical value using matrix inverse.


grad_dF_dw=f_grad_dF_dw(in_state,Model,label,w_y,d_GB_d_w,source_node);
grad_d_diff=[];
for i_t=1:T_diff
    grad_d_diff(i_t).vec=z_iter(i_t).M*[grad_dF_dw(T_diff_mat(i_t,1)).M;grad_dF_dw(T_diff_mat(i_t,2)).M];
end

d_e_w_d_w=grad_d_diff;

grad_w_vec=0;
for i_t=1:T_diff
    grad_w_vec=grad_w_vec+d_e_w_d_w(i_t).vec;
end
grad_w_vec=grad_w_vec/T_diff;

grad_w_y=w_y;
for i_w=1:length(w_y)
    temp_idx=(i_w-1)*12;
    grad_w_y(i_w).g=grad_w_vec(temp_idx+[1:6])';
    grad_w_y(i_w).b=grad_w_vec(temp_idx+[7:12])';
end

grad_w_z=f_GBgrad2RXgrad(grad_w_y,w_z,w_y,d_GB_d_w); % chain rule to calcualte gradient over R X.

end

%%% function to transfer grad_A of actual value model to the volt diff
%%% model
function grad_A_diff=f_grad_A_diff(grad_A,T_diff_mat)
% Input:
% T_diff_mat: time index table, time difference is second column's time
% minus first column's time.
% Output: grad_A_diff--the matrix of each time stamp is 12N-by-12N, (N the 
% number of nonsource nodes) at time t.
T_diff=size(T_diff_mat,1);
node_list=grad_A(1).node_list;
grad_A_diff=[];
for i=1:T_diff
    grad_A_diff(i).node_list=node_list;
    grad_A_diff(i).M=blkdiag(grad_A(T_diff_mat(i,1)).M,grad_A(T_diff_mat(i,2)).M);
end
end

%%% function to calculate b(t) for gradient, volt diff version.
function grad_b_diff=f_grad_b_diff(T_diff_mat,output,output_correct,output_model,in_state,source_node)
% Input:
% T_diff_mat: time index table, time difference is second column's time
% minus first column's time.
% output--a structure of output. output(t).meter_list is a vector containing the
% list of meter names (same for all t). output(t).value is a vector of
% output value (in this case the volt measurement) matching to each meter
% at time t.
% output_correct--the same structure as output, but contains the actual
% output measurements.
% output_model--structure containing how the state is mapped to the output.
% in_state--input state structure containing the nodal volt real and imaginary part.
% source_node--the source node.

% Output:
% grad_b--a structure containg d e_w(t)/d o(t) * d G(x(t)/d x(t).
% grad_b(t).node_list is the list of nodes. grad_b(t).vec is
% a 1-by-12N vector (N the number of nonsource nodes) at time t.


% Structure of output_model:
% output_model(i).meter--the name of the meter i.
% output_model(i).node--the node which the meter is connected to.
% output_model(i).phase--the phase of the smart meter i.
% 1,2,3,12,23,31,301,302,303 represents phase A, B, C, AB, BC, CA, 3-phase
% A, 3-phase B, and 3-phase C.
% output_model(i).f_v--the vector model of meter i. f_v is 1-by-3, a
% selection vetor of 1,-1, and 0 indicating which phase has which sign. f_v
% is the same for both real and imaginary part of a state of one particular
% meter.

% Strucutre of in_state or complex state
% complex_state(i).node--name of node i.
% complex_state(i).real--a 3-by-T matrix of nodal volt real part of node i of all
% time through T.
% complex_state(i).imag-a 3-by-T matrix of nodal volt imaginary part of node i of
% all time through T.
%----------------------------------------------------------

node_list=[in_state.node]; % list of node
I1=find(node_list==source_node);
nonsource_node_list=node_list;
nonsource_node_list(I1)=[];% nonsource node list
N=length(nonsource_node_list); % number of nonsource node.
T=size(in_state(1).real,2);
T_diff=size(T_diff_mat,1);
meter_list=[output(1).meter_list];
output_model_meter_list=[output_model.meter];
M=length(output(1).meter_list);

de_do=zeros(T,M);
for i_t=1:T % initialize grad_b, and prepare de/do.
    grad_g(i_t).node_list=nonsource_node_list; % grad_g stores M-by-6N d_o/d_x
    grad_g(i_t).M=zeros(M,6*N);
    temp_v=2/M*(output(i_t).value-output_correct(i_t).value);
    de_do(i_t,:)=temp_v';% This is 1-by-M.
end

de_do_diff=zeros(T_diff,M); % volt diff version
for i_t=1:T_diff
    de_do_diff(i_t,:)=de_do(T_diff_mat(i_t,2),:)-de_do(T_diff_mat(i_t,1),:);
end

% filling the actual grad_b
for i_meter=1:M % for each meter
    target_meter=meter_list(i_meter);
    I1=find(output_model_meter_list==target_meter);
    target_node=output_model(I1).node; % the node connected to the target_meter
    f_v=output_model(I1).f_v;
    phase_type=output_model(I1).phase;
    
    I1=find(node_list==target_node);
    temp_real=f_v*in_state(I1).real; % 1-by-T
    temp_imag=f_v*in_state(I1).imag;
    
    denomenator=sqrt(temp_real.^2+temp_imag.^2);
    
    idx_node=find(nonsource_node_list==target_node);
    %%% below the calculation is based on
    %%% d_e/d_x_k=sum_over_m(d_e/d_o_m*d_g_m/d_x_r). More code but less
    %%% calculation.
    switch phase_type % filling grad_b based on phase type
        case 1 % phase A
            idx1=1;
            
            real_grad=temp_real./denomenator; % d_gm/d_real
            imag_grad=temp_imag./denomenator; % d_gm/d_imag
            for i_t=1:T % update each time
                temp_idx=(idx_node-1)*6+idx1; % idx of real of target node
                grad_g(i_t).M(i_meter,temp_idx)=real_grad(i_t);
                temp_idx=(idx_node-1)*6+idx1+3; % idx of imag of target node
                grad_g(i_t).M(i_meter,temp_idx)=imag_grad(i_t);
            end
            
        case 2 % phase B
            idx1=2;
            
            real_grad=temp_real./denomenator; % d_gm/d_real
            imag_grad=temp_imag./denomenator; % d_gm/d_imag
            for i_t=1:T % update each time
                temp_idx=(idx_node-1)*6+idx1; % idx of real of target node
                grad_g(i_t).M(i_meter,temp_idx)=real_grad(i_t);
                temp_idx=(idx_node-1)*6+idx1+3; % idx of imag of target node
                grad_g(i_t).M(i_meter,temp_idx)=imag_grad(i_t);
            end
            
        case 3 % phase C
            idx1=3;
            
            real_grad=temp_real./denomenator; % d_gm/d_real
            imag_grad=temp_imag./denomenator; % d_gm/d_imagg
            for i_t=1:T % update each time
                temp_idx=(idx_node-1)*6+idx1; % idx of real of target node
                grad_g(i_t).M(i_meter,temp_idx)=real_grad(i_t);
                temp_idx=(idx_node-1)*6+idx1+3; % idx of imag of target node
                grad_g(i_t).M(i_meter,temp_idx)=imag_grad(i_t);
            end
            
        case 12 % phase AB
            idx1=1; 
            idx2=2;
            
            real_grad=temp_real./denomenator; % d_gm/d_real
            imag_grad=temp_imag./denomenator; % d_gm/d_imag
            for i_t=1:T % update each time
                temp_idx=(idx_node-1)*6+idx1; % idx of real of target node (first phase)
                grad_g(i_t).M(i_meter,temp_idx)=real_grad(i_t);
                temp_idx=(idx_node-1)*6+idx1+3; % idx of imag of target node
                grad_g(i_t).M(i_meter,temp_idx)=imag_grad(i_t);
                
                % below pay attention to idx and sign before de_do. (second pahse)
                temp_idx=(idx_node-1)*6+idx2; % idx of real of target node
                grad_g(i_t).M(i_meter,temp_idx)=-real_grad(i_t);
                temp_idx=(idx_node-1)*6+idx2+3; % idx of imag of target node
                grad_g(i_t).M(i_meter,temp_idx)=-imag_grad(i_t);
            end
            
        case 23 % phase BC
            idx1=2; 
            idx2=3;
            
            real_grad=temp_real./denomenator; % d_gm/d_real
            imag_grad=temp_imag./denomenator; % d_gm/d_imag
            for i_t=1:T % update each time
                temp_idx=(idx_node-1)*6+idx1; % idx of real of target node (first phase)
                grad_g(i_t).M(i_meter,temp_idx)=real_grad(i_t);
                temp_idx=(idx_node-1)*6+idx1+3; % idx of imag of target node
                grad_g(i_t).M(i_meter,temp_idx)=imag_grad(i_t);
                
                % below pay attention to idx and sign before de_do. (second pahse)
                temp_idx=(idx_node-1)*6+idx2; % idx of real of target node
                grad_g(i_t).M(i_meter,temp_idx)=-real_grad(i_t);
                temp_idx=(idx_node-1)*6+idx2+3; % idx of imag of target node
                grad_g(i_t).M(i_meter,temp_idx)=-imag_grad(i_t);
            end
                 
        case 31 % phase CA
            idx1=3;
            idx2=1;
            
            real_grad=temp_real./denomenator; % d_gm/d_real
            imag_grad=temp_imag./denomenator; % d_gm/d_imag
            for i_t=1:T % update each time
                temp_idx=(idx_node-1)*6+idx1; % idx of real of target node (first phase)
                grad_g(i_t).M(i_meter,temp_idx)=real_grad(i_t);
                temp_idx=(idx_node-1)*6+idx1+3; % idx of imag of target node
                grad_g(i_t).M(i_meter,temp_idx)=imag_grad(i_t);
                
                % below pay attention to idx and sign before de_do. (second pahse)
                temp_idx=(idx_node-1)*6+idx2; % idx of real of target node
                grad_g(i_t).M(i_meter,temp_idx)=-real_grad(i_t);
                temp_idx=(idx_node-1)*6+idx2+3; % idx of imag of target node
                grad_g(i_t).M(i_meter,temp_idx)=-imag_grad(i_t);
            end
            
        case 301 % phase ABC-A
            idx1=1;
            
            real_grad=temp_real./denomenator; % d_gm/d_real
            imag_grad=temp_imag./denomenator; % d_gm/d_imag
            for i_t=1:T % update each time
                temp_idx=(idx_node-1)*6+idx1; % idx of real of target node
                grad_g(i_t).M(i_meter,temp_idx)=real_grad(i_t);
                temp_idx=(idx_node-1)*6+idx1+3; % idx of imag of target node
                grad_g(i_t).M(i_meter,temp_idx)=imag_grad(i_t);
            end
            
        case 302 % phase ABC-B
            idx1=2;
            
            real_grad=temp_real./denomenator; % d_gm/d_real
            imag_grad=temp_imag./denomenator; % d_gm/d_imag
            for i_t=1:T % update each time
                temp_idx=(idx_node-1)*6+idx1; % idx of real of target node
                grad_g(i_t).M(i_meter,temp_idx)=real_grad(i_t);
                temp_idx=(idx_node-1)*6+idx1+3; % idx of imag of target node
                grad_g(i_t).M(i_meter,temp_idx)=imag_grad(i_t);
            end
            
        case 303 % phase ABC-C
            idx1=3;
            
            real_grad=temp_real./denomenator; % d_gm/d_real
            imag_grad=temp_imag./denomenator; % d_gm/d_imag
            for i_t=1:T % update each time
                temp_idx=(idx_node-1)*6+idx1; % idx of real of target node
                grad_g(i_t).M(i_meter,temp_idx)=real_grad(i_t);
                temp_idx=(idx_node-1)*6+idx1+3; % idx of imag of target node
                grad_g(i_t).M(i_meter,temp_idx)=imag_grad(i_t);
            end
    end
end

for i_t=1:T_diff
    grad_b_diff(i_t).node_list=nonsource_node_list; % grad_b stores 1-by-12N vector
    grad_b_diff(i_t).vec=de_do_diff(i_t,:)*[-grad_g(T_diff_mat(i_t,1)).M,grad_g(T_diff_mat(i_t,2)).M];
end

end

%% 2020-08-16: function added for algorithms improvement

%%% function to check if a weight RX structure contains NaN values
function NaNinf_flag=f_w_NaNinf_check(w_in)
NaNinf_flag=0;
for i=1:length(w_in)
    I1=find(isnan(w_in(i).r) | isinf(w_in(i).r),1);
    if ~isempty(I1)
        NaNinf_flag=1;
        break
    end
    
    I1=find(isnan(w_in(i).x) | isinf(w_in(i).x),1);
    if ~isempty(I1)
        NaNinf_flag=1;
        break
    end
end
end

%% 2020-08-22: add functions to enable range limit of parameters.
%%% function to generate parameter weight limit based on erroneous weight
%%% and error range
% 2020-10-16: modified to deal with negative parameter values.
function [w_min,w_max]=f_w_generate_limit(w_start,error_ratio)
% Input:
% w_start: the weight structure containing the initial erroneous weight
% parameters.
% error_ratio: the error ratio: 0.5 means 50%.
% Output:
% w_min: lower bound of weight value
% w_max: higher bound of weight value.

% initialize values
w_min=w_start;
w_max=w_start;

for i=1:length(w_start)
    for i2=1:length(w_start(i).r)
        % r
        if w_start(i).r(i2)>=0
            w_min(i).r(i2)=w_start(i).r(i2)/(1+error_ratio);
            w_max(i).r(i2)=w_start(i).r(i2)/(1-error_ratio);
        else
            w_min(i).r(i2)=w_start(i).r(i2)/(1-error_ratio);
            w_max(i).r(i2)=w_start(i).r(i2)/(1+error_ratio);
        end
        % x
        if w_start(i).x(i2)>=0
            w_min(i).x(i2)=w_start(i).x(i2)/(1+error_ratio);
            w_max(i).x(i2)=w_start(i).x(i2)/(1-error_ratio);
        else
            w_min(i).x(i2)=w_start(i).x(i2)/(1-error_ratio);
            w_max(i).x(i2)=w_start(i).x(i2)/(1+error_ratio);
        end
            
    end
end

end

%%% function to calculate the projection into the limit range domain
function w_proj=f_w_limit_project(w_in,w_min,w_max)
% Input:
% w_in: the weight structure.
% w_min: lower bound of weight value
% w_max: higher bound of weight value.
% Output:
% w_proj--the projection of w_in within the min max limit

w_proj=w_in; %initialization
for i=1:length(w_in)
    w_proj(i).r=max(w_proj(i).r,w_min(i).r); % apply lower bound on r
    w_proj(i).r=min(w_proj(i).r,w_max(i).r); % apply higher bound on r
    
    w_proj(i).x=max(w_proj(i).x,w_min(i).x); % apply lower bound on x
    w_proj(i).x=min(w_proj(i).x,w_max(i).x); % apply higher bound on x
end

end

%%% function to calculate the difference between two weight structure
function w_diff=f_2w_diff(w_1,w_2)
% Output:
% w_diff=w_1-w_2.

w_diff=w_1;
for i=1:length(w_diff)
    w_diff(i).r=w_1(i).r-w_2(i).r;
    w_diff(i).x=w_1(i).x-w_2(i).x;
end
end

%%% function to calculate the minimum ratio between two weight structure
function min_ratio=f_2w_min_ratio(w_1,w_2)
% Output:
% mean_ratio=min(w_1/w_2).

epsilon=1e-50; % to avoid singularity, or numerical error by divide a very small value over another very small value. (and in such case, the output ratio of this funciton should be 1.)
temp_v=[];
for i=1:length(w_1)
    temp_v=[temp_v;abs(w_1(i).r)./(abs(w_2(i).r)+epsilon)]; % w_1 and w_2 should have the same corresponding sign, so using abs() will not affect the result.
    temp_v=[temp_v;abs(w_1(i).x)./(abs(w_2(i).x)+epsilon)];
end
min_ratio=min(abs(temp_v),[],'omitnan'); % temp_v should be non negative, but to avoid potential errors, use abs()
min_ratio=max(min_ratio,1); % the ratio should not be smaller than 1 (avoid numerical errors).
end

%% 2020-08-30: add functions to enable Gaussian prior of line parameters.
%%% function to generate parameter weight limit of +-inf, i.e., a unlimited
%%% limit.
function [w_min,w_max]=f_w_generate_inf_limit(w_start)
% Input:
% w_start: the weight structure containing the initial erroneous weight
% parameters.
% w_min: lower bound of weight value
% w_max: higher bound of weight value.

% initialize values
w_min=w_start;
w_max=w_start;

use_size=size(w_start(1).r);

for i=1:length(w_start)
    w_min(i).r=-inf(use_size);
    w_min(i).x=-inf(use_size);
    
    w_max(i).r=inf(use_size);
    w_max(i).x=inf(use_size);
end

end

%%% function that generates the prior Gaussian distribution mean and sigma
function prior_dist=f_generate_prior_dist(w_start,error_ratio)
% Input:
% (1) w_start--the parameter structure.
% (2) error_ratio--the error ratio that is used to generate initial
% erroneous parameter.
% Output:
% prior_dist--prior Gaussian distribution data. Contains 2 structure, mean,
% and sigma_square. Each has the same structure as the w_start.

mean_struct=w_start; %initialize structure.
var_struct=w_start;

for i=1:length(w_start)
    var_struct(i).r=(w_start(i).r*error_ratio/3).^2;
    var_struct(i).x=(w_start(i).x*error_ratio/3).^2;
end

prior_dist.mean=mean_struct;
prior_dist.var=var_struct;
end

%%% funtion that calculates the regularization loss by Gaussian prior.
function loss=f_loss_prior(w_in,prior_dist)
% Input:
% (1) w_in--weight structure.
% (2) prior_dist--the structure containing parameter mean and variance.

loss=0;

for i=1:length(w_in)
    loss=loss+sum( (w_in(i).r-prior_dist.mean(i).r).^2./prior_dist.var(i).r );
    loss=loss+sum( (w_in(i).x-prior_dist.mean(i).x).^2./prior_dist.var(i).x );
end
end


%%% function that calculates the gradient of prior regularization
function grad_w=f_prior_grad(w_in,prior_dist)
% Input:
% (1) w_in--weight structure.
% (2) prior_dist--the structure containing parameter mean and variance.
% Output:
% grad_w: the gradient w.r.t. the parameter's summed square error with
% mean, normalized by the variance. It has the same structure as the
% weight.

grad_w=w_in; %initialize the gradient structure.

for i=1:length(w_in)
    grad_w(i).r=2*(w_in(i).r-prior_dist.mean(i).r)./prior_dist.var(i).r;
    grad_w(i).x=2*(w_in(i).x-prior_dist.mean(i).x)./prior_dist.var(i).x;
end

end


%%% function that add two weight structre's corresponding terms with
%%% multipliers.
function w_out=f_add2w(w1,w2,m1,m2)
% Input:
% (1) w1, and w2--two weight structure.
% (2) m1, and m2--two multipliers.
% Output:
% w_out--the sumed weight structure.

w_out=w1; %initialize the structure

for i=1:length(w_out)
    w_out(i).r=m1*w1(i).r+m2*w2(i).r;
    w_out(i).x=m1*w1(i).x+m2*w2(i).x;
end
end


%% 2020-09-20: functions used to improve efficiency.

%%% a function that transfer original state structure to a structure that
%%% uses more matrices.
function state_mat=f_state2state_mat(in_state)
% Input:
% in_state--the original complex state structure.
% Output:
% state_mat--the new state structure that uses more matrices.

% Strucutre of complex state
% complex_state(i).node--name of node i.
% complex_state(i).real--a 3-by-T matrix of nodal volt real part of node i of all
% time through T.
% complex_state(i).imag-a 3-by-T matrix of nodal volt imaginary part of node i of
% all time through T.

% Structur of the state_mat
% state_mat.node_list--list of node (N node)
% state_mat.real=a 3N-by-T matrix of real state part.
% state_mat.imag=a 3N-by-T matrix of complex state part.

state_mat.node_list=[in_state.node]'; % change list to column vector
N=length(state_mat.node_list);
T=size(in_state(1).real,2);
state_mat.real=zeros(3*N,T);
state_mat.imag=zeros(3*N,T);

for i=1:N
    temp_idx=(i-1)*3+(1:3);
    state_mat.real(temp_idx,:)=in_state(i).real;
    state_mat.imag(temp_idx,:)=in_state(i).imag;
end

end

%%% a function that transfer original state structure to a structure that
%%% uses is more compact
function state_compact=f_state2state_compact(in_state)
% Input:
% in_state--the original complex state structure.
% Output:
% state_mat--the new state structure that uses more matrices.

% Strucutre of complex state
% complex_state(i).node--name of node i.
% complex_state(i).real--a 3-by-T matrix of nodal volt real part of node i of all
% time through T.
% complex_state(i).imag-a 3-by-T matrix of nodal volt imaginary part of node i of
% all time through T.

% Structur of the state_compact
% state_compact.node_list--list of node (N node)
% state_compact.scalor_struct= a N-by-1 structure array, 
% state_compact.scalor_struct(i).scalor is a 6-by-T matrix of nodal volt real 
% and imaginary part of node i of all time through T.
% state_compact.complex_struct= a N-by-1 structure array, 
% state_compact.conj_struct(i).conj is a 3-by-T matrix of nodal conjugate complex volt
% of node i of all time through T.

state_compact.node_list=[in_state.node]'; % change list to column vector
N=length(state_compact.node_list);

for i=1:N
    state_compact.scalor_struct(i).scalor=[in_state(i).real;in_state(i).imag];
    state_compact.conj_struct(i).conj=in_state(i).real-1i*in_state(i).imag;
end

end


%%% function to transfer compact state structure (real and reactive part) to
%%% standard state volt and angle matrix.
function out_state=f_compact2magradmat(compact_state)
% Input:
% compact_state--input state structure containing the nodal volt in
% complex form--real and imaginary part, node_list in compact form.

% Output:
% out_state--output state structure containing the nodal volt magnitude and
% angle. Organized in a 6N x T big matrix. Each 6 rows represent a node's
% volt magnitude and angle. It contains 2 field, the node_list, and magrad_mat.

% Structur of the state_compact
% state_compact.node_list--list of node (N node)
% state_compact.scalor_struct= a N-by-1 structure array, 
% state_compact.scalor_struct(i).scalor is a 6-by-T matrix of nodal volt real 
% and imaginary part of node i of all time through T.
% state_compact.complex_struct= a N-by-1 structure array, 
% state_compact.conj_struct(i).conj is a 3-by-T matrix of nodal conjugate complex volt
% of node i of all time through T.

out_state.node_list=compact_state.node_list;
N=length(compact_state.node_list);
T=size(compact_state.conj_struct(1).conj,2);
out_state.magrad_mat=zeros(6*N,T);

for i=1:length(compact_state.node_list)
    temp_x=conj(compact_state.conj_struct(i).conj);
    temp_idx=(i-1)*6+(1:6);
    out_state.magrad_mat(temp_idx(1:3),:)=abs(temp_x);
    out_state.magrad_mat(temp_idx(4:6),:)=angle(temp_x);
end

end

%%% function to transfer compact state structure (real and reactive part) to
%%% complex state structure.
function out_state_complex=f_compact2state_complex(compact_state)
% Input:
% compact_state--input state structure containing the nodal volt in
% complex form--real and imaginary part, node_list in compact form.

% Output:
% out_state--output state structure containing the complex nodal voltage.

% Strucutre of complex state
% complex_state(i).node--name of node i.
% complex_state(i).real--a 3-by-T matrix of nodal volt real part of node i of all
% time through T.
% complex_state(i).imag-a 3-by-T matrix of nodal volt imaginary part of node i of
% all time through T.

% Structur of the state_compact
% state_compact.node_list--list of node (N node)
% state_compact.scalor_struct= a N-by-1 structure array, 
% state_compact.scalor_struct(i).scalor is a 6-by-T matrix of nodal volt real 
% and imaginary part of node i of all time through T.
% state_compact.complex_struct= a N-by-1 structure array, 
% state_compact.conj_struct(i).conj is a 3-by-T matrix of nodal conjugate complex volt
% of node i of all time through T.

out_state_complex=[];
for i=1:length(compact_state.node_list)
    out_state_complex(i).node=compact_state.node_list(i);
    out_state_complex(i).real=compact_state.scalor_struct(i).scalor(1:3,:);
    out_state_complex(i).imag=compact_state.scalor_struct(i).scalor(4:6,:);
end

end

%%% function to sort the edges by their distance to the source node
function sort_config=f_sort_line(line_config,source_node,sort_txt)
% Input:
% (1) line_config--each row is an edge, col 1 is edge's first node (a
% number), col 2 is the edge's second node, col 3 is the edge's length.
% (2) source_node--the source_node's id number.
% (3) sort_txt--the text indicating the sorting direction. It can only be
% "ascend" or "descend".

% Output:
% (1) sort_config--sort the rows of line_config, based on line's distance
% to the source node.

node_list=unique([line_config(:,1);line_config(:,2)]);
[Lia,idx_table]=ismember(line_config(:,1:2),node_list);
source_idx=find(node_list==source_node);
G=graph(idx_table(:,1),idx_table(:,2)); % build graph
D_node=distances(G,source_idx); % distance of each node to the root.
temp_M=D_node(idx_table);
D_line=min(temp_M,[],2); % this is the distance of line to root. The distance is the minimum distance of the edge's 2 nodes to the source.


if sort_txt=='ascend' % sort the line
    [temp_x,sort_idx]=sort(D_line,'ascend');
elseif sort_txt=='descend'
    [temp_x,sort_idx]=sort(D_line,'descend');
else
    disp('Sorting direction wrong.')
end

sort_config=line_config(sort_idx,:);
end

%%% function to sort the nodes by their distance to the source node
function sort_node_list=f_sort_node(in_node_list,line_config,source_node,sort_txt)
% Input:
% (1) in_node_list--the input list of nodes.
% (2) line_config--each row is an edge, col 1 is edge's first node (a
% number), col 2 is the edge's second node, col 3 is the edge's length.
% (3) source_node--the source_node's id number.
% (4) sort_txt--the text indicating the sorting direction. It can only be
% "ascend" or "descend".

% Output:
% (1)--sort_node_list--the sorted node list based on their distance to the
% source node.

[Lia,idx_table]=ismember(line_config(:,1:2),in_node_list);
source_idx=find(in_node_list==source_node);
G=graph(idx_table(:,1),idx_table(:,2)); % build graph
D_node=distances(G,source_idx); % distance of each node to the root.

if strcmp(sort_txt,'ascend') % sort the line
    [temp_x,sort_idx]=sort(D_node,'ascend');
elseif strcmp(sort_txt,'descend')
    [temp_x,sort_idx]=sort(D_node,'descend');
else
    disp('Sorting direction wrong.')
end

sort_node_list=in_node_list(sort_idx');

end

%% 2020-09-29 functions used for sub-network parameter estimation

%%% The function to do parameter estimation for a sub-network. Using prior
%%% and range limit.
function [save_result]=f_subnet_est_NL_GNNdiff_prlm(para_set,data_set)

%%% load needed parameters (will be saved later)
alg_seed=para_set.alg_seed; % seed of SGD algorithm
rng(alg_seed);
batch_size=para_set.batch_size;
max_iter=para_set.max_iter;
threshold_ratio=para_set.threshold_ratio;
early_stop_patience=para_set.early_stop_patience; % number of epochs of no improvement, after which stop the iteration.
early_stop_threshold_ratio=para_set.early_stop_threshold_ratio; % decide improvement: at least early_stop_threshold_ratio*initial obj function value.
prior_adjust_ratio=para_set.prior_adjust_ratio; % a ratio to adjust the regularization factor.
% Backtracking linesearch parameter
initial_step_size=para_set.initial_step_size; % standard step size
alpha=para_set.alpha; % Backtracking line search parameter.
beta=para_set.beta; % Backtracking line search parameter.
% Step size adjusting parameter
step_dynamic_beta=para_set.step_dynamic_beta; % parameter to decide dynamic initial step size
min_step_threshold=para_set.min_step_threshold;% minimum initial step size when using dynamic initial step size
max_step_threshold=para_set.max_step_threshold;
% primal_var=para_set.primal_var;
subnet_id=para_set.subnet_id; % a number showing the id of the sub-network
% line weight
main_line_config=para_set.line_config;
w_start=para_set.w_start;% starting weight
w_iter=w_start;
w_correct=para_set.w_correct;% correct z weight of the subnetwork
w_max=para_set.w_max;% limit range
w_min=para_set.w_min;
% save directory/description information
folder_dir=para_set.folder_dir; % folder name to save the result
file_dir=para_set.file_dir; % saved file name
dir_basic_result=para_set.dir_basic_result;


%%% load needed data for sub network (will not be saved later)
label=data_set.label;
state_initial_complex=data_set.state_initial_complex;
source_state_complex=data_set.source_state_complex;
output_model=data_set.output_model;
T_diff_mat_full=data_set.T_diff_mat_full;
output_correct_diff=data_set.output_correct_diff;
output_correct=data_set.output_correct;
prior_dist=data_set.prior_dist;
d_GB_d_w=data_set.d_GB_d_w;

%%% other data/parameter preperation
num_sample_diff=size(label(1).conj,2)-1; % number of time difference samples
num_load=length(output_model);
source_node_list=[source_state_complex.node]';

%%% Estimate the model's volt output's variance w.r.t. the theoretical
%%% value given the correct parameter data. (This code needs modification
%%% based on different model and code.
% first is the ground truth primal_var
[out_state_complex,iter_num]=f_FORWARD_RX_partition(w_correct,state_initial_complex,label,main_line_config,threshold_ratio,source_state_complex);
output=f_GNN_output(out_state_complex,output_model);
output_diff=f_output2diff(output,T_diff_mat_full);
temp_x=f_error(output_diff,output_correct_diff);
primal_var_true=temp_x*num_sample_diff/(num_sample_diff-1); % This is the estimated variance by OpenDSS simulation.

% Estimate primal_var from basic NLGNN_diff
primal_var=f_est_primal_var_by_basic(dir_basic_result,subnet_id,alg_seed,num_sample_diff);

% initial performance
% Model_iter=f_build_transition_model_RX(w_iter,main_line_config);
[out_state_complex,iter_num]=f_FORWARD_RX_partition(w_iter,state_initial_complex,label,main_line_config,threshold_ratio,source_state_complex);
output=f_GNN_output(out_state_complex,output_model);
output_diff=f_output2diff(output,T_diff_mat_full);
error_iter=f_error(output_diff,output_correct_diff);
error_iter=error_iter+prior_adjust_ratio*primal_var/num_sample_diff/num_load*f_loss_prior(w_iter,prior_dist); % add regularizatoin term.

loss_fun_history=error_iter;
best_f=error_iter;
best_f_idx=1; % current index of best obj.
best_w=w_iter;
best_loss_fun_history=best_f;

[R_vec_correct,X_vec_correct]=f_wRX2Vec(w_correct);
[R_vec_iter,X_vec_iter]=f_wRX2Vec(w_iter);
MADR_iter=f_MADR([R_vec_iter;X_vec_iter],[R_vec_correct;X_vec_correct]);
MADR_history=[MADR_iter];

[R_vec_start,X_vec_start]=f_wRX2Vec(w_start);
MADR_start=f_MADR([R_vec_start;X_vec_start],[R_vec_correct;X_vec_correct]);

temp_x=norm([R_vec_iter;X_vec_iter]-[R_vec_correct;X_vec_correct])^2/(2*length(R_vec_iter));
MSE_history=temp_x;

w_history_record(1).w=w_iter; % a structure to record the w history
initial_step_size_history=[];% record the dynamic initial step size.
MADR_improve_history=[];

% SGD
step_iter=initial_step_size; % intial step size of each iteration
patience_count=0;

for iter_num=1:max_iter
    disp(['sub-net ',num2str(subnet_id),', alg_seed ',num2str(alg_seed),', epoch ',num2str(iter_num),'.'])
    
    temp_step_collect=[]; % a temp collection of chosen step size in this iteration.
    %     w_old=w_iter;
    
    % build batch of samples for SGD.
    temp_v=1:batch_size:num_sample_diff;
    batch_table=[]; % two columns, col 1--a batch's starting index, col 2--a batch's ending index.
    for i1=1:(length(temp_v)-1)
        batch_table=[batch_table;temp_v(i1),temp_v(i1+1)-1];
    end
    batch_table=[batch_table;temp_v(end),num_sample_diff];
    batch_num=size(batch_table,1); %number of batches.
    
    perm_idx=randperm(num_sample_diff);
    
    % optimize over a batch
    for i_batch=1:batch_num
        pick_idx=perm_idx(batch_table(i_batch,1):batch_table(i_batch,2)); % index of samples in this batch
        
        %%% TUNABLE HERE.
        %         batch_T=length(pick_idx);% this is the size of batch in this round. This will increase the weight of prior in each batch
        batch_T=num_sample_diff; % This is for another scaling trial. This will keep the weight of prior fixed for the whole data.
        
        % build time difference batch
        %         T_diff_temp=[pick_idx',pick_idx'+1];
        use_unique_idx=unique([pick_idx,pick_idx+1]);
        [Lia,T_diff_batch]=ismember([pick_idx',pick_idx'+1],use_unique_idx);
        
        % use model with batch sample data
        state_initial_batch=f_state_batch(state_initial_complex,use_unique_idx); %get the state only on the indexed time stamps.
        label_batch=f_label_batch(label,use_unique_idx);
        source_state_batch=f_state_batch(source_state_complex,use_unique_idx);
        output_correct_batch=output_correct(use_unique_idx);
        Model_batch=f_build_transition_model_RX(w_iter,main_line_config);
        
        [out_state_batch,iter_num_temp]=f_FORWARD_RX_partition(w_iter,state_initial_batch,label_batch,main_line_config,threshold_ratio,source_state_batch);
        output_batch=f_GNN_output(out_state_batch,output_model);
        output_batch_diff=f_output2diff(output_batch,T_diff_batch);
        output_correct_batch_diff=f_output2diff(output_correct_batch,T_diff_batch);
        
        error_batch=f_error(output_batch_diff,output_correct_batch_diff);
        error_batch=error_batch+prior_adjust_ratio*primal_var/batch_T/num_load*f_loss_prior(w_iter,prior_dist); % add regularizatoin term.
        
        
        [grad_w_batch,iter_num_temp]=f_BACKWARD_RX_diff_partition(T_diff_batch,out_state_batch,output_correct_batch,output_model,Model_batch,label_batch,w_iter,d_GB_d_w,source_node_list,threshold_ratio);
        grad_temp=f_prior_grad(w_iter,prior_dist); % gradient of prior
        grad_w_batch=f_add2w(grad_w_batch,grad_temp,1,prior_adjust_ratio*primal_var/batch_T/num_load);
        
        grad_vec_batch=f_w_struct2vecRX(grad_w_batch); % gradient in a vector.
        grad_norm=norm(grad_vec_batch)^2;
        
        
        % Backtracking line search
        temp_step=step_iter;
        w_temp=f_update_wRX(w_iter,grad_w_batch,-temp_step); % note: use negative step to do gradient descent.
        
        w_proj=f_w_limit_project(w_temp,w_min,w_max);
        w_diff_1=f_2w_diff(w_temp,w_iter); % this is the unlimited step
        w_diff_2=f_2w_diff(w_proj,w_iter); % this is the step after projection
        temp_min_ratio=f_2w_min_ratio(w_diff_1,w_diff_2);
        temp_step=temp_step/temp_min_ratio;
        w_temp=f_update_wRX(w_iter,grad_w_batch,-temp_step);
        w_temp=f_w_limit_project(w_temp,w_min,w_max);% this is the first line search trial.
        
        [out_state_temp,iter_num_temp,NaN_flag]=f_FORWARD_RX_partition(w_temp,state_initial_batch,label_batch,main_line_config,threshold_ratio,source_state_batch);
        output_temp=f_GNN_output(out_state_temp,output_model);
        output_temp_diff=f_output2diff(output_temp,T_diff_batch);
        
        error_temp=f_error(output_temp_diff,output_correct_batch_diff);
        error_temp=error_temp+prior_adjust_ratio*primal_var/batch_T/num_load*f_loss_prior(w_temp,prior_dist); % add regularizatoin term.
        
        
        % avoid too big steps causing f_FORWARD goes to NaN.
        % line search criterion
        % step size not extremely small
        while ((NaN_flag) || (error_temp>(error_batch-alpha*temp_step*grad_norm))) && (temp_step>1e-30)
            temp_step=temp_step*beta;
            w_temp=f_update_wRX(w_iter,grad_w_batch,-temp_step); % note: use negative step to do gradient descent.
            w_temp=f_w_limit_project(w_temp,w_min,w_max); % projection
            
            [out_state_temp,iter_num_temp,NaN_flag]=f_FORWARD_RX_partition(w_temp,state_initial_batch,label_batch,main_line_config,threshold_ratio,source_state_batch);
            if NaN_flag
                continue
            end
            output_temp=f_GNN_output(out_state_temp,output_model);
            output_temp_diff=f_output2diff(output_temp,T_diff_batch);
            error_temp=f_error(output_temp_diff,output_correct_batch_diff);
            error_temp=error_temp+prior_adjust_ratio*primal_var/batch_T/num_load*f_loss_prior(w_temp,prior_dist); % add regularizatoin term.
        end
        
        % update the parameters
        % update the parameters
        if (NaN_flag) || (error_temp>(error_batch-alpha*temp_step*grad_norm))
            % to determine the initial step size dynamically
            temp_step_collect=[temp_step_collect;temp_step];
            continue % if no suitable weight is found, skip this round
        end
        
        % to determine the initial step size dynamically
        temp_step_collect=[temp_step_collect;temp_step];
        
        w_NaNinf_flag=f_w_NaNinf_check(w_temp);
        if ~w_NaNinf_flag % this is to avoid numerical issues.
            w_iter=w_temp;
        end
    end
    
    % recrod history
    [out_state,iter_num_temp]=f_FORWARD_RX_partition(w_iter,state_initial_complex,label,main_line_config,threshold_ratio,source_state_complex);
    output=f_GNN_output(out_state,output_model);
    output_diff=f_output2diff(output,T_diff_mat_full);
    error_iter=f_error(output_diff,output_correct_diff);
    error_iter=error_iter+prior_adjust_ratio*primal_var/num_sample_diff/num_load*f_loss_prior(w_iter,prior_dist); % add regularizatoin term.
    loss_fun_history=[loss_fun_history;error_iter];
    patience_count=patience_count+1;
    
    [R_vec_iter,X_vec_iter]=f_wRX2Vec(w_iter);
    MADR_iter=f_MADR([R_vec_iter;X_vec_iter],[R_vec_correct;X_vec_correct]);
    temp_x=norm([R_vec_iter;X_vec_iter]-[R_vec_correct;X_vec_correct])^2/(2*length(R_vec_iter));
    MADR_history=[MADR_history;MADR_iter];
    MSE_history=[MSE_history;temp_x];
    
    w_history_record(iter_num+1).w=w_iter; % a structure to record the w history
    
    initial_step_size_history=[initial_step_size_history;step_iter];
    step_iter=max(median(temp_step_collect,'omitnan')*step_dynamic_beta,min_step_threshold); % decide the step_iter by the median, but also set a lower bond
    step_iter=min(step_iter,max_step_threshold); %limit step size
    
    %temp showing the MADR improve
    temp_MADR_improve=(MADR_start-MADR_iter)/MADR_start*100;
    disp(['sub-net ',num2str(subnet_id),', alg_seed ',num2str(alg_seed),', epoch ',num2str(iter_num),'. ','MADR_improve: ',num2str(temp_MADR_improve),'%.']);
    MADR_improve_history=[MADR_improve_history;temp_MADR_improve];
    
    %     % check parameter convergence
    %     new_vec=f_w_struct2vec(w_iter);
    %     old_vec=f_w_struct2vec(w_old);
    %     vec_diff=new_vec-old_vec;
    %
    %     if (norm(vec_diff)^2)<((norm(old_vec)^2)*end_threshold_ratio) % if the parameter update is small, end the iteration.
    %         break;
    %     end
    
    % check early stopping
    if error_iter<best_f % first update best loss function value
        best_f=error_iter;
        best_f_idx=iter_num+1;
        best_w=w_iter;
    end
    best_loss_fun_history=[best_loss_fun_history;best_f];
    
    if length(best_loss_fun_history)>early_stop_patience % check stopping cretria
        temp_x=1-best_loss_fun_history(end)/best_loss_fun_history(end-early_stop_patience);
        if temp_x<early_stop_threshold_ratio % if the improvement is less than such ratio, then stop.
            break
        end
    end
    
    if abs(error_iter-loss_fun_history(end-1))<1e-50 % if the objective function is not updated at all (it gets stuck).
        break
    end
    
    % save temperary result for procedure monitor
    save_result.w_history_record=w_history_record;
    save_result.loss_fun_history=loss_fun_history;
    save_result.best_loss_fun_history=best_loss_fun_history;
    save_result.MADR_history=MADR_history;
    save_result.MSE_history=MSE_history;
    save_result.initial_step_size_history=initial_step_size_history;
    save_result.patience_count=patience_count;
    save_result.w_end=best_w;
    save_result.MADR_improve=temp_MADR_improve;
    save_result.finish_flag='not finished';
    save_result.MADR_improve_history=MADR_improve_history;
    save_result.primal_var_est=primal_var; %estimated primal_var
    save_result.primal_var_true=primal_var_true;% true primal_var, for comparison
    
    save([folder_dir,'/',file_dir],'save_result','para_set'); % save the estimation result.
    
end

w_end=best_w;

[R_vec_start,X_vec_start]=f_wRX2Vec(w_start);
[R_vec_end,X_vec_end]=f_wRX2Vec(w_end);

MADR_start=f_MADR([R_vec_start;X_vec_start],[R_vec_correct;X_vec_correct]);
MADR_end=f_MADR([R_vec_end;X_vec_end],[R_vec_correct;X_vec_correct]);
MADR_improve=(MADR_start-MADR_end)/MADR_start*100;

%%% save result
save_result.w_history_record=w_history_record;
save_result.loss_fun_history=loss_fun_history;
save_result.best_loss_fun_history=best_loss_fun_history;
save_result.MADR_history=MADR_history;
save_result.MSE_history=MSE_history;
save_result.initial_step_size_history=initial_step_size_history;
save_result.patience_count=patience_count;
save_result.w_end=w_end;
save_result.MADR_improve=MADR_improve;
save_result.finish_flag='finished';
save_result.MADR_improve_history=MADR_improve_history;
save_result.primal_var_est=primal_var; %estimated primal_var
save_result.primal_var_true=primal_var_true;% true primal_var, for comparison

save([folder_dir,'/',file_dir],'save_result','para_set'); % save the estimation result.

end


%%% forward function for partition, which considers more than one source
%%% nodes.
function [out_state_compact,iter_num,NaN_flag]=f_FORWARD_RX_partition(w_z,in_state_complex,label,Line_config,threshold_ratio,source_state_complex)
% Input:
% w_z--the weight/parameter. In our problem this is the set of line
% parameters in forms of R X.
% in_state_complex--input state structure containing the nodal volt real and imaginary part. 
% In this problem, it is the initialized state.
% label--label structure containing the nodal real and reactive power
% injection.
% Line_config--a 2-column matrix showing the connection of the circuit.
% Each row is a line; each column is one vertex of the line.
% threshold_ratio --threshold to stop iteration. It is a ratio indicating how much
% is changed compared with the previous iteration.
% source_state_complex--the state of the source node, has the same structure as
% one element in the in_state_complex.

% Output:
% out_state_compact--output state structure containing the nodal volt real
% and imaginary part.
% iter_num--number of iteration to get the state converging
% NaN_flag--used to indicate iteration failure. Some wrong w may cause inf
% or NaN in this function. If so, stop the function and return NaN_flag=1.
% The very wrong w may be caused by a too large step size.

%%% Details of the structures---------------------------------
% Structure of w:
% w(i).line--a 2-element vector containing the vertices of the line. Length
% of w is the number of lines.
% w(i).g--a vector of conductance of line i, contain 6 elements.
% w(i).b--a vector of susceptance of line i, contain 6 elements.

% Strucutre of complex state
% complex_state(i).node--name of node i.
% complex_state(i).real--a 3-by-T matrix of nodal volt real part of node i of all
% time through T.
% complex_state(i).imag-a 3-by-T matrix of nodal volt imaginary part of node i of
% all time through T.

% Structure of label:
% label(i).node--name of node i.
% label(i).P--a 3-by-T matrix of nodal real power of node i of all
% time through T.
% label(i).Q--a 3-by-T matrix of nodal reactive power of node i of
% all time through T.

% Structure of out_state_complex:
% out_state_complex has the same structure as in_state_complex.

%---------------------------------------------------------------------
% some setup
max_iter=1e4; % define maximum iteration
NaN_flag=0;
in_state_compact=f_state2state_compact(in_state_complex);


% Prepare for the iteration
% T=size(label(1).P,2); % number of timestamps

% Build model based on line configuration and parameters
% Model--the structure that stores the needed A1 matrix and A2 matrics for 
% each node's local transition function.
% Structure of the Model:
% Model(i).node--node name of node i.
% Model(i).neighbor--vector of node i's neighbors.
% Model(i).A1--A1 matrix for node i's local transition function
% Model(i).A2--A2 matrix for node i's local transition function.
% Model(i).A2(j).M--the A2 matrix corresponding to node i's neighbor node
% j.

% w_y=f_RX2GB_struct(w_z); % transfer RX to GB.

Model=f_build_transition_model_RX(w_z,Line_config);
node_list_in_state=in_state_compact.node_list;
source_node_list=[source_state_complex.node]';
% source_idx=find(node_list_in_state==source_state_complex.node);

% Iteration
iter_state=in_state_compact; %initialize state
% reset each source node to the correct value;
for i=1:length(source_node_list)
    source_idx=find(node_list_in_state==source_node_list(i));
    iter_state.scalor_struct(source_idx).scalor=[source_state_complex(i).real;source_state_complex(i).imag];
    iter_state.conj_struct(source_idx).conj=source_state_complex(i).real-1i*source_state_complex(i).imag;
end

iter_num=0; % count the number of iterations.
stop_flag=0;
% iter_state_magrad=f_compact2magradmat(iter_state);
iter_state_scalor=[iter_state.scalor_struct(:).scalor];
while iter_num<=max_iter && stop_flag==0
    iter_num=iter_num+1;
    
    old_state_scalor=iter_state_scalor;
    iter_state=f_global_transition_partition(iter_state,Model,label,source_node_list);
    % reset source node to the correct value;
%     iter_state.scalor_struct(source_idx).scalor=[source_state_complex.real;source_state_complex.imag];
%     iter_state.conj_struct(source_idx).conj=source_state_complex.real-1i*source_state_complex.imag;
%     iter_state_magrad=f_compact2magradmat(iter_state);
    
    iter_state_scalor=[iter_state.scalor_struct(:).scalor];
    % debug
    xx=find(isnan(old_state_scalor),1);
    xx2=find(isinf(old_state_scalor),1);
    if ~isempty(xx) || ~isempty(xx2) % if there is NaN of inf
        debug_flag=1;
        NaN_flag=1;
        stop_flag=1;
    end
    
    temp_improve=norm(iter_state_scalor-old_state_scalor,'fro')^2;
    old_MSE=norm(old_state_scalor,'fro')^2;
    if temp_improve<old_MSE*threshold_ratio
        stop_flag=1; % if the MSE of state update is smaller than a ratio of previous state's MSE, then stop iteration.
    end
    
    % debug use, remove when in formal use
%     state_magrad_history(iter_num).state_magrad=iter_state_magrad;
    

%     % use maximum absolute value update as threshold
%     temp_1=[[iter_state(:).mag];[iter_state(:).rad]];
%     if f_state_update_MSE(iter_state,old_state)<f_state_MSE(old_state)*threshold_ratio 
%         stop_flag=1; % if the MSE of state update is smaller than a ratio of previous state's MSE, then stop iteration.
%     end
end
out_state_compact=f_compact2state_complex(iter_state);


end

%%% A global transistion function, which is a stack of all local transition
%%% function. This version is for partition work, which will not update
%%% source nodes' state.
function out_state_compact=f_global_transition_partition(in_state_compact,Model,label,source_node_list)
% Input:
% in_state_compact--input state structure containing the nodal volt real
% and imaginary part. Of compact state structure
% Model--the structure that stores the needed A1 matrix and A2 matrics for 
% each node's local transition function.
% label--label structure containing the nodal real and reactive power
% injection.
% source_node_list--a list of source nodes.

% Output:
% out_state_compact--output state structure containing the nodal volt real
% and imaginary part. Of compact state structure.

% Structur of the state_compact
% state_compact.node_list--list of node (N node)
% state_compact.scalor_struct= a N-by-1 structure array, 
% state_compact.scalor_struct(i).scalor is a 6-by-T matrix of nodal volt real 
% and imaginary part of node i of all time through T.
% state_compact.complex_struct= a N-by-1 structure array, 
% state_compact.conj_struct(i).conj is a 3-by-T matrix of nodal conjugate complex volt
% of node i of all time through T.

% Structure of the Model:
% Model(i).node--node name of node i.
% Model(i).neighbor--vector of node i's neighbors.
% Model(i).rZ_nn--Z_nn real matrix for node i's local transition function
% (6-by-6).
% Model(i).rY_nk--structure of Y_nk real matrix for node i's local transition function.
% Model(i).rY_nk(j).M--the Y_nk real matrix corresponding to node ji's
% neighbor node j (6-by-6).

% Structure of label:
% label(i).node--name of node i.
% label(i).P--a 3-by-T matrix of nodal real power of node i of all
% time through T.
% label(i).Q--a 3-by-T matrix of nodal reactive power of node i of
% all time through T.

% Structure of out_state:
% out_state has the same structure as in_state.
%---------------------------------------------------------------

out_state_compact=in_state_compact;
state_node_list=in_state_compact.node_list; %list of node names in in_state.
Model_node_list=[Model.node]; % list of node names in Model.
label_node_list=[label.node]; % list of node names in label.

for i=1:length(state_node_list)
    target_node=state_node_list(i);
    I1=find(source_node_list==target_node);
    if ~isempty(I1) %if this target node is a souce node, skip this iteraton.
        continue;
    end
    model_idx=find(Model_node_list==target_node); % find the local model for the target node.
    label_idx=find(label_node_list==target_node); % find the local model for the target node.
    % This is Gauss iteration.
    target_out_state=f_local_transition(in_state_compact,Model(model_idx),label(label_idx));% update the target node's state.
    % below try using Gauss-Seidel iteration (result not good, abandon)
%     target_out_state=f_local_transition(out_state_compact,Model(model_idx),label(label_idx));% update the target node's state.
    out_state_compact.scalor_struct(i).scalor=target_out_state.scalor; % update the state of node i.
    out_state_compact.conj_struct(i).conj=target_out_state.conj; % update the state of node i.
end

end

%%% function of the BACKWARD, RX, of volt difference nonlinear GNN model
function [grad_w_z,iter_num]=f_BACKWARD_RX_diff_partition(T_diff_mat,in_state,output_correct,output_model,Model,label,w_z,d_GB_d_w,source_node_list,threshold_ratio)
% Input:
% T_diff_mat: time index table, time difference is second column's time
% minus first column's time.
% in_state--input state structure containing the nodal volt real and
% imaginary part.
% output_correct--the same structure as output, but contains the actual
% output measurements.
% output_model--structure containing how the state is mapped to the output.
% Model--the structure that stores the needed Z_nn matrix and Y_nk matrics for 
% each node's local transition function.
% label--label structure containing the nodal real and reactive power
% injection.
% w_z--the weight/parameter. In our problem this is the set of line
% parameters in forms of R X
% d_GB_d_w--a structure storing the fixed derivative matrices for line
% parameter.d_GB_d_w(i).M is a 3-by-3 matrix. i=1~6, representing 6 phase
% types: aa, ab, ac, bb, bc, cc. G and B share the same patter, so we only 
% i=1~6 to represent these cases.
% source_node_list--the source node list.


% Output
% grad_w_z: gradient structure over R X.


% Strucutre of in_state or complex state
% complex_state(i).node--name of node i.
% complex_state(i).real--a 3-by-T matrix of nodal volt real part of node i of all
% time through T.
% complex_state(i).imag-a 3-by-T matrix of nodal volt imaginary part of node i of
% all time through T.

% Structure of output_model:
% output_model(i).meter--the name of the meter i.
% output_model(i).node--the node which the meter is connected to.
% output_model(i).phase--the phase of the smart meter i.
% 1,2,3,12,23,31,301,302,303 represents phase A, B, C, AB, BC, CA, 3-phase
% A, 3-phase B, and 3-phase C.
% output_model(i).f_v--the vector model of meter i. f_v is 1-by-3, a
% selection vetor of 1,-1, and 0 indicating which phase has which sign. f_v
% is the same for both real and imaginary part of a state of one particular
% meter.

% Structure of the Model:
% Model(i).node--node name of node i.
% Model(i).neighbor--vector of node i's neighbors.
% Model(i).rZ_nn--Z_nn real matrix for node i's local transition function
% (6-by-6).
% Model(i).rY_nk--structure of Y_nk real matrix for node i's local transition function.
% Model(i).rY_nk(j).M--the Y_nk real matrix corresponding to node i's
% neighbor node j (6-by-6).

% Structure of label:
% label(i).node--name of node i.
% label(i).P--a 3-by-T matrix of nodal real power of node i of all
% time through T.
% label(i).Q--a 3-by-T matrix of nodal reactive power of node i of
% all time through T.

% Structure of w:
% w(i).line--a 2-element vector containing the vertices of the line. Length
% of w is the number of lines.
% w(i).g--a vector of conductance of line i, contain 6 elements.
% w(i).b--a vector of susceptance of line i, contain 6 elements.
% The 6 elements are in the order of phase aa, ab, ac, bb, bc, cc.

max_iter=1e4;
w_y=f_RX2GB_struct(w_z); % transfer RX to GB.

node_list=[in_state.node]; % list of node
Lia=ismember(node_list,source_node_list);
I1=find(Lia);
nonsource_node_list=node_list;
nonsource_node_list(I1)=[];% nonsource node list
N=length(nonsource_node_list); % number of nonsource node.

% T=size(in_state(1).real,2);
T_diff=size(T_diff_mat,1);% number of time difference

output=f_GNN_output(in_state,output_model);
% output_diff=f_output2diff(output,T_diff_mat);
grad_A=f_grad_A_partition(in_state,label,Model,source_node_list);
grad_A_diff=f_grad_A_diff(grad_A,T_diff_mat); %calculate the grad_A of the diff model
grad_b_diff=f_grad_b_diff_partition(T_diff_mat,output,output_correct,output_model,in_state,source_node_list);


%%% calcualte z
% initialize z
z_iter=[];
for i_t=1:T_diff
    z_iter(i_t).M=zeros(1,12*N);
end

iter_num=0; % count the number of iterations.
stop_flag=0;
while iter_num<=max_iter && stop_flag==0
    iter_num=iter_num+1;
    
    old_z=z_iter;
    
    for i_t=1:T_diff
        z_iter(i_t).M=z_iter(i_t).M*grad_A_diff(i_t).M+grad_b_diff(i_t).vec;
    end
    
    temp_new=[z_iter.M];
    temp_old=[old_z.M];
    norm_update=(norm(temp_new-temp_old,'fro')^2)/(T_diff*length(z_iter(1).M));
    norm_old=(norm(temp_old,'fro')^2)/(T_diff*length(z_iter(1).M));
    if norm_update<norm_old*threshold_ratio
        stop_flag=1; % if the MSE of state update is smaller than a ratio of previous state's MSE, then stop iteration.
    end
end

% % or calculate z by using theoretical inverse multiplication (this method causes singular matrix in simulation, so abandon it)
% iter_num=0;
% para_number=6*N;
% z_iter=[];% theoretical converged z
% I_a=eye(para_number);
% for i_t=1:T
%     z_iter(i_t).M=grad_b(i_t).vec*inv(I_a-grad_A(i_t).M); % This is quicker than iteration, but if condition number is large, use iteration instead.
% end

% % for debug or replace code
% para_number=6*N;
% z_converge=[];% theoretical converged z
% for i_t=1:T
%     z_converge(i_t).M=grad_b(i_t).vec*inv(eye(para_number)-grad_A(i_t).M); % I guess using iteration to calculate z is to avoid inverse.
% end
% i_t=1;
% figure;plot(1:36,[(z_iter(i_t).M)',(z_converge(i_t).M)'])
% legend('iter result z','theoretical z')
% % results show that the z calculated by iteration is very close to the
% % thoretical value using matrix inverse.


grad_dF_dw=f_grad_dF_dw_partition(in_state,Model,label,w_y,d_GB_d_w,source_node_list);
grad_d_diff=[];
for i_t=1:T_diff
    grad_d_diff(i_t).vec=z_iter(i_t).M*[grad_dF_dw(T_diff_mat(i_t,1)).M;grad_dF_dw(T_diff_mat(i_t,2)).M];
end

d_e_w_d_w=grad_d_diff;

grad_w_vec=0;
for i_t=1:T_diff
    grad_w_vec=grad_w_vec+d_e_w_d_w(i_t).vec;
end
grad_w_vec=grad_w_vec/T_diff;

grad_w_y=w_y;
for i_w=1:length(w_y)
    temp_idx=(i_w-1)*12;
    grad_w_y(i_w).g=grad_w_vec(temp_idx+[1:6])';
    grad_w_y(i_w).b=grad_w_vec(temp_idx+[7:12])';
end

grad_w_z=f_GBgrad2RXgrad(grad_w_y,w_z,w_y,d_GB_d_w); % chain rule to calcualte gradient over R X.

end


%%% function to calculate A(t) for gradient, partition version
function grad_A=f_grad_A_partition(in_state,label,Model,source_node_list)
% Input
% in_state--input state structure containing the nodal volt real and
% imaginary part.
% label--label structure containing the nodal real and reactive power
% injection.
% Model--the structure that stores the needed Z_nn matrix and Y_nk matrics for 
% each node's local transition function.
% source_node_list--the source node list

% Output:
% grad_A--a structure containg (d F_w(x(t),l(t)))/d x(t).
% grad_A(t).node_list is the list of nodes (same for all t). grad_A(t).M is
% a 6N-by-6N matrix (N the number of nonsource nodes) at time t.

% Strucutre of in_state or complex state
% complex_state(i).node--name of node i.
% complex_state(i).real--a 3-by-T matrix of nodal volt real part of node i of all
% time through T.
% complex_state(i).imag-a 3-by-T matrix of nodal volt imaginary part of node i of
% all time through T.

% Structure of label:
% label(i).node--name of node i.
% label(i).P--a 3-by-T matrix of nodal real power of node i of all
% time through T.
% label(i).Q--a 3-by-T matrix of nodal reactive power of node i of
% all time through T.

% Structure of the Model:
% Model(i).node--node name of node i.
% Model(i).neighbor--vector of node i's neighbors.
% Model(i).rZ_nn--Z_nn real matrix for node i's local transition function
% (6-by-6).
% Model(i).rY_nk--structure of Y_nk real matrix for node i's local transition function.
% Model(i).rY_nk(j).M--the Y_nk real matrix corresponding to node i's
% neighbor node j (6-by-6).

% The columns and rows of grad_A(t).M:
% The index of columns and rows are the same. 1~6 represent 3 nodal real
% and 3 nodal imaginary part of node 1; then the next 6 indices represent the states
% of node 2. And so on.
%-------------------------------------------------

Model_node_list=[Model.node]; % list of node in the Model structure
label_node_list=[label.node]; % list of node in the label structure
in_state_node_list=[in_state.node]; % list of node
Lia=ismember(in_state_node_list,source_node_list);
I1=find(Lia);
nonsource_node_list=in_state_node_list;
nonsource_node_list(I1)=[];% nonsource node list
N=length(nonsource_node_list); % number of nonsource node.
T=size(in_state(1).real,2);

% Speed up the code, initialize grad_A matrix into a structure by time
temp_A=zeros(6*N);
grad_A=[];
for i_t=1:T
    grad_A(i_t).node_list=nonsource_node_list;
    grad_A(i_t).M=temp_A;
end


% Then filling the values for grad_A's matrices of each time section
for i_x=1:N
    temp_x_idx=(i_x-1)*6+[1:6];
    
    target_node=nonsource_node_list(i_x);
    I1=find(Model_node_list==target_node);
    Model_local=Model(I1);
    rZ_nn=Model_local.rZ_nn;
    
    for i_y=1:N
        dev_node=nonsource_node_list(i_y);
        neighbor_idx=find(Model_local.neighbor==dev_node);
        
        if isempty(neighbor_idx) && dev_node~=target_node % if the dev_node is neither the target node nor the neighbor
            continue;
            
        elseif dev_node==target_node % if the dev_node is the target node
            I1=find(label_node_list==dev_node); % take the node's data
            local_P=label(I1).P; % P and Q data, 3-by-T
            local_Q=label(I1).Q;
            I1=find(in_state_node_list==dev_node);
            local_real=in_state(I1).real;% State real and imaginary data, 3-by-T
            local_imag=in_state(I1).imag;
            
            dR_R=[];
            dR_I=[];
            dI_R=[];
            dI_I=[];
            
            for i_phase=1:3 % phase a b c
                temp_P=local_P(i_phase,:); % take that phase's data
                temp_Q=local_Q(i_phase,:);
                temp_real=local_real(i_phase,:);
                temp_imag=local_imag(i_phase,:);
                
                denominator=temp_real.^2+temp_imag.^2;
                numerator_real=temp_P.*temp_real+temp_Q.*temp_imag;
                numerator_imag=temp_P.*temp_imag-temp_Q.*temp_real;
                divider=denominator.^2;
                
                dR_R(i_phase).v=(temp_P.*denominator-numerator_real.*temp_real*2)./divider; % derivative real to real.
                dR_I(i_phase).v=(temp_Q.*denominator-numerator_real.*temp_imag*2)./divider; % derivative real to imag.
                dI_R(i_phase).v=dR_I(i_phase).v; % derivative imag to real.
                dI_I(i_phase).v=-dR_R(i_phase).v; % derivative imag to imag.
            end
                        
            temp_y_idx=(i_y-1)*6+[1:6];
            for i_t=1:T
                temp_RR=diag([dR_R(1).v(i_t),dR_R(2).v(i_t),dR_R(3).v(i_t)]);
                temp_RI=diag([dR_I(1).v(i_t),dR_I(2).v(i_t),dR_I(3).v(i_t)]);
                temp_IR=diag([dI_R(1).v(i_t),dI_R(2).v(i_t),dI_R(3).v(i_t)]);
                temp_II=diag([dI_I(1).v(i_t),dI_I(2).v(i_t),dI_I(3).v(i_t)]);
                temp_M=[temp_RR,temp_RI;temp_IR,temp_II];
                
                grad_A(i_t).M(temp_x_idx,temp_y_idx)=rZ_nn*temp_M;
            end
            
        else % if the dev node is a neighbor
            rY_nk=Model_local.rY_nk(neighbor_idx).M;
            temp_M=-rZ_nn*rY_nk;
            temp_y_idx=(i_y-1)*6+[1:6];
            for i_t=1:T
                grad_A(i_t).M(temp_x_idx,temp_y_idx)=temp_M;
            end
        end
        
    end
end

end

%%% function to calculate b(t) for gradient, volt diff version. Consider
%%% parition (more than one source node)
function grad_b_diff=f_grad_b_diff_partition(T_diff_mat,output,output_correct,output_model,in_state,source_node_list)
% Input:
% T_diff_mat: time index table, time difference is second column's time
% minus first column's time.
% output--a structure of output. output(t).meter_list is a vector containing the
% list of meter names (same for all t). output(t).value is a vector of
% output value (in this case the volt measurement) matching to each meter
% at time t.
% output_correct--the same structure as output, but contains the actual
% output measurements.
% output_model--structure containing how the state is mapped to the output.
% in_state--input state structure containing the nodal volt real and imaginary part.
% source_node_list--the source node list.

% Output:
% grad_b--a structure containg d e_w(t)/d o(t) * d G(x(t)/d x(t).
% grad_b(t).node_list is the list of nodes. grad_b(t).vec is
% a 1-by-12N vector (N the number of nonsource nodes) at time t.


% Structure of output_model:
% output_model(i).meter--the name of the meter i.
% output_model(i).node--the node which the meter is connected to.
% output_model(i).phase--the phase of the smart meter i.
% 1,2,3,12,23,31,301,302,303 represents phase A, B, C, AB, BC, CA, 3-phase
% A, 3-phase B, and 3-phase C.
% output_model(i).f_v--the vector model of meter i. f_v is 1-by-3, a
% selection vetor of 1,-1, and 0 indicating which phase has which sign. f_v
% is the same for both real and imaginary part of a state of one particular
% meter.

% Strucutre of in_state or complex state
% complex_state(i).node--name of node i.
% complex_state(i).real--a 3-by-T matrix of nodal volt real part of node i of all
% time through T.
% complex_state(i).imag-a 3-by-T matrix of nodal volt imaginary part of node i of
% all time through T.
%----------------------------------------------------------

node_list=[in_state.node]; % list of node
Lia=ismember(node_list,source_node_list);
I1=find(Lia);
nonsource_node_list=node_list;
nonsource_node_list(I1)=[];% nonsource node list
N=length(nonsource_node_list); % number of nonsource node.
T=size(in_state(1).real,2);
T_diff=size(T_diff_mat,1);
meter_list=[output(1).meter_list];
output_model_meter_list=[output_model.meter];
M=length(output(1).meter_list);

de_do=zeros(T,M);
for i_t=1:T % initialize grad_b, and prepare de/do.
    grad_g(i_t).node_list=nonsource_node_list; % grad_g stores M-by-6N d_o/d_x
    grad_g(i_t).M=zeros(M,6*N);
    temp_v=2/M*(output(i_t).value-output_correct(i_t).value);
    de_do(i_t,:)=temp_v';% This is 1-by-M.
end

de_do_diff=zeros(T_diff,M); % volt diff version
for i_t=1:T_diff
    de_do_diff(i_t,:)=de_do(T_diff_mat(i_t,2),:)-de_do(T_diff_mat(i_t,1),:);
end

% filling the actual grad_b
for i_meter=1:M % for each meter
    target_meter=meter_list(i_meter);
    I1=find(output_model_meter_list==target_meter);
    target_node=output_model(I1).node; % the node connected to the target_meter
    f_v=output_model(I1).f_v;
    phase_type=output_model(I1).phase;
    
    I1=find(node_list==target_node);
    temp_real=f_v*in_state(I1).real; % 1-by-T
    temp_imag=f_v*in_state(I1).imag;
    
    denomenator=sqrt(temp_real.^2+temp_imag.^2);
    
    idx_node=find(nonsource_node_list==target_node);
    %%% below the calculation is based on
    %%% d_e/d_x_k=sum_over_m(d_e/d_o_m*d_g_m/d_x_r). More code but less
    %%% calculation.
    switch phase_type % filling grad_b based on phase type
        case 1 % phase A
            idx1=1;
            
            real_grad=temp_real./denomenator; % d_gm/d_real
            imag_grad=temp_imag./denomenator; % d_gm/d_imag
            for i_t=1:T % update each time
                temp_idx=(idx_node-1)*6+idx1; % idx of real of target node
                grad_g(i_t).M(i_meter,temp_idx)=real_grad(i_t);
                temp_idx=(idx_node-1)*6+idx1+3; % idx of imag of target node
                grad_g(i_t).M(i_meter,temp_idx)=imag_grad(i_t);
            end
            
        case 2 % phase B
            idx1=2;
            
            real_grad=temp_real./denomenator; % d_gm/d_real
            imag_grad=temp_imag./denomenator; % d_gm/d_imag
            for i_t=1:T % update each time
                temp_idx=(idx_node-1)*6+idx1; % idx of real of target node
                grad_g(i_t).M(i_meter,temp_idx)=real_grad(i_t);
                temp_idx=(idx_node-1)*6+idx1+3; % idx of imag of target node
                grad_g(i_t).M(i_meter,temp_idx)=imag_grad(i_t);
            end
            
        case 3 % phase C
            idx1=3;
            
            real_grad=temp_real./denomenator; % d_gm/d_real
            imag_grad=temp_imag./denomenator; % d_gm/d_imagg
            for i_t=1:T % update each time
                temp_idx=(idx_node-1)*6+idx1; % idx of real of target node
                grad_g(i_t).M(i_meter,temp_idx)=real_grad(i_t);
                temp_idx=(idx_node-1)*6+idx1+3; % idx of imag of target node
                grad_g(i_t).M(i_meter,temp_idx)=imag_grad(i_t);
            end
            
        case 12 % phase AB
            idx1=1; 
            idx2=2;
            
            real_grad=temp_real./denomenator; % d_gm/d_real
            imag_grad=temp_imag./denomenator; % d_gm/d_imag
            for i_t=1:T % update each time
                temp_idx=(idx_node-1)*6+idx1; % idx of real of target node (first phase)
                grad_g(i_t).M(i_meter,temp_idx)=real_grad(i_t);
                temp_idx=(idx_node-1)*6+idx1+3; % idx of imag of target node
                grad_g(i_t).M(i_meter,temp_idx)=imag_grad(i_t);
                
                % below pay attention to idx and sign before de_do. (second pahse)
                temp_idx=(idx_node-1)*6+idx2; % idx of real of target node
                grad_g(i_t).M(i_meter,temp_idx)=-real_grad(i_t);
                temp_idx=(idx_node-1)*6+idx2+3; % idx of imag of target node
                grad_g(i_t).M(i_meter,temp_idx)=-imag_grad(i_t);
            end
            
        case 23 % phase BC
            idx1=2; 
            idx2=3;
            
            real_grad=temp_real./denomenator; % d_gm/d_real
            imag_grad=temp_imag./denomenator; % d_gm/d_imag
            for i_t=1:T % update each time
                temp_idx=(idx_node-1)*6+idx1; % idx of real of target node (first phase)
                grad_g(i_t).M(i_meter,temp_idx)=real_grad(i_t);
                temp_idx=(idx_node-1)*6+idx1+3; % idx of imag of target node
                grad_g(i_t).M(i_meter,temp_idx)=imag_grad(i_t);
                
                % below pay attention to idx and sign before de_do. (second pahse)
                temp_idx=(idx_node-1)*6+idx2; % idx of real of target node
                grad_g(i_t).M(i_meter,temp_idx)=-real_grad(i_t);
                temp_idx=(idx_node-1)*6+idx2+3; % idx of imag of target node
                grad_g(i_t).M(i_meter,temp_idx)=-imag_grad(i_t);
            end
                 
        case 31 % phase CA
            idx1=3;
            idx2=1;
            
            real_grad=temp_real./denomenator; % d_gm/d_real
            imag_grad=temp_imag./denomenator; % d_gm/d_imag
            for i_t=1:T % update each time
                temp_idx=(idx_node-1)*6+idx1; % idx of real of target node (first phase)
                grad_g(i_t).M(i_meter,temp_idx)=real_grad(i_t);
                temp_idx=(idx_node-1)*6+idx1+3; % idx of imag of target node
                grad_g(i_t).M(i_meter,temp_idx)=imag_grad(i_t);
                
                % below pay attention to idx and sign before de_do. (second pahse)
                temp_idx=(idx_node-1)*6+idx2; % idx of real of target node
                grad_g(i_t).M(i_meter,temp_idx)=-real_grad(i_t);
                temp_idx=(idx_node-1)*6+idx2+3; % idx of imag of target node
                grad_g(i_t).M(i_meter,temp_idx)=-imag_grad(i_t);
            end
            
        case 301 % phase ABC-A
            idx1=1;
            
            real_grad=temp_real./denomenator; % d_gm/d_real
            imag_grad=temp_imag./denomenator; % d_gm/d_imag
            for i_t=1:T % update each time
                temp_idx=(idx_node-1)*6+idx1; % idx of real of target node
                grad_g(i_t).M(i_meter,temp_idx)=real_grad(i_t);
                temp_idx=(idx_node-1)*6+idx1+3; % idx of imag of target node
                grad_g(i_t).M(i_meter,temp_idx)=imag_grad(i_t);
            end
            
        case 302 % phase ABC-B
            idx1=2;
            
            real_grad=temp_real./denomenator; % d_gm/d_real
            imag_grad=temp_imag./denomenator; % d_gm/d_imag
            for i_t=1:T % update each time
                temp_idx=(idx_node-1)*6+idx1; % idx of real of target node
                grad_g(i_t).M(i_meter,temp_idx)=real_grad(i_t);
                temp_idx=(idx_node-1)*6+idx1+3; % idx of imag of target node
                grad_g(i_t).M(i_meter,temp_idx)=imag_grad(i_t);
            end
            
        case 303 % phase ABC-C
            idx1=3;
            
            real_grad=temp_real./denomenator; % d_gm/d_real
            imag_grad=temp_imag./denomenator; % d_gm/d_imag
            for i_t=1:T % update each time
                temp_idx=(idx_node-1)*6+idx1; % idx of real of target node
                grad_g(i_t).M(i_meter,temp_idx)=real_grad(i_t);
                temp_idx=(idx_node-1)*6+idx1+3; % idx of imag of target node
                grad_g(i_t).M(i_meter,temp_idx)=imag_grad(i_t);
            end
    end
end

for i_t=1:T_diff
    grad_b_diff(i_t).node_list=nonsource_node_list; % grad_b stores 1-by-12N vector
    grad_b_diff(i_t).vec=de_do_diff(i_t,:)*[-grad_g(T_diff_mat(i_t,1)).M,grad_g(T_diff_mat(i_t,2)).M];
end

end

%%% function to calculate d F/d w for gradient, partition version to
%%% consider more than one source node.
function grad_dF_dw=f_grad_dF_dw_partition(in_state,Model,label,w,d_GB_d_w,source_node_list)
% Input:
% in_state--input state structure containing the nodal volt real and
% reactive part.
% Model--the structure that stores the needed A1 matrix and A2 matrics for 
% each node's local transition function.
% label--label structure containing the nodal real and reactive power
% injection.
% w--the weight/parameter. In our problem this is the set of line
% parameters.
% d_GB_d_w--a structure storing the fixed derivative matrices for line
% parameter.d_GB_d_w(i).M is a 3-by-3 matrix. i=1~6, representing 6 phase
% types: aa, ab, ac, bb, bc, cc. G and B share the same patter, so we only 
% i=1~6 to represent these cases.

% Output:
% grad_dF_dw--a structure containg d F_w(x(t),l(t))/d w.
% grad_dF_dw(t).line_list is the list of lines, which is a 2-column matrix, in
% which each row is a line and each column is a vertex. 
% grad_dF_dw(t).node_list is the list of nodes (except source node). grad_dF_dw(t).M is
% a 6N-by-|w| vector (|w| the number of line parameters, i.e., 12*number of lines) at time t.


% Strucutre of in_state or complex state
% complex_state(i).node--name of node i.
% complex_state(i).real--a 3-by-T matrix of nodal volt real part of node i of all
% time through T.
% complex_state(i).imag-a 3-by-T matrix of nodal volt imaginary part of node i of
% all time through T.

% Structure of the Model:
% Model(i).node--node name of node i.
% Model(i).neighbor--vector of node i's neighbors.
% Model(i).rZ_nn--Z_nn real matrix for node i's local transition function
% (6-by-6).
% Model(i).rY_nk--structure of Y_nk real matrix for node i's local transition function.
% Model(i).rY_nk(j).M--the Y_nk real matrix corresponding to node i's
% neighbor node j (6-by-6).

% Structure of label:
% label(i).node--name of node i.
% label(i).P--a 3-by-T matrix of nodal real power of node i of all
% time through T.
% label(i).Q--a 3-by-T matrix of nodal reactive power of node i of
% all time through T.

% Structure of w:
% w(i).line--a 2-element vector containing the vertices of the line. Length
% of w is the number of lines.
% w(i).g--a vector of conductance of line i, contain 6 elements.
% w(i).b--a vector of susceptance of line i, contain 6 elements.
% The 6 elements are in the order of phase aa, ab, ac, bb, bc, cc.
%-----------------------------------------------------------------

state_node_list=[in_state.node]; % list of node
Lia=ismember(state_node_list,source_node_list);
I1=find(Lia);
nonsource_node_list=state_node_list;
nonsource_node_list(I1)=[];% nonsource node list
N=length(nonsource_node_list); % number of nonsource node.
T=size(in_state(1).real,2);

% state_node_list=[in_state.node]; %list of node names in in_state.
Model_node_list=[Model.node]; % list of node names in Model.
label_node_list=[label.node]; % list of node names in label.

temp_store=[];
for i=1:N
    target_node=nonsource_node_list(i);
    model_idx=find(Model_node_list==target_node); % find the local model for the target node.
    label_idx=find(label_node_list==target_node); % find the local model for the target node.
    grad_local_df_dw=f_grad_local_df_dw(in_state,Model(model_idx),label(label_idx),w,d_GB_d_w); % gradient of a local node
    % grad_local_df_dw--a structure containg d f_n_w(x_ne(n)(t),l_n(t))/d w.
    % grad_local_df_dw(t).line_list is the list of lines, which is a 2-column matrix, in
    % which each row is a line and each column is a vertex. grad_local_df_dw(t).M is
    % a 6-by-|w| vector (|w| the number of line parameters, i.e., 12*number of lines) at time t.
    
    temp_store(i).struct=grad_local_df_dw;
end

grad_dF_dw=[];
one_temp_store=temp_store(1).struct;
for i_t=1:T
    grad_dF_dw(i_t).line_list=one_temp_store(1).line_list;
    grad_dF_dw(i_t).node_list=nonsource_node_list;
    
    temp_M=[];
    for i_node=1:N
        pick_temp_store=temp_store(i_node).struct;
        temp_M=[temp_M;pick_temp_store(i_t).M];
    end
    
    grad_dF_dw(i_t).M=temp_M;
end

end

%%% function to build w_structure for a sub-network. The sub-net's w
%%% structure will be alined with the line_config.
function w_out=f_build_subnet_w(w_in,line_config)

w_out=[];
for i_w=1:length(w_in) % loop each line in the w_in
    check_line=w_in(i_w).line;
    I1=find(line_config(:,1)==check_line(1) & line_config(:,2)==check_line(2));
    I2=find(line_config(:,1)==check_line(2) & line_config(:,2)==check_line(1));
    I3=[I1;I2];
    if ~isempty(I3) % if the line exists in the line_config
        w_out(I3).line=line_config(I3,:);
        w_out(I3).r=w_in(i_w).r;
        w_out(I3).x=w_in(i_w).x;
    end
end   

end


%%% function to keep the label of nodes only exist in the sub-net.
function label_out=f_filter_subnet_label(label_in,subnet_node_list)
full_label_node_list=[label_in.node]';
[Lia,Locb]=ismember(subnet_node_list,full_label_node_list);
label_out=label_in(Locb);

% for i=1:length(subnet_node_list)
%     I1=find(full_label_node_list==subnet_node_list(i));
%     label_out(i)=label_in(I1);
% end
end

%%% function to build build subnet's complex state from subnet config
function out_state_complex=f_build_subnet_complex_state(in_state_complex,subnet_node_list)
full_node_list=[in_state_complex.node]';
[Lia,Locb]=ismember(subnet_node_list,full_node_list);
out_state_complex=in_state_complex(Locb);

% for i=1:length(subnet_node_list)
%     temp_node=subnet_node_list(i);
%     I1=find(full_node_list==temp_node);
%     out_state_complex(i)=in_state_complex(I1);
% end
end

%%% function to build subnet's output model from subnet nodes, need to
%%% remove output on the source nodes, because their error will not be
%%% changed.
function output_model_subnet=f_build_subnet_output_model(output_model,subnet_node_list,source_node_list)
Lia=ismember(subnet_node_list,source_node_list);
subnet_node_list(Lia)=[];
full_output_node_list=[output_model.node]';
Lia=ismember(full_output_node_list,subnet_node_list);
output_model_subnet=output_model(Lia);
end

%%% function to build subnet's output or output diff based on the subnet output model
function output_subnet=f_build_subnet_output(output,output_model_subnet)
subnet_meter_list=[output_model_subnet.meter]';
full_meter_list=output(1).meter_list;
[Lia,Locb]=ismember(subnet_meter_list,full_meter_list);
for i=1:length(output)
    output_subnet(i).meter_list=subnet_meter_list;
    output_subnet(i).value=output(i).value(Locb,:);
end
end

%%% function to build subnet's prior distribution structure
function prior_dist_subnet=f_build_subnet_prior_dist(prior_dist,line_config)
prior_dist_subnet.mean=[];
prior_dist_subnet.var=[];

for i_line=1:length(prior_dist.mean) % loop each line in the mean structure
    check_line=prior_dist.mean(i_line).line;
    I1=find(line_config(:,1)==check_line(1) & line_config(:,2)==check_line(2));
    I2=find(line_config(:,1)==check_line(2) & line_config(:,2)==check_line(1));
    I3=[I1;I2];
    if ~isempty(I3) % if the line exists in the line_config
        prior_dist_subnet.mean(I3).line=line_config(I3,:);
        prior_dist_subnet.mean(I3).r=prior_dist.mean(i_line).r;
        prior_dist_subnet.mean(I3).x=prior_dist.mean(i_line).x;
    end
end 


for i_line=1:length(prior_dist.var) % loop each line in the var structure
    check_line=prior_dist.var(i_line).line;
    I1=find(line_config(:,1)==check_line(1) & line_config(:,2)==check_line(2));
    I2=find(line_config(:,1)==check_line(2) & line_config(:,2)==check_line(1));
    I3=[I1;I2];
    if ~isempty(I3) % if the line exists in the line_config
        prior_dist_subnet.var(I3).line=line_config(I3,:);
        prior_dist_subnet.var(I3).r=prior_dist.var(i_line).r;
        prior_dist_subnet.var(I3).x=prior_dist.var(i_line).x;
    end
end 

end

%% 2020-11-03: function to estimate the primal_var from basic NLGNNdiff result
%%% function to calculate primal_var from the basic NLGNNdiff estimation
%%% result
function primal_var=f_est_primal_var_by_basic(dir_basic_result,subnet_id,alg_seed,num_sample_diff)
% Input:
% (1) dir_basic_result--the directory that stores the basic result.
% (2) subnet_id--which subnet it is for.
% (3) alg_seed--which random algorithm seed.
% (4) num_sample_diff--the number of samples in the diff output.
full_dir=[dir_basic_result,'/subnet-',num2str(subnet_id),'_algseed-',num2str(alg_seed)];
load(full_dir);
temp_x=min(save_result.loss_fun_history);
primal_var=temp_x*num_sample_diff/(num_sample_diff-1); % This is the estimated variance, corrected for variance estimation.
end