% 2022-02-22: code to summarize the performance of parallel simulaiton of
% multiple parition. 

clear all

folder_dir='SaveResult\20220222_37b_GitHub';
% open_alg_seed_list=[3,5,7];
open_alg_seed_list=[1]';
subnet_num=3;

load([folder_dir,'\global_setup_save']); % load global set up

%% collect perforamnce of each alg_seed
MADR_improve_end_table=[];
MADR_improve_best_table=[];
subnet_epoch_length_table=[]; % record the simulation number of steps.

alg_seed_num=length(open_alg_seed_list);

for i_alg=1:alg_seed_num
    unify_algseed=open_alg_seed_list(i_alg);
    
    empty_w=struct('line',[],'r',[],'x',[]);
    global_w_end=empty_w; % global selected ending weight
    global_w_start=empty_w; % global starting weight
    global_w_correct=empty_w; % global correct weight
    global_w_best=empty_w; % global best weight
    
    %%% prepare before any subnet
    global_w_line_count=0;
    subnet_MADR_improve_by_end=[];
    subnet_MADR_improve_by_best=[];
    
    %%% check each subnet----------------------------
    for i_subnet=1:subnet_num
        current_subnet_id=i_subnet;
        current_alg_seed=unify_algseed;
        current_test_name=['subnet-',num2str(current_subnet_id),', algseed-',num2str(current_alg_seed)]; % test name for figures
        load([folder_dir,'\subnet-',num2str(current_subnet_id),'_algseed-',num2str(current_alg_seed)]); % load one subnet result
        
        %     epoch_length=length(save_result.loss_fun_history);
        
        for i=1:length(save_result.w_end) % organize weight result, each time add one line.
            global_w_line_count=global_w_line_count+1;
            global_w_end(global_w_line_count)=save_result.w_end(i);
            global_w_correct(global_w_line_count)=para_set.w_correct(i);
            global_w_start(global_w_line_count)=para_set.w_start(i);
            [temp_x,I1]=min(save_result.MADR_history);
            global_w_best(global_w_line_count)=save_result.w_history_record(I1).w(i);
        end
        
        temp_MADR_start=save_result.MADR_history(1);
        temp_MADR_best=temp_x;
        subnet_MADR_improve_by_end=[subnet_MADR_improve_by_end;save_result.MADR_improve];
        subnet_MADR_improve_by_best=[subnet_MADR_improve_by_best;(temp_MADR_start-temp_x)/temp_MADR_start*100];
        
        subnet_epoch_length_table=[subnet_epoch_length_table;length(save_result.MADR_history)-1];
    end
    
    %%% summarize this alg_seed result
    num_L=length(global_w_correct); % number of lines
    
    % turn structure to vectors
    [R_vec_correct,X_vec_correct]=f_wRX2Vec(global_w_correct);
    [R_vec_start,X_vec_start]=f_wRX2Vec(global_w_start);
    [R_vec_end,X_vec_end]=f_wRX2Vec(global_w_end);
    [R_vec_best,X_vec_best]=f_wRX2Vec(global_w_best);
    
    % calcualte different kinds of MADR improvement
    MADR_start=f_MADR([R_vec_start;X_vec_start],[R_vec_correct;X_vec_correct]);
    MADR_end=f_MADR([R_vec_end;X_vec_end],[R_vec_correct;X_vec_correct]);
    MADR_best=f_MADR([R_vec_best;X_vec_best],[R_vec_correct;X_vec_correct]);
    MADR_improve_by_end=(MADR_start-MADR_end)/MADR_start*100;
    MADR_improve_by_best=(MADR_start-MADR_best)/MADR_start*100;
    
    % store result in table: col 1 is seed, col 2 is improvement
    MADR_improve_end_table=[MADR_improve_end_table;unify_algseed,MADR_improve_by_end];
    MADR_improve_best_table=[MADR_improve_best_table;unify_algseed,MADR_improve_by_best];
    global_w_end_full(i_alg).w=global_w_end;
    global_w_end_full(i_alg).alg_seed=unify_algseed;
end

figure;
plot(MADR_improve_end_table(:,2))
grid on
xlabel('Test cases')
ylabel('MADR improvement (%)')
title('MADR improvement (%) of each case')

figure;
plot(subnet_epoch_length_table)
grid on
xlabel('Test subnet cases')
ylabel('Number of epochs')
title('Number of epochs of each subnet test')

% figure;
% semilogy(loss_value_table)
% grid on
% xlabel('Test cases')
% ylabel('Loss function value')
% title('Loss function value')

%% This is the end of code
end_x=1;

%% define functions
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

%%% function to calculate MADR of two sets of parameters
function MADR=f_MADR(vec_est,vec_true)
vec_diff=vec_est-vec_true;
MADR=sum(abs(vec_diff))/sum(abs(vec_true))*100;
end