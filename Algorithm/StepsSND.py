import numpy as np
from scipy import stats
import copy 
from Algorithm.NodalOPF import InitSubProblem
from Algorithm.HelperFunctions import decimalToBinary,load_data
from Algorithm.LazyConstraints import cb_lower_approx


def initialize_SND(instance_file,time_series_file,bin_exp,dro,T):
    full_data = load_data(instance_file,time_series_file,bin_exp,dro)
    line_data,gen_data,num_lines,gen_num = full_data['line_data'],full_data['gen_data'],full_data['line_num'],full_data['gen_num']
    #Initial states
    a_init,z_init = line_data.a_init.loc[dro['failure_set']],line_data.z_init.loc[full_data['Switchable_lines']]

    #Parameters for binary expansion of continuous variables
    s,bin_vec_gen = full_data['s'],full_data['bin_vec_gen']

    dummy = 0 ######### Dummy value for the maximum representation
    #We prepare both continuous and binary cuts for the variables (to allow for cheaper lower bounds (no upper bounds avaiable so easily))
    #Encoding intial states for power generation
    P_gen_init_bin,pi_P_gen_bin = {},{} #For sb cuts
    P_gen_init_int,pi_P_gen_int = {},{} #For integer cuts
    P_gen_init_cont,pi_P_gen_cont = {},{} #For continuous cuts

    #From restricted separation algorithm
    P_gen_init_B,pi_P_gen_B = {},{} #For Benders cuts
    P_gen_init_L,pi_P_gen_L = {},{} #For Langrangian cuts

    
    for gen in range(gen_num): #we use all variables
        P_gen_init_bin[gen],pi_P_gen_bin[gen] = {},{} #For binary expanded representation
        P_gen_init_int[gen],pi_P_gen_int[gen] = {},{} #For integer cuts
        P_gen_init_cont[gen],pi_P_gen_cont[gen] = gen_data.P_init[gen],1*dummy #For continuous representation

        #For restricted separation problem
        P_gen_init_B[gen],pi_P_gen_B[gen] = {},{}
        P_gen_init_L[gen],pi_P_gen_L[gen] = {},{}

        #For binary representation
        for e_pos in range(len(bin_vec_gen[gen])): #Constructing vector of zeros for expansion
            P_gen_init_bin[gen][e_pos],pi_P_gen_bin[gen][e_pos] = 0,1*dummy #For SB cuts
            P_gen_init_int[gen][e_pos],pi_P_gen_int[gen][e_pos] = 0,1*dummy #For integer cuts

            P_gen_init_B[gen][e_pos],pi_P_gen_B[gen][e_pos] = 0,1*dummy #For S cuts
            P_gen_init_L[gen][e_pos],pi_P_gen_L[gen][e_pos] = 0,1*dummy #For L cuts

        #adding the corresponding values
        for e_pos,e in enumerate(list(reversed(list(decimalToBinary(int(gen_data.P_init[gen]/s)))))):
            P_gen_init_bin[gen][e_pos] = int(e) #for binary sb cuts
            P_gen_init_int[gen][e_pos] = int(e) #for integer cuts

            P_gen_init_B[gen][e_pos] = int(e) #for benders cuts
            P_gen_init_L[gen][e_pos] = int(e) #for Lagrangian cuts

    # #Encoding initial state for power limit and line availability (always binary)
    P_limit_init,pi_P_limit,pi_z = {},{},{}
    gamma_a = {}
    for line in range(num_lines):
        if line in full_data['Switchable_lines']:
            pi_z[line] = 1*dummy

        if line in dro['failure_set']:
            P_limit_init[line],pi_P_limit[line],gamma_a[line] = line_data.P_limit_init.loc[line],1*dummy,1*dummy

    scenario_tree = dro['scenario_tree']
    
    in_vars_bin = [P_limit_init[l] for l in dro['failure_set']]
    in_vars_bin += [P_gen_init_bin[g][e] for g in range(gen_num) for e in range(full_data['E_gen'][g])]

    in_vars_z_bin = [z_init[l] for l in full_data['Switchable_lines']]
    var_bin_count = len(in_vars_bin)

    in_vars_int = [P_limit_init[l] for l in dro['failure_set']]
    in_vars_int += [P_gen_init_int[g][e] for g in range(gen_num) for e in range(full_data['E_gen'][g])]
    in_vars_z_int = [z_init[l] for l in full_data['Switchable_lines']]
    var_int_count = len(in_vars_int)

    in_vars_cont = [P_limit_init[l] for l in dro['failure_set']]
    in_vars_cont += [P_gen_init_cont[g] for g in range(gen_num)]
    in_vars_z_cont = [z_init[l] for l in full_data['Switchable_lines']]
    var_cont_count = len(in_vars_cont)
    
    vertices = {} #To generate the cuts
    reverse_dict = None
    #Initilizing the node data
    node_tree = {}  
    #transition data
    nm_data = {}
    
    #traversing scenario_tree 
    for t in range(T):
        for parent_id in scenario_tree[t]:
            nm_data[parent_id] = {}
            if parent_id not in node_tree.keys():
                node_tree[parent_id] = {'cuts':{'RSP':{'L_I':{},'L_C':{}},'SB':{'B':{},'C':{}},'I':{}},'vertices':{}}
            for children_id in scenario_tree[t][parent_id]:
                nm_data[parent_id][children_id] = {}
                if children_id not in node_tree.keys():
                    node_tree[children_id] = {'cuts':{'RSP':{'L_I':{},'L_C':{}},'SB':{'B':{},'C':{}},'I':{}},'vertices':{},
                                              'x':{},'y':{},'z':{},'r':{},'a':{},'var_vec':None,'RSP':{'x':{},'a':{},'f':None,'E_s':[],'E_temp':[]}}
                
                if parent_id != None: #We still generate the delta holder regardless of optimization mode, since it's small
                    #For I cuts, we want the first problem to use all switching vars

                    if dro['z_type'] == 'M': #The cuts only consider a single z_var
                        z_count = -(T-t)+1 
                        I_count = z_count
                    else:#We consider the progressive decisions
                        z_count = 1
                        if t == 1: #We maintain all variables
                            I_count = t #Makes T
                        else: # we remove all variables
                            I_count = -T+t

                    #Binary cuts
                    node_tree[parent_id]['cuts']['SB']['B'][children_id] = {0:{'pi':copy.deepcopy(np.zeros(var_bin_count)),'gamma':copy.deepcopy(np.zeros(len(a_init.values))),'delta':copy.deepcopy(np.zeros(len(z_init.values)*(T-t+z_count))),'L':0}}
                    #Integer cuts
                    node_tree[parent_id]['cuts']['I'][children_id] = {0:{'x_bar':copy.deepcopy(np.zeros(var_bin_count+len(z_init.values)*(T-t+I_count))),'L':0}}
                    #Continuous cuts
                    node_tree[parent_id]['cuts']['SB']['C'][children_id] = {0:{'pi':copy.deepcopy(np.zeros(var_cont_count)),'gamma':copy.deepcopy(np.zeros(len(a_init.values))),'delta':copy.deepcopy(np.zeros(len(z_init.values)*(T-t+z_count))),'L':0}}
                    
                    
                    #Vertices for upper bounds
                    node_tree[parent_id]['vertices'][children_id] = {'reverse_dict':copy.deepcopy(reverse_dict),'vertices':copy.deepcopy(vertices)}
    node_tree['check'] = [] #WE will check solutions
    global_data = {'Statistical Upper bound':{0:1.5*dro['uB_M']},'Lower bound':{0:0},'Strict UB':{0:1.5*dro['uB_M']},'Std':{},'Current_best':{'UB':np.inf,'Best_tree':None},'Det_UB':{}}
    

    return scenario_tree,node_tree,global_data,nm_data,a_init,z_init,P_gen_init_bin,P_gen_init_cont,P_limit_init,full_data

def forward_pass(M,T,scenario_tree,instance_file,time_series_file,bin_exp,dro,a_init,z_init,P_gen_init,P_limit_init,node_tree,nm_data,global_data,master_iteration,samples,BiLin_M):
    #We discriminate between P_gen_init binary and continuous by bin_exp contents 
    #Forward pass
    F = {} #statistical upperbounds
    penalties = {}
    plot_stats = {}
    for k in range(M):
        path,F[k] = samples[k],0
        node_path = []
        penalties[k] = {}
        plot_stats[k] = {}
        for (t,parent_id,current_id),_ in path.items():
            
            # Setting parameters to construct a single forward model 
            if current_id == 0:
                entering_a,entering_z,entering_gen,entering_limit = a_init,z_init,P_gen_init,P_limit_init
            else:
                node_path.append(current_id)
                entering_a,entering_z,entering_gen,entering_limit = node_tree[parent_id]['a'][current_id],node_tree[parent_id]['x']['z_out'],node_tree[parent_id]['x']['P_gen_out'],node_tree[parent_id]['x']['P_limit_out']
            if t == T-1:
                terminal,cb = True,False
            else:
                terminal,cb = False,True 

            # Setting parameters to construct a single forward model 

            flowmod = InitSubProblem(current_id,bin_exp=bin_exp,cb=cb,dro=dro,BiLin_M=BiLin_M)
            flowmod.set_data(instance_file,time_series_file)
            flowmod.set_problem(entering_a,entering_z,entering_gen,entering_limit,scenario_tree,t,current_id,parent_id,node_tree,nm_data,terminal,phase='F')
            if t < T-1:
                flowmod.register_callback(cb_lower_approx)
                flowmod.m._register_cuts = True #Registering to obtain worst case distrib
            flowmod.solve_problem(True)
            flowmod.push_solution()
            
            # ================ Gathering statistics ================ #
            local_cost = 0
            tot_gen = 0
            for g in range(flowmod.m._gen_num):
                if bin_exp == None:
                    tot_gen += node_tree[current_id]['x']['P_gen_out'][g]
                    local_cost += flowmod.m._gen_cost[g]*node_tree[current_id]['x']['P_gen_out'][g]
                else:
                    tot_gen += np.dot(flowmod.m._bin_vec_gen[g],list(node_tree[current_id]['x']['P_gen_out'][g].values()))
                    local_cost += flowmod.m._gen_cost[g]*np.dot(flowmod.m._bin_vec_gen[g],list(node_tree[current_id]['x']['P_gen_out'][g].values()))
            for l in range(flowmod.m._line_num):
                local_cost += flowmod.m._switch_cost[l]*node_tree[current_id]['y']['z'][l]
            
            penalty = np.dot(flowmod.m._loss_cost.to_list(),list(node_tree[current_id]['y']['delta_p'].values())) + np.dot(flowmod.m._loss_cost.to_list(),list(node_tree[current_id]['y']['delta_m'].values()))
            penalties[k][current_id] = (np.dot(flowmod.m._loss_cost.to_list(),list(node_tree[current_id]['y']['delta_p'].values())),np.dot(flowmod.m._loss_cost.to_list(),list(node_tree[current_id]['y']['delta_m'].values())))
            plot_stats[k][current_id] = {'local_cost':local_cost,'Penalty':penalty,'gen':tot_gen,'loss':node_tree[current_id]['y']['delta_m'],'switching':node_tree[current_id]['y']['z']}
            
            local_cost += penalty


            F[k] += local_cost
            
            # ================ Gathering statistics ================ #

            if t > 0:
                if bin_exp != None:
                    flowmod.push_solution_vector()#For upper bound update procedure
                    global_data['Det_UB'][current_id] = {'In':{'a':flowmod.m._a_in,'z':flowmod.m._z_in,'P_gen':flowmod.m._P_gen_in,'P_limit':flowmod.m._P_limit_in},
                                                                    'Out':flowmod.push_solution(True)}
                    if 'RSP' in dro['cut_types']:
                        flowmod.push_RSP_solution(local_cost) #Storing the information for the generation of E sets
                        
            if t < T-1:
                #We should first push the cuts, then optimize, then fix the solution to copmute the relaxation.
                flowmod.push_Q_cuts() #Writing the cuts from the lower approximation lazy constraints to be able to use them to sample wc distrib
                flowmod.solve_problem(True,True) #We do not use the lower approximation since cuts where pushed
                flowmod.push_solution() #pushing solution to fix LP
                flowmod.fix_sol_LP(True) #Fixing LP by fixing the MIP solution. 
                flowmod.solve_problem(True,True,0) #Solving the LP without presolve and without lower approximation
                flowmod.push_P_worst_n_sample() #Sampling a vector from the worst case distribution
        #If we are in the binary phase, and we include restricted separation problem, we populate E with the forward pass info
        if bin_exp != None:
            if 'RSP' in dro['cut_types']:
                r_node_path = list(reversed(node_path))
                # print(r_node_path)
                for n_idx,node in enumerate(r_node_path):
                    if n_idx == 0: #We trivially add the final cost data
                        node_tree[node]['RSP']['E_temp'].append((node_tree[node]['RSP']['x'],node_tree[node]['RSP']['a'],node_tree[node]['RSP']['f']))
                    else: #We need to add one for each added E_s in the next one, to add
                        # print(node_tree[r_node_path[n_idx-1]]['RSP']['E_temp'])
                        for _,_,f_plus in node_tree[r_node_path[n_idx-1]]['RSP']['E_temp']:
                            node_tree[node]['RSP']['E_temp'].append((node_tree[node]['RSP']['x'],
                                                                     node_tree[node]['RSP']['a'],
                                                                     node_tree[node]['RSP']['f']+f_plus))
                    #     print(node_tree[node]['RSP']['E_s'].append())
                for node in node_path:
                    for e_temp in node_tree[node]['RSP']['E_temp']:
                        node_tree[node]['RSP']['E_s'].append(e_temp)
    
    #computing ubiased estimator
    mu = (1/M)*(sum(F.values()))
    sigma_2,alpha,n_sided = sum([(F[k] - mu)**2 for k in range(M)])*(1/(max(M-1,1))),0.05,2
    z_crit=stats.norm.ppf(1-alpha/n_sided)
    global_data['Statistical Upper bound'][master_iteration],global_data['Current_best']['UB'],global_data['Current_best']['Best_tree'],global_data['Std'][master_iteration],global_data['Penalties'],global_data['Statistics'] = mu+z_crit*(np.sqrt(sigma_2)/np.sqrt(M)),mu,copy.copy(node_tree),sigma_2,penalties,plot_stats
    return
             
def backward_pass(scenario_tree,node_tree,nm_data,bin_exp,dro,n_list,T,instance_file,time_series_file,a_init,P_limit_init,P_gen_init,z_init,global_data,master_iteration,BiLin_M):
    for t in reversed(range(T-1)):
        for grandparent in scenario_tree[t]:
            gp_node = node_tree[grandparent]
            for parent in scenario_tree[t][grandparent]:
                if parent in n_list:
                    p_node = node_tree[parent]
                    for child in scenario_tree[t+1][parent]:
                        if t+1 == T-1:
                            terminal = True
                            SB_pi = None #no estimation
                        else:
                            terminal = False
                            if bin_exp == None: #We work with the continuous
                                c_type = 'C'
                            else:
                                c_type = 'B'
                            
                            # print(t,len(SB_pi['delta']))
         
                        #------------- Lowerbounding ---------------
                        if 'SB' in dro['cut_types']: #strong benders cuts
                            #Implementation that uses a theoretically better implementation
                            #First we solve the MIP and fix the solution to 
                            pi_compute = InitSubProblem(bin_exp=bin_exp,dro=dro,BiLin_M=BiLin_M)
                            pi_compute.set_data(instance_file,time_series_file)
                            pi_compute.set_problem(p_node['a'][child],p_node['x']['z_out'],p_node['x']['P_gen_out'],p_node['x']['P_limit_out'],scenario_tree,t+1,child,parent,node_tree,nm_data,terminal,None,False,None,None,None,phase='B') #First like this to get initial dual pi
                            if t+1 < T-1: #we have to consider the current approximation
                                pi_compute.register_callback(cb_lower_approx)
                                pi_compute.m._register_cuts = False
                            pi_compute.solve_problem(True)
                            pi_compute.push_solution()
                            #We can use the already solved problem to push the integer cut if we satisfy requirements
                            if bin_exp != None:
                                if 'I' in dro['cut_types']:
                                    pi_compute.push_I_cut()
                            pi_compute.fix_sol_LP()
                            pi_compute.solve_problem(True,False,0)
                            SB_pi = pi_compute.get_pi()

                            # print(f'We solve problem {child} at time {t+1} to give node {parent} at time {t} a delta of length {len(SB_pi["delta"])}')
                            #Now that we obtain the \pi, we solve the lagrangian relaxation 
                            Lagrangian_relax = InitSubProblem(bin_exp=bin_exp,dro=dro,BiLin_M=BiLin_M)
                            Lagrangian_relax.set_data(instance_file,time_series_file)
                            Lagrangian_relax.set_problem(p_node['a'][child],p_node['x']['z_out'],p_node['x']['P_gen_out'],p_node['x']['P_limit_out'],scenario_tree,t+1,child,parent,node_tree,nm_data,terminal,None,False,None,None,SB_pi,phase='B') #First like this to get initial dual pi
                            if t+1 < T-1: #we have to consider the current approximation
                                Lagrangian_relax.register_callback(cb_lower_approx)
                                Lagrangian_relax.m._register_cuts = False
                            Lagrangian_relax.solve_problem(True)
                            Lagrangian_relax.push_SB_cut(SB_pi)
                            

                        if bin_exp != None:
                            if ('I' in dro['cut_types']) and ('SB' not in dro['cut_types']): #We solve the problem whenever we only have integer cuts
                                I_compute = InitSubProblem(bin_exp=bin_exp,dro=dro,BiLin_M=BiLin_M)
                                I_compute.set_data(instance_file,time_series_file)
                                I_compute.set_problem(p_node['a'][child],p_node['x']['z_out'],p_node['x']['P_gen_out'],p_node['x']['P_limit_out'],scenario_tree,t+1,child,parent,node_tree,nm_data,terminal,None,False,None,None,None,phase='B')
                                if t+1 < T-1: #we have to consider the current approximation
                                    I_compute.register_callback(cb_lower_approx)
                                    I_compute.m._register_cuts = False
                                I_compute.solve_problem(True)
                                I_compute.push_I_cut()
                            
                            #------------- Lowerbounding ---------------

                            #Upperbounding procedure (we can only update if the child was solved and if we are in the binary expanded)    
                        
                            if child in n_list: 
                                #Upperbounding procedure (we can only update if the child was solved and if we are in the binary expanded)
                                if t+1 == T-1:
                                    next_key = len(node_tree[parent]['vertices'][child]['vertices'])
                                    node_tree[parent]['vertices'][child]['vertices'][next_key] = node_tree[child]['var_vec']
                                    #++++++++++++++++++++++++++++ Upper bounding procedure ++++++++++++++++++++++++++++
                                else:
                                    #++++++++++++++++++++++++++++ Upper bounding procedure ++++++++++++++++++++++++++++
                                    #Updating specific approximation
                                    upperBound = InitSubProblem(child,bin_exp=bin_exp,dro=dro)
                                    upperBound.set_data(instance_file,time_series_file)
                                    upperBound.set_problem(p_node['a'][child],p_node['x']['z_out'],p_node['x']['P_gen_out'],p_node['x']['P_limit_out'],scenario_tree,t+1,child,parent,node_tree,nm_data,False,None,False,None,dro['uB_M'],phase='B')
                                    upperBound.solve_problem(True) #We expect it to be different from the big-M because of the updated value 
                                    global_data['Det_UB'][child] = {'In':{'a':upperBound.m._a_in,'z':upperBound.m._z_in,'P_gen':upperBound.m._P_gen_in,'P_limit':upperBound.m._P_limit_in},
                                                                    'Out':upperBound.push_solution(True)}
                                    upperBound.push_solution_vector()
                                    upperBound.update_UB()
                                    

                                    # print(f'For the upper bound, we have at node {child} from parent {parent} with objective {upperBound.m.ObjVal}')
                                    #++++++++++++++++++++++++++++ Upper bounding procedure ++++++++++++++++++++++++++++

    #Solving P1 to update the lowerbound
    generalLB = InitSubProblem(bin_exp=bin_exp,dro=dro,BiLin_M=BiLin_M)
    generalLB.set_data(instance_file,time_series_file)
    generalLB.set_problem(a_init,z_init,P_gen_init,P_limit_init,scenario_tree,0,0,None,node_tree,nm_data,False,phase='F')
    generalLB.register_callback(cb_lower_approx)
    generalLB.m._register_cuts = False #Only for evaluation, we dont write them into memory
    generalLB.solve_problem(True)
    global_data['Lower bound'][master_iteration] = generalLB.m.ObjVal


    if bin_exp != None:
        #Solving \tilde{P1} to update the strict upper bound
        generalUB = InitSubProblem(0,bin_exp=bin_exp,dro=dro)
        generalUB.set_data(instance_file,time_series_file)
        generalUB.set_problem(a_init,z_init,P_gen_init,P_limit_init,scenario_tree,0,0,None,node_tree,nm_data,False,None,False,None,dro['uB_M'],phase='F')
        generalUB.solve_problem(True) #We expect it to be different from the big-M because of the updated value 
        global_data['Strict UB'][master_iteration] = generalUB.m.ObjVal
        global_data['Det_UB'][0] = {'In':{'a':generalUB.m._a_in,'z':generalUB.m._z_in,'P_gen':generalUB.m._P_gen_in,'P_limit':generalUB.m._P_limit_in},
                                    'Out':generalUB.push_solution(True)}
    else:
        global_data['Strict UB'][master_iteration] = global_data['Strict UB'][master_iteration-1]

