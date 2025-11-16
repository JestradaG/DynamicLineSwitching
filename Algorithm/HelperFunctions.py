import pandas as pd
import numpy as np

#Miscellaneous functions that help us preprocess the data

def decimalToBinary(n): 
    return bin(n).replace("0b", "") 

#This is the main preprocessing function, reading the excel files for inputs
def load_data(data_file,time_series_file,bin_exp,dro):

    full_data = {}
    #Reading parameter file
    full_data['node_data'] = pd.read_excel(data_file,sheet_name='Buses')
    full_data['line_data'] = pd.read_excel(data_file,sheet_name='Lines')
    #setting up the load time series
    load_ts = pd.read_excel(time_series_file)
    cols = list(load_ts.columns)
    cols[0] = 'Node_ID'
    load_ts.columns = cols
    full_data['time_series'] = load_ts

    #Getting parameters
    full_data['gen_data'] = full_data['node_data'][full_data['node_data']['Type']=='G'].reset_index(drop=True)
    full_data['bus_data'] = full_data['node_data'][full_data['node_data']['Type']=='B'].reset_index(drop=True)

    full_data['node_list'] = full_data['node_data'].Node_ID
    full_data['gen_list'] = full_data['gen_data'].Node_ID

    #Mapping node index with corresponding id
    full_data['Node_ID_map'] = {'n':{},'r':{}}
    for idx,id in enumerate(full_data['node_list']):
        full_data['Node_ID_map']['n'][id] = idx 
        full_data['Node_ID_map']['r'][idx] = id
    
    full_data['Gen_ID_map'] = {'n':{},'r':{}}
    for idx,id in enumerate(full_data['gen_list']):
        full_data['Gen_ID_map']['n'][id] = idx
        full_data['Gen_ID_map']['r'][idx] = id

    #set cardinality
    full_data['node_num'] = len(full_data['node_list'])
    full_data['gen_num'] = len(full_data['gen_list'])
    full_data['line_num'] = len(full_data['line_data'].Line_ID)
    
    #Cost vectors (no swtiching cost)
    full_data['gen_cost'] = full_data['gen_data'].Gen_Cost
    full_data['loss_cost'] = full_data['node_data'].Loss_Cost
    full_data['switch_cost'] = full_data['line_data'].Switch_Cost

    #Max/Min angles
    full_data['theta_max'] = full_data['node_data'].Angle_Max
    full_data['theta_min'] = full_data['node_data'].Angle_Min

    #Operating parameters
    full_data['d'] = full_data['node_data'].Power_load*-1
    full_data['B'] = full_data['line_data'].B
    max_theta_diff = max(full_data['theta_max'])-min(full_data['theta_min'])
    full_data['bigM'] = max_theta_diff*full_data['B']
    full_data['P_line_max'] = full_data['line_data'].Rating_Max
    full_data['P_line_min'] = full_data['line_data'].Rating_Min
    full_data['P_line_normal'] = full_data['line_data'].Rating_Normal
    full_data['P_gen_max'] = full_data['gen_data'].P_Max
    full_data['P_gen_min'] = full_data['gen_data'].P_Min
    full_data['Max_Rampup'] = full_data['gen_data'].RampUp_Max
    full_data['Max_Rampdown'] = full_data['gen_data'].RampDown_Max

    #Constructing coefficient matrices
    full_data['from_nodes'] = dict((i,[]) for i in full_data['node_list'])
    full_data['from_nodes_id'] = dict((i,[]) for i in full_data['node_list'])

    full_data['to_nodes'] = dict((i,[]) for i in full_data['node_list'])
    full_data['to_nodes_id'] = dict((i,[]) for i in full_data['node_list'])

    for line_id in range(full_data['line_num']):
        full_data['from_nodes'][full_data['line_data'].loc[line_id,'from_node']].append(full_data['line_data'].loc[line_id,"to_node"])
        full_data['to_nodes'][full_data['line_data'].loc[line_id,'to_node']].append(full_data['line_data'].loc[line_id,"from_node"])

        full_data['from_nodes_id'][full_data['line_data'].loc[line_id,'from_node']].append(line_id)
        full_data['to_nodes_id'][full_data['line_data'].loc[line_id,'to_node']].append(line_id)

    #Forming to from node matrix
    full_data['In_Out'] = []
    # print(full_data['from_nodes_id'][])
    for node in full_data['node_list']:
        coeffs = np.zeros(full_data['line_num'])
        #Plus
        for i in full_data['from_nodes_id'][node]:
            coeffs[i] = -1.0

        #Minus
        for j in full_data['to_nodes_id'][node]:
            coeffs[j] = 1.0
        full_data['In_Out'].append(coeffs)

    full_data['In_Out'] = np.matrix(full_data['In_Out'])

    #Matrix for generator per node
    full_data['Arr'] = []
    for node in range(full_data['node_num']):
        node_id = full_data['Node_ID_map']['r'][node]
        coeffs = np.zeros(full_data['gen_num'])
        # println(node_data[node,"Type"])
        if full_data['node_data'].loc[node,"Type"] == "G":
            coeffs[full_data['Gen_ID_map']['n'][node_id]] = 1.0
        full_data['Arr'].append(coeffs)
    full_data['Arr'] = np.matrix(full_data['Arr'])

    full_data['theta_line'] = []
    for line in range(full_data['line_num']):
        coeffs = np.zeros(full_data['node_num'])
        from_b = full_data['line_data'].loc[line,"from_node"]
        fb_idx = full_data['Node_ID_map']['n'][from_b]
        coeffs[fb_idx] = 1.0
        to_b = full_data['line_data'].loc[line,"to_node"]
        tb_idx = full_data['Node_ID_map']['n'][to_b]
        coeffs[tb_idx] = -1.0
        full_data['theta_line'].append(coeffs)

    full_data['theta_line'] = np.matrix(full_data['theta_line'])

    if bin_exp != None: #We are asking for an iteration with binary expansion
        
        #computing number of digits to be represented for power generation
        full_data['s'] = bin_exp['s'] #Step such that we can have
        full_data['max_num_gen'] =  full_data['P_gen_max'].max()
        full_data['E_gen'] = {}
        full_data['bin_vec_gen'] = {}
        for g in range(full_data['gen_num']):
            full_data['E_gen'][g] = len(decimalToBinary(int(full_data['P_gen_max'][g]/full_data['s'])))
            full_data['bin_vec_gen'][g]= []
            for e in range(full_data['E_gen'][g]):
                full_data['bin_vec_gen'][g].append(full_data['s']*2**e)
    

    #Additional parameters
    full_data['Switchable_lines'] = list(full_data['line_data']['Switchable'][full_data['line_data']['Switchable'] == 1].index)
    full_data['num_failures'] = dro['num_failures']
    full_data['A_inf']= dro['A_inf']

    return full_data

def find_K_smallest(li,K,failure_set):
    #Find the K smallest numbers and negative
    li_dict = {}
    for l in range(len(li)): #Creating dictionary only with those indices that can fail
        if l in failure_set: #Only adding to the failure those lines that can fail
            li_dict[l] = li[l]

    failures = []
    for _ in range(K): #searching the K smallest and negative 
        min_val = np.inf
        min_line = None
        for line,val in li_dict.items():
            if val < min_val:
                min_val = val
                min_line = line

        if min_val >= 0: #If we don't have any more negative values
            break
        else: #we found one
            failures.append(min_line) #appending the index (line_id)
            del li_dict[min_line]

    return failures


def get_failures(a,K,gamma,psi,failure_set,line_num,cut_ref,cut_count,cut_type='SB',Lagrangian=None):
    Z = len(a)-sum(a)
    # print(f'Currently we have {Z} failures out of the {model._K} possible')
    if Z == K: #We already have all possible failures (a_n = a_p(n))
        pass #no more failures can happen so we do not add a cut
    else:
        #Maximizing the cut wrt a
        # print(f'For child {m}:\n Phi = {phi[m]} \n P_t = {P_trans} \n rho = {rho[m]} \n psi = {psi[m]} \n gamma = {gamma}')
        if cut_type == 'SB':
            gamma_psi = gamma-np.array(list(psi.values()))
        else: #We use integer cuts
            #gamma_psi = (2*Lagrangian*np.asarray(gamma) - np.ones(len(gamma))*Lagrangian) - np.array(list(psi.values()))
            gamma_psi = - np.array(list(psi.values()))
        fail_idx = find_K_smallest(gamma_psi,K,failure_set)
        #fail_idx is does not consider the current a, we now correct it
        
        if len(fail_idx) > 0:
            # print(f'Fail index is {fail_idx}')
            if sum(a) == line_num - K:#already all lines fail
                fail_idx = []
            elif sum(a) == line_num - 1: #We only add one
                #Choosing the worst 
                worst_a = fail_idx[0]
                if a[worst_a] == 0: #the worst is already 0, we choose the second one
                    if len(fail_idx) > 1:
                        fail_idx = [fail_idx[1]]
                    else:
                        fail_idx = []
                else:
                    fail_idx = [fail_idx[0]] #Worst was not selected yet, we select it now
        else: #Randomly selecting one to fail if none are selected
            fail_idx = [np.random.choice(failure_set,1)[0]]
        
        cut_ref[cut_count] = fail_idx
        for idx in fail_idx:
            a[idx] = 0
    return a

def sample_T(M,T,scenario_tree):
    paths_count = 0
    for parent in scenario_tree[T-1].keys():
        for _ in scenario_tree[T-1][parent]:
            paths_count += 1
    
    n_list = []
    T_list = []
    samples = {}
    k = 0
    done = False
    while done == False:
        parent_node = 0
        n_list.append(parent_node)
        t = 0
        path = {(t,None,parent_node):scenario_tree[t][None][parent_node]}
        for t in range(1,T):
            children = scenario_tree[t][parent_node]
            sel_idx = np.random.choice(list(children.keys()))
            # print(t,parent_node,list(children.keys()),sel_idx)
            selected_child = children[sel_idx]
            path[(t,parent_node,sel_idx)]=selected_child
            parent_node = sel_idx
            n_list.append(sel_idx)
        if (sel_idx not in T_list)|(len(T_list) >= paths_count): #End leaf has not been used
            T_list.append(sel_idx)
            samples[k] = path
            k += 1
        
        if len(samples) == M:
            done = True

        
    return samples,list(set(n_list))