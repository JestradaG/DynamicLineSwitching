import gurobipy as gp
from gurobipy import GRB
import numpy as np
import random as rnd
from Algorithm.HelperFunctions import load_data
from Algorithm.CutConstructor import SB_cut,I_cut
import copy
from typing import Callable

#Stage problem
class InitSubProblem:
    def __init__(self,id='',bin_exp=None,BiLin_M=1e10,dro=None,cb=True):
        self.m = gp.Model(f'StageSubproblem_{id}')
        self.m._callback = cb
        if self.m._callback == True:
            gp.setParam('Heuristics', 0)
            self.m._cb: Callable = None
            self.m.Params.LogToConsole = 0
            self.m.Params.lazyConstraints = 1
            self.m._iterations = 0
        self.m._bin_exp = bin_exp
        self.m._bigM_Lin = BiLin_M
        self.m._dro = dro

    def register_callback(self, cb: Callable):
        self.m._cb = cb
    
    def set_data(self,data_file,time_series_file):
        #Reading parameter file
        full_data = load_data(data_file,time_series_file,self.m._bin_exp,self.m._dro)
        #Getting the time series data
        self.m._time_series = full_data['time_series']
        self.m._node_data,self.m._line_data = full_data['node_data'],full_data['line_data']
        #Getting parameters
        self.m._gen_data,self.m._bus_data = full_data['gen_data'],full_data['bus_data']
        #set cardinality
        self.m._node_num,self.m._gen_num,self.m._line_num = full_data['node_num'],full_data['gen_num'],full_data['line_num']
        #Cost vectors
        self.m._gen_cost,self.m._loss_cost,self.m._switch_cost = full_data['gen_cost'],full_data['loss_cost'],full_data['switch_cost']
        #Max/Min angles
        self.m._theta_max,self.m._theta_min = full_data['theta_max'],full_data['theta_min']
        #Operating parameters
        self.m._B,self.m._bigM = full_data['B'],full_data['bigM']
        self.m._P_line_max,self.m._P_line_min,self.m._P_line_norm = full_data['P_line_max'],full_data['P_line_min'],full_data['P_line_normal']
        self.m._P_gen_max,self.m._P_gen_min = full_data['P_gen_max'],full_data['P_gen_min']
        self.m._Max_Rampup,self.m._Max_Rampdown = full_data['Max_Rampup'],full_data['Max_Rampdown']
        #Constructing coefficient matrices
        self.m._from_nodes,self.m._from_nodes_id = full_data['from_nodes'],full_data['from_nodes_id']
        self.m._to_nodes,self.m._to_nodes_id = full_data['to_nodes'],full_data['to_nodes_id']
        #Forming to-from node matrix
        self.m._In_Out,self.m._Arr,self.m._theta_line = full_data['In_Out'],full_data['Arr'],full_data['theta_line']
        if self.m._bin_exp != None:
            #computing number of digits to be represented
            self.m._s,self.m._max_num_gen,self.m._E_gen,self.m._bin_vec_gen = full_data['s'],full_data['max_num_gen'],full_data['E_gen'],full_data['bin_vec_gen'] #Expansion step
        #Paramters for the DRO part of the problem
        self.m._switchable_lines,self.m._failure_set,self.m._num_failures,self.m._K,self.m._A_inf,self.m._b_inf = full_data['Switchable_lines'],self.m._dro['failure_set'],full_data['num_failures'],full_data['num_failures'],full_data['A_inf'],1-full_data['A_inf']
        #Initially setting the approximation for the Q cuts to have standard sign
        self.m._vector_sign = 'Standard'
    
    def set_problem(self,a_pn,z_pn,P_gen_pn,P_limit_pn,scenario_tree,t,current_id,parent_id,node_tree,nm_data,final=False,lagrangian=None,initialization=False,initial_cuts=None,upper_bound=None,S_benders=None,RSP=False,phase='F'):
        
        #Benders cut information to adapt to the two-stage model
        self.m._S_benders = S_benders
        
        #loading data into model instance
        self.m._T,self.m._t,self.m._current_id,self.m._parent_id,self.m._node_tree,self.m._lagrangian = len(scenario_tree),t,current_id,parent_id,node_tree,lagrangian

        #loading the corresponding time series time
        self.m._d = self.m._time_series[t+1]*-1

        #Variables for fishing constraints
        self.m._a_in,self.m._z_in,self.m._P_gen_in,self.m._P_limit_in,self.m._scenario_tree = a_pn,z_pn,P_gen_pn,P_limit_pn,scenario_tree

        #arranging the incoming variables into a vector to be able to update the overestimations in terms of the incoming vars (to have a 'cut' w.r.t. previous node vars)
        self.m._in_vec = [self.m._P_limit_in[l] for l in self.m._failure_set]
        
        #For generation we consider relaxed of binary expanded
        if self.m._bin_exp == None: #for continuous variables
            self.m._in_vec += [np.round(self.m._P_gen_in[g],0) for g in range(self.m._gen_num)]
        else:
            self.m._in_vec += [np.round(self.m._P_gen_in[g][e],0) for g in range(self.m._gen_num) for e in range(self.m._E_gen[g])]
        
        if self.m._dro['z_type'] == 'M': #We keep one for each stage
            self.m._in_vec += [np.round(self.m._z_in[l],0) for l in self.m._switchable_lines]
        else: #Here we only have at the first stage, other do not have it
            if t == 1: #If we are at the second stage, we get all the variables as in vars to have the complete representation
                for time in range(self.m._T):
                    self.m._in_vec += [np.round(node_tree[0]['x']['z_plan'][time][l],0) for l in node_tree[0]['x']['z_plan'][time]]
        #Parameters 

        self.m._initialization,self.m._initial_cuts = initialization,initial_cuts

        #Generation variables (state) for ramping constraints
        if self.m._bin_exp == None:
            self.m._P_gen_out = self.m.addVars(self.m._gen_num,name='Generation')
            self.m._P_gen_fish = self.m.addVars(self.m._gen_num,name='Generation_Fishing')
        else:
            #Binarizing power generation (for ramping constraints)
            self.m._P_gen_out = {} #Dictionary of the variables that will be inherited
            self.m._P_gen_fish = {} #Dictionary for the variables that were inherited
            for g in range(self.m._gen_num): 
                if self.m._bin_exp == None:
                    self.m._P_gen_out[g] = self.m.addVar(vtype=GRB.BINARY,ub=1,lb=0,name=f'BinaryExpanded_PowerGeneration_{g}')
                    self.m._P_gen_fish[g] = self.m.addVar(vtype=GRB.BINARY,ub=1,lb=0,name=f'BinaryExpanded_PowerGenerationFishing_{g}')
                else:
                    self.m._P_gen_out[g] = {}
                    self.m._P_gen_fish[g] = {}
                    for e in range(self.m._E_gen[g]):
                        self.m._P_gen_out[g][e] = self.m.addVar(vtype=GRB.BINARY,ub=1,lb=0,name=f'BinaryExpanded_PowerGeneration_{g}_{self.m._bin_vec_gen[g][e]}')
                        self.m._P_gen_fish[g][e] = self.m.addVar(vtype=GRB.BINARY,ub=1,lb=0,name=f'BinaryExpanded_PowerGenerationFishing_{g}_{self.m._bin_vec_gen[g][e]}')
        
        
        
        #Switching state variables
        self.m._z = self.m.addVars(self.m._line_num,vtype=GRB.BINARY,ub=1,lb=0,name='Switching') #Mixed dictionary of variables and parameters
        self.m._z_out,self.m._z_fish,self.m._delta_vec = {},{},[]
        if self.m._dro['z_type'] == 'T':
            self.m._e,self.m._z_t = {},{} #To handle possibly infeasible switching decision when planning
            for time in range(t+1,self.m._T):#one set of additional vectors for each stage t -> T (at 0 we don't add fishing constraints because we decide there)
                self.m._z_t[time] = {} 
                for l in self.m._switchable_lines:
                    self.m._z_t[time][l] = self.m.addVar(vtype=GRB.BINARY,ub=1,lb=0,name=f'Switching_set_{time}_{l}') #Mixed dictionary of variables and parameters
                
        #Line availability state variables
        self.m._a = self.m.addVars(self.m._line_num,vtype=GRB.BINARY,ub=1,lb=0,name='Line_availability')
        self.m._a_fish = {}

        #Line limitation state variables
        self.m._P_limit_out,self.m._P_limit_fish = {},{}

        for l in range(self.m._line_num):
            
            if l in self.m._failure_set: #for those line that are subject to failure we have an incoming variable
                
                self.m._a_fish[l] = self.m._a[l]
                #State variables (under the assumption we now dont need to expand the variables)
                self.m._P_limit_out[l] = self.m.addVar(vtype=GRB.BINARY,ub=1,lb=0,name=f'Powerflow_limit_{l}')
                self.m._P_limit_fish[l] = self.m.addVar(vtype=GRB.BINARY,ub=1,lb=0,name=f'Powerflow_limit_Fishing_{l}')
            else: #By assumption means that it is always available (1)
                self.m._a[l].ub,self.m._a[l].lb = 1,1

            if l in self.m._switchable_lines: #for those lines that are switchable we have incoming and outgoing variables
                self.m._z_out[l] =  self.m._z[l]
                self.m._delta_vec.append(self.m._z[l]) #CURRENT stage variables
                self.m._z_fish[l] = self.m.addVar(vtype=GRB.BINARY,ub=1,lb=0,name=f'SwitchingFishing_{l}')
                if self.m._dro['z_type'] =='T': #In two-stage we add an infeasibility handler
                    self.m._e[l] = self.m.addVar(vtype=GRB.BINARY,ub=1,lb=0,name=f'Two_stage_Switch_handle_{l}')
            else: #By assumption means that takes value 1 as long as it is not on the failure set
                if l not in self.m._failure_set:
                    self.m._z[l].ub,self.m._z[l].lb = 1,1

        if self.m._dro['z_type'] == 'T': #Adding the fishing variables
            if S_benders != None: #If we are solving the relaxation, we need to add the fishing vars
                self.m._delta_vec = [self.m._z_fish[l] for l in self.m._switchable_lines] + self.m._delta_vec
            for time in range(t+1,self.m._T):
                self.m._delta_vec += [self.m._z_t[time][l] for l in self.m._z_t[time]]
        
        #Local control variables
        self.m._P_trans = self.m.addVars(self.m._line_num,name='Transmission') #Transmission lines (now localized)
        self.m._theta = self.m.addVars(self.m._node_num,name='Angle')
        self.m._delta_p = self.m.addVars(self.m._node_num,name='LoadShedding')
        self.m._delta_m = self.m.addVars(self.m._node_num,name='LoadLoss')
        
        #Arranging the fishing variables into a vector to vectorize strong bender cut computation
        #arranging the outgoing variables into a vector to be able to vectorize inner products
        self.m._fish_vec = [self.m._P_limit_fish[l] for l in self.m._failure_set]
        if self.m._bin_exp ==None:
            self.m._fish_vec += [self.m._P_gen_fish[g] for g in range(self.m._gen_num)]
        else:
            self.m._fish_vec += [self.m._P_gen_fish[g][e] for g in range(self.m._gen_num) for e in range(self.m._E_gen[g])]
        self.m._fish_delta_vec = [self.m._z_fish[l] for l in self.m._switchable_lines]
        
        self.m._a_f_vec = [self.m._a_fish[l] for l in self.m._failure_set]
        
        #arranging the outgoing variables into a vector to be able to vectorize inner products
        self.m._var_vec = [self.m._P_limit_out[l] for l in self.m._failure_set]
        if self.m._bin_exp ==None:
            self.m._var_vec += [self.m._P_gen_out[g] for g in range(self.m._gen_num)]
        else:
            self.m._var_vec += [self.m._P_gen_out[g][e] for g in range(self.m._gen_num) for e in range(self.m._E_gen[g])]
        
        if (self.m._dro['z_type'] == 'T') and (self.m._t != 0): #We treat them differently (for upper bounds, we dont add the variables up)
            self.m._full_vec = self.m._var_vec
        else:
            self.m._full_vec = self.m._var_vec + self.m._delta_vec #For upperbounding, consider also planned variables
        
        #Lagrangian dual optimization parameters
        self.m._final = final
        if self.m._final == False:

            if upper_bound == None: #We work with lower approximations
                #DRO reformulation variables
                #Constructing next state from information on our current node
                self.m._A,self.m._b,self.m._qs,self.m._psi,self.m._phi,self.m._rho,self.m._ms = {},{},{},{},{},{},[]

                #Data for Strong Benders cuts 'C'
                self.m._a_SB_C_ref,self.m._Q_SB_C_cuts,self.m._SB_C_cut_count = {},{},0

                #Data for Strong Benders cuts 'B'
                self.m._a_SB_B_ref,self.m._Q_SB_B_cuts,self.m._SB_B_cut_count = {} ,{},0

                #Data for Integer cuts 'I'
                self.m._a_I_ref,self.m._Q_I_cuts,self.m._I_cut_count,self.m._nm_data = {} ,{},0,nm_data

                if self.m._initialization == False:
                    for m,(states,qnm,_) in scenario_tree[t+1][current_id].items():
                        self.m._ms.append(m)
                        A_m,b_m = self.constructAb(states)
                        self.m._A[m],self.m._b[m],self.m._qs[m],self.m._psi[m],self.m._phi[m],self.m._rho[m] = A_m,b_m.T.flatten(),qnm,{},self.m.addVar(name=f'TotalFailure_DRO_{m}'),{}
                        
                        #Variable to linearize the product of the transmission and psi (for ambiguity set definition)
                        for l in self.m._failure_set:
                            self.m._psi[m][l] = self.m.addVar(name=f'Line_DRO_{m}_{l}')
                            self.m._rho[m][l] = self.m.addVar(name=f'LinProd_Psi_{m}_P_limit_{l}')
            
            else: #We work with vertex enumerations as upper bounds, so no additional additional variables
                self.m._phi,self.m._qs,self.m._ms,self.m._delta,self.m._delta_vecs,self.m._delta_M,self.m._delta_u,self.m._upperbound_M = {},{},[],{},{},{},{},upper_bound 

                for m,(states,qnm,_) in scenario_tree[t+1][current_id].items():
                    self.m._phi[m],self.m._qs[m],self.m._delta[m],self.m._delta_vecs[m],self.m._delta_M[m],self.m._delta_u[m] = self.m.addVar(name=f'Upperbound_{m}'),qnm,{},[],len(self.m._node_tree[self.m._current_id]['vertices'][m]['vertices'].keys()),[]
                    self.m._ms.append(m)
                    #For new upperbounding approach we generate one vector of variables $\delta$ for each explored nodes m and i \in M(m)
                    for m_i in range(self.m._delta_M[m]+1):
                        self.m._delta[m][m_i] = self.m.addVar(vtype=GRB.BINARY,name=f'delta_{m}_{m_i}')
                        self.m._delta_vecs[m].append(self.m._delta[m][m_i])
                        if m_i == self.m._delta_M[m]:
                            self.m._delta_u[m].append(self.m._upperbound_M)
                        else:
                            self.m._delta_u[m].append(self.m._node_tree[self.m._current_id]['vertices'][m]['vertices'][m_i][1])

        #Constraints related to nodes
        for n in range(self.m._node_num):
            #Angle constraints
            self.m.addLConstr(self.m._theta[n] <= self.m._theta_max[n],name=f'Max_Angle_{n}')
            self.m.addLConstr(self.m._theta[n] >= self.m._theta_min[n],name=f'Min_Angle_{n}')
            
            #Flow balance constraints
            if self.m._bin_exp == None: #(We work with continuous generation (everything else is the same))
                self.m.addLConstr(gp.LinExpr(self.m._In_Out[n].tolist()[0],[self.m._P_trans[j] for j in self.m._P_trans]) + 
                                 gp.LinExpr(self.m._Arr[n].tolist()[0],[self.m._P_gen_out[j] for j in self.m._P_gen_out]) + 
                                 self.m._delta_p[n] - 
                                 self.m._delta_m[n] == 
                                 self.m._d[n],name=f'FlowBalance_{n}')
            else: #We expand generation
                self.m.addLConstr(gp.LinExpr(self.m._In_Out[n].tolist()[0],[self.m._P_trans[j] for j in self.m._P_trans]) + 
                                 gp.quicksum([self.m._Arr[n,g]*gp.LinExpr(self.m._bin_vec_gen[g],[self.m._P_gen_out[g][j] for j in self.m._P_gen_out[g]]) for g in self.m._P_gen_out]) + 
                                 self.m._delta_p[n] - 
                                 self.m._delta_m[n] == 
                                 self.m._d[n],name=f'FlowBalance_{n}')
        
        #Constraints related to lines (always continuous)
        for l in range(self.m._line_num):

            #Period-binding constraints (since they involve a, which comes from the worst-case distribution, it should not be a problem to use, because we assume the a_in is part of the perfect information)
            #Logic to handle possibly infeasible line switching for planned decisions in Two-stage model (For internal equivalence, but we keep them in the out vars)
            #Only can swuitch if line is available wrt to the state
            # if (l in self.m._switchable_lines):

            if (self.m._dro['z_type'] == 'T'):#We work with e to handle infeasibility  
                self.m.addLConstr(self.m._e[l]-self.m._z[l] <= 0,name=f'Switch_Handler_{l}')
                self.m.addLConstr(self.m._e[l]-self.m._a[l] <= 0,name=f'Switch_Availability_{l}')
                #Line failure constraints
                self.m.addLConstr(self.m._B[l]*gp.LinExpr(self.m._theta_line[l].tolist()[0],[self.m._theta[j] for j in self.m._theta])-self.m._P_trans[l]+(2*self.m._bigM[l]) - self.m._e[l]*self.m._bigM[l] - self.m._a[l]*self.m._bigM[l] >= 0,name=f'LineFailure_1_{l}')
                self.m.addLConstr(self.m._B[l]*gp.LinExpr(self.m._theta_line[l].tolist()[0],[self.m._theta[j] for j in self.m._theta])-self.m._P_trans[l]-(2*self.m._bigM[l]) + self.m._e[l]*self.m._bigM[l] + self.m._a[l]*self.m._bigM[l] <= 0,name=f'LineFailure_2_{l}')

                #Line rate constraints
                self.m.addLConstr(self.m._P_trans[l] - self.m._P_line_max[l]*self.m._e[l] <= 0,name=f'Max_Trans_{l}')
                self.m.addLConstr(self.m._P_trans[l] - self.m._P_line_min[l]*self.m._e[l] >= 0,name=f'Min_Trans_{l}')
        
            else: #No two-stage and everything remains as is so no need to have special treatment
                self.m.addLConstr(self.m._z[l]-self.m._a[l] <= 0,name=f'Switch_Availability_{l}')
                #Line failure constraints
                self.m.addLConstr(self.m._B[l]*gp.LinExpr(self.m._theta_line[l].tolist()[0],[self.m._theta[j] for j in self.m._theta])-self.m._P_trans[l]+(2*self.m._bigM[l]) - self.m._z[l]*self.m._bigM[l] - self.m._a[l]*self.m._bigM[l] >= 0,name=f'LineFailure_1_{l}')
                self.m.addLConstr(self.m._B[l]*gp.LinExpr(self.m._theta_line[l].tolist()[0],[self.m._theta[j] for j in self.m._theta])-self.m._P_trans[l]-(2*self.m._bigM[l]) + self.m._z[l]*self.m._bigM[l] + self.m._a[l]*self.m._bigM[l] <= 0,name=f'LineFailure_2_{l}')

                #Line rate constraints
                self.m.addLConstr(self.m._P_trans[l] - self.m._P_line_max[l]*self.m._z[l] <= 0,name=f'Max_Trans_{l}')
                self.m.addLConstr(self.m._P_trans[l] - self.m._P_line_min[l]*self.m._z[l] >= 0,name=f'Min_Trans_{l}')
            

            if l in self.m._failure_set:
                #New constraint 
                self.m.addLConstr(self.m._P_trans[l] - (self.m._P_line_max[l]-self.m._P_line_norm[l])*self.m._P_limit_out[l] <= self.m._P_line_norm[l],name=f'Line_flow_safety_{l}')
        
        #Logic for binary expanded variables or not
        if self.m._bin_exp == None:
            #Constraints related to generators
            for g in range(self.m._gen_num):
                #Generation constraints
                self.m.addLConstr(self.m._P_gen_out[g] <= self.m._P_gen_max[g],name=f'Max_Generation_{g}')
                self.m.addLConstr(self.m._P_gen_out[g] >= self.m._P_gen_min[g],name=f'Min_Generation_{g}')

                #Ramping constraints
                self.m.addLConstr(self.m._P_gen_out[g] - self.m._P_gen_fish[g] <= self.m._Max_Rampup[g],name=f'Max_Rampup_{g}')
                self.m.addLConstr(self.m._P_gen_fish[g] - self.m._P_gen_out[g] <= self.m._Max_Rampdown[g],name=f'Max_Rampdown_{g}')

            self.m.setObjective(gp.LinExpr(self.m._gen_cost.to_list(),[self.m._P_gen_out[g] for g in self.m._P_gen_out])+ gp.LinExpr(self.m._loss_cost.to_list(),[self.m._delta_p[g] for g in self.m._delta_p])+gp.LinExpr(self.m._loss_cost.to_list(),[self.m._delta_m[g] for g in self.m._delta_m]),GRB.MINIMIZE)
        else:
            if RSP == False: #If we use an n-period model (RSP = True) we skip these constraints. Only for binary model
                for g in range(self.m._gen_num):
                    #Generation constraints
                    self.m.addLConstr(gp.LinExpr(self.m._bin_vec_gen[g],[self.m._P_gen_out[g][k] for k in self.m._P_gen_out[g]]) <= self.m._P_gen_max[g],name=f'Max_Generation_{g}')
                    self.m.addLConstr(gp.LinExpr(self.m._bin_vec_gen[g],[self.m._P_gen_out[g][k] for k in self.m._P_gen_out[g]]) >= self.m._P_gen_min[g],name=f'Min_Generation_{g}')

                    #Ramping constraints
                    self.m.addLConstr(gp.LinExpr(self.m._bin_vec_gen[g],[self.m._P_gen_out[g][k] for k in self.m._P_gen_out[g]]) - gp.LinExpr(self.m._bin_vec_gen[g],[self.m._P_gen_fish[g][k] for k in self.m._P_gen_fish[g]]) <= self.m._Max_Rampup[g],name=f'Max_Rampup_{g}')
                    self.m.addLConstr(gp.LinExpr(self.m._bin_vec_gen[g],[self.m._P_gen_fish[g][k] for k in self.m._P_gen_fish[g]]) - gp.LinExpr(self.m._bin_vec_gen[g],[self.m._P_gen_out[g][k] for k in self.m._P_gen_out[g]]) <= self.m._Max_Rampdown[g],name=f'Max_Rampdown_{g}')
            
            if self.m._final == False:
                if upper_bound == None: #the cost to go function is expressed by the lower approximations
                    #Product linearization constraints
                    if self.m._initialization == False:
                        for m in self.m._ms:
                            #For linearization (with or without binary expansion,we only have binary state variables in the ambiguity set)
                            for l in self.m._failure_set:
                                self.m.addLConstr(- (self.m._bigM_Lin*(1-self.m._P_limit_out[l])) - self.m._psi[m][l] + self.m._rho[m][l] <= 0,name=f'PsiPTrans_Linearization_{1}_{m}_{l}')
                                self.m.addLConstr( self.m._psi[m][l] - self.m._rho[m][l] - (self.m._bigM_Lin*(1-self.m._P_limit_out[l])) <= 0,name=f'PsiPTrans_Linearization_{2}_{m}_{l}')
                                self.m.addLConstr(-self.m._bigM_Lin*self.m._P_limit_out[l] - self.m._rho[m][l] <= 0,name=f'PsiPTrans_Linearization_{3}_{m}_{l}')
                                self.m.addLConstr(self.m._rho[m][l] - self.m._bigM_Lin*self.m._P_limit_out[l] <= 0,name=f'PsiPTrans_Linearization_{4}_{m}_{l}')
                            
                            ## =============================== Lowerbounds ==========================================
                    else: #We are in the initialization procedure
                        m = t+1
                        #For linearization (with or without binary expansion,we only have binary state variables in the ambiguity set)
                        for l in self.m._failure_set:
                            self.m.addLConstr(- (self.m._bigM_Lin*(1-self.m._P_limit_out[l])) - self.m._psi[m][l] + self.m._rho[m][l] <= 0,name=f'PsiPTrans_Linearization_{1}_{m}_{l}')
                            self.m.addLConstr( self.m._psi[m][l] - self.m._rho[m][l] - (self.m._bigM_Lin*(1-self.m._P_limit_out[l])) <= 0,name=f'PsiPTrans_Linearization_{2}_{m}_{l}')
                            self.m.addLConstr(-self.m._bigM_Lin*self.m._P_trans_out[l] - self.m._rho[m][l] <= 0,name=f'PsiPTrans_Linearization_{3}_{m}_{l}')
                            self.m.addLConstr(self.m._rho[m][l] - self.m._bigM_Lin*self.m._P_trans_out[l] <= 0,name=f'PsiPTrans_Linearization_{4}_{m}_{l}')
                            
                    ##Setting the product of the linearized variable product and the matrix
                    if lagrangian == None:
                        if S_benders is None:
                            self.m.setObjective(gp.quicksum([self.m._gen_cost[g]*gp.LinExpr(self.m._bin_vec_gen[g],[self.m._P_gen_out[g][j] for j in self.m._P_gen_out[g]]) for g in self.m._P_gen_out])
                                                +gp.LinExpr(self.m._loss_cost.to_list(),[self.m._delta_p[g] for g in self.m._delta_p])
                                                +gp.LinExpr(self.m._loss_cost.to_list(),[self.m._delta_m[g] for g in self.m._delta_m])
                                                +gp.LinExpr(self.m._switch_cost.to_list(),[self.m._z[l] for l in self.m._z] )
                                                +gp.quicksum(self.m._qs[m]*(gp.quicksum(self.m._A[m][l_pos,l_pos]*self.m._rho[m][l] for l_pos,l in enumerate(self.m._failure_set))
                                                                            +gp.LinExpr(self.m._b[m],[self.m._psi[m][l] for l in self.m._psi[m]])
                                                                            +self.m._phi[m]) for m in self.m._ms),GRB.MINIMIZE)
                        #If we have the strong benders option, we modify the objective with the pi vector
                        else:
                            #print(list(range(t,self.m._T)),self.m._t,self.m._z_t.keys())
                            self.m.setObjective(gp.quicksum([self.m._gen_cost[g]*gp.LinExpr(self.m._bin_vec_gen[g],[self.m._P_gen_out[g][j] for j in self.m._P_gen_out[g]]) for g in self.m._P_gen_out])
                                                +gp.LinExpr(self.m._loss_cost.to_list(),[self.m._delta_p[g] for g in self.m._delta_p])
                                                +gp.LinExpr(self.m._loss_cost.to_list(),[self.m._delta_m[g] for g in self.m._delta_m])
                                                +gp.LinExpr(self.m._switch_cost.to_list(),[self.m._z[l] for l in self.m._z] )
                                                +gp.quicksum(self.m._qs[m]*(gp.quicksum(self.m._A[m][l_pos,l_pos]*self.m._rho[m][l] for l_pos,l in enumerate(self.m._failure_set))
                                                                            +gp.LinExpr(self.m._b[m],[self.m._psi[m][l] for l in self.m._psi[m]])
                                                                            +self.m._phi[m]) for m in self.m._ms)
                                                -gp.LinExpr(S_benders['pi'],self.m._fish_vec)
                                                -gp.LinExpr(S_benders['gamma'],self.m._a_f_vec)
                                                -gp.LinExpr(S_benders['delta'],self.m._delta_vec),
                                                GRB.MINIMIZE)

                else: #Instead of having the linearizations, we just work with the vertex enumeration
                    ## New upperbounding approach
                    for m in self.m._ms:
                        self.m.addLConstr(self.m._phi[m] + gp.LinExpr(self.m._delta_u[m],self.m._delta_vecs[m]) >= np.sum(self.m._delta_u[m]),name=f'Phi_epigraph_UB_{m}')
                        self.m.addLConstr(gp.LinExpr(np.ones(self.m._delta_M[m]+1),self.m._delta_vecs[m])==self.m._delta_M[m],name=f'Only_1_delta_UB_{m}')
                        vertices = self.m._node_tree[self.m._current_id]['vertices'][m]['vertices']
                        for m_i,(vector,_) in vertices.items():
                            N_x = len(vector)
                            self.m.addLConstr(N_x*self.m._delta[m][m_i]+2*gp.LinExpr(vector,self.m._full_vec)-gp.LinExpr(np.ones(N_x),self.m._full_vec) >= np.dot(np.ones(N_x),vector),name=f'Delta_norm_{m}_{m_i}')

                    self.m.setObjective(gp.quicksum([self.m._gen_cost[g]*gp.LinExpr(self.m._bin_vec_gen[g],[self.m._P_gen_out[g][j] for j in self.m._P_gen_out[g]]) for g in self.m._P_gen_out])
                                                +gp.LinExpr(self.m._loss_cost.to_list(),[self.m._delta_p[g] for g in self.m._delta_p])
                                                +gp.LinExpr(self.m._loss_cost.to_list(),[self.m._delta_m[g] for g in self.m._delta_m])
                                                +gp.LinExpr(self.m._switch_cost.to_list(),[self.m._z[l] for l in self.m._z] )
                                                +gp.quicksum(self.m._qs[m]*self.m._phi[m] for m in self.m._ms),GRB.MINIMIZE)

            else:
                self.m.setObjective(gp.quicksum([self.m._gen_cost[g]*gp.LinExpr(self.m._bin_vec_gen[g],[self.m._P_gen_out[g][j] for j in self.m._P_gen_out[g]]) for g in self.m._P_gen_out])
                                    +gp.LinExpr(self.m._loss_cost.to_list(),[self.m._delta_p[g] for g in self.m._delta_p])
                                    +gp.LinExpr(self.m._loss_cost.to_list(),[self.m._delta_m[g] for g in self.m._delta_m])
                                    +gp.LinExpr(self.m._switch_cost.to_list(),[self.m._z[l] for l in self.m._z] ),GRB.MINIMIZE)

        
        if lagrangian == None: #Fishing constraints for evaluation (we round to deal with numerical instability)
            
            if S_benders is None: # If we are in RSP, we ignore these constraints, since we will co-optimize them
                # print('We write the fishing constraints')
                for l in self.m._failure_set: # for the lines that as subject to failure (For RSP this is fine)
                    self.m.addConstr(self.m._a_fish[l] == np.round(self.m._a_in[l],0),name=f'Fishing_a_{l}')



                # for l in self.m._z_fish: # for the lines that are switchable
                if self.m._dro['z_type'] == 'T':
                    
                    if self.m._t != 0: #If we are not in the first stage, we fix the values of variables t -> T (for generated cuts)
                        #We fix the current stage in two-stage model
                        for l in self.m._switchable_lines:
                            self.m.addConstr(self.m._z[l] == np.round(node_tree[0]['x']['z_plan'][t][l],0),name=f'Fishing_z_{t}_{l}')
                        #We fix the future stages for means of adding the cutting planes
                        for time in range(t+1,self.m._T):
                            for plan_line in self.m._z_t[time]:
                                    self.m.addConstr(self.m._z_t[time][plan_line] == np.round(node_tree[0]['x']['z_plan'][time][plan_line],0),name=f'Fishing_z_{time}_{plan_line}')
                #regardless of the future fishing vars, we always fish the previous one (from SDDiP)
                for line in self.m._switchable_lines: 
                    self.m.addConstr(self.m._z_fish[line] == np.round(self.m._z_in[line],0),name=f'Fishing_z_{t-1}_{line}')
                
                for g in self.m._P_gen_in: #z_P_gen = x_P_gen(pn)
                    if self.m._bin_exp == None: #Relaxed iteration
                        self.m.addConstr(self.m._P_gen_fish[g] == np.round(self.m._P_gen_in[g],0),name=f'Fishing_P_gen_{g}')
                    else: #Binary iteration
                        for e in self.m._P_gen_in[g]:
                            self.m.addConstr(self.m._P_gen_fish[g][e] == np.round(self.m._P_gen_in[g][e],0),name=f'Fishing_P_gen_{g}_{e}')
                
                for l in self.m._P_limit_fish: #z_P_trans = x_P_trans(pn)
                    self.m.addConstr(self.m._P_limit_fish[l] == np.round(self.m._P_limit_in[l],0),name=f'Fishing_P_limit_{l}')

    def solve_problem(self,silent,override_cb=False,presolve=1):
        #stat_dict = {sc.__dict__[k]: k for k in sc.__dict__.keys() if k[0] >= 'A' and k[0] <= 'Z'}
        if silent == False:
            self.m.Params.LogToConsole = 1
        else:
            self.m.Params.LogToConsole = 0
        
        self.m.Params.Presolve = presolve #We assume we can presolve. When fixing solutions we do not presolve

        if self.m._callback == True:
            if override_cb == False: #for some pusing worst-case distrib we might want to suppress callbacks
                # print('We are solving with callback')
                self.m.optimize(self.m._cb)
        else:
            #print('We are solving without callback')
            self.m.optimize()
        return
    
    
    #Saving solutions and pushing them into the node data dictionary
    def push_solution(self,extract = False):

        
        #####x variables (inter state)
        #P_gen_out, P_gen_fish
        self.m._node_tree[self.m._current_id]['x']['P_gen_out'],self.m._node_tree[self.m._current_id]['z']['P_gen_fish'] = {},{}

        for gen in self.m._P_gen_out:
            if self.m._bin_exp == None:
                self.m._node_tree[self.m._current_id]['x']['P_gen_out'][gen],self.m._node_tree[self.m._current_id]['z']['P_gen_fish'][gen] = self.m._P_gen_out[gen].X,self.m._P_gen_fish[gen].X
            else:
                self.m._node_tree[self.m._current_id]['x']['P_gen_out'][gen],self.m._node_tree[self.m._current_id]['z']['P_gen_fish'][gen] = {},{}
                for i in self.m._P_gen_out[gen]:
                    self.m._node_tree[self.m._current_id]['x']['P_gen_out'][gen][i],self.m._node_tree[self.m._current_id]['z']['P_gen_fish'][gen][i] = np.round(self.m._P_gen_out[gen][i].X,0),np.round(self.m._P_gen_fish[gen][i].X,0)
            
        if extract == True: #For the gathering statistics we just car about the de-binarized data
            var_holder = {'P_gen':{},'P_limit':{},'z':{}}
            for gen in self.m._P_gen_out:
                curr_node = self.m._node_tree[self.m._current_id]['x']['P_gen_out'][gen]
                binary_vec = 2 ** np.arange(len(curr_node.values()))
                var_holder['P_gen'][gen] = np.dot(list(curr_node.values()),binary_vec)
        #decision variable vars
        self.m._node_tree[self.m._current_id]['x']['P_limit_out'],self.m._node_tree[self.m._current_id]['x']['z_out'],self.m._node_tree[self.m._current_id]['z']['P_limit_fish'],self.m._node_tree[self.m._current_id]['z']['z_fish'] = {},{},{},{}
        # r (previous a)
        self.m._node_tree[self.m._current_id]['r']['a_fish'] = {}

        for line in self.m._failure_set: #For those lines that are subject to failure we store transmission and availability
            self.m._node_tree[self.m._current_id]['x']['P_limit_out'][line],self.m._node_tree[self.m._current_id]['z']['P_limit_fish'][line],self.m._node_tree[self.m._current_id]['r']['a_fish'][line] = np.round(self.m._P_limit_out[line].X,0),np.round(self.m._P_limit_fish[line].X,0),np.round(self.m._a_fish[line].X,0)

        for line in self.m._switchable_lines: #For those lines that are switchable, we store switching decisions
            self.m._node_tree[self.m._current_id]['x']['z_out'][line],self.m._node_tree[self.m._current_id]['z']['z_fish'][line] = np.round(self.m._z_out[line].X,0),np.round(self.m._z_fish[line].X,0)
        
        ####### When in two stage we push the planned variables (only if root node) for use in later stages


        if self.m._dro['z_type'] == 'T':
            if self.m._t == 0: #We are at the root node
                self.m._node_tree[self.m._current_id]['x']['z_plan'] = {}
                #Pushing the first stage variables
                self.m._node_tree[self.m._current_id]['x']['z_plan'][0] = {}
                for line_pos in self.m._switchable_lines:
                    self.m._node_tree[self.m._current_id]['x']['z_plan'][0][line_pos] = np.round(self.m._z_out[line_pos].X,0)
                #Pushing the future values
                for time in range(self.m._t+1,self.m._T):
                    self.m._node_tree[self.m._current_id]['x']['z_plan'][time] = {}
                    for line_pos in self.m._switchable_lines:
                        self.m._node_tree[self.m._current_id]['x']['z_plan'][time][line_pos] = np.round(self.m._z_t[time][line_pos].X,0)
    

        #The complete switching decisions
         

        #transmission,theta, delta_p, delta_m
        self.m._node_tree[self.m._current_id]['y']['transmission'] = {}
        self.m._node_tree[self.m._current_id]['y']['z'] = {}
        for line in self.m._P_trans:
            self.m._node_tree[self.m._current_id]['y']['z'][line] = self.m._z[line].X
            self.m._node_tree[self.m._current_id]['y']['transmission'][line] = np.round(self.m._P_trans[line].X,2)
        self.m._node_tree[self.m._current_id]['y']['theta'],self.m._node_tree[self.m._current_id]['y']['delta_p'],self.m._node_tree[self.m._current_id]['y']['delta_m'] = {},{},{}
        for node in self.m._theta:
            self.m._node_tree[self.m._current_id]['y']['theta'][node],self.m._node_tree[self.m._current_id]['y']['delta_p'][node],self.m._node_tree[self.m._current_id]['y']['delta_m'][node] = self.m._theta[node].X,self.m._delta_p[node].X,self.m._delta_m[node].X

        if extract == True:
            var_holder['Transmission'] = self.m._node_tree[self.m._current_id]['y']['transmission']
            var_holder['all_z'] = self.m._node_tree[self.m._current_id]['y']['z']
            var_holder['z'] = self.m._node_tree[self.m._current_id]['x']['z_out']
            var_holder['P_limit'] = self.m._node_tree[self.m._current_id]['x']['P_limit_out']
            return var_holder

    #We push the specialized solution vector for the upper bounding procedure
    def push_solution_vector(self):
        self.m._node_tree[self.m._current_id]['var_vec'] = [self.m._in_vec,self.m.ObjVal]

    def update_UB(self): #Specialized for last period
        #Updating specific approximation
        # print(f'Mid stage with node {self.m._current_id} with objective value {self.m._node_tree[self.m._current_id]["var_vec"][1]} writing unto {self.m._parent_id}')
        next_key = len(self.m._node_tree[self.m._parent_id]['vertices'][self.m._current_id]['vertices'])
        self.m._node_tree[self.m._parent_id]['vertices'][self.m._current_id]['vertices'][next_key] = self.m._node_tree[self.m._current_id]['var_vec']
    
    #For integer cuts
    def push_I_cut(self):
        
        self.m._node_tree['check'].append((self.m._parent_id,self.m._in_vec,self.m.ObjVal))
        next_key = len(self.m._node_tree[self.m._parent_id]['cuts']['I'][self.m._current_id])
        self.m._node_tree[self.m._parent_id]['cuts']['I'][self.m._current_id][next_key] = {}
        self.m._node_tree[self.m._parent_id]['cuts']['I'][self.m._current_id][next_key]['x_bar'] = self.m._in_vec
        self.m._node_tree[self.m._parent_id]['cuts']['I'][self.m._current_id][next_key]['L'] = self.m.ObjVal

    def push_SB_cut(self,SB_pi):
        SB_L = self.m.ObjVal
        #These variables are naturally binary
        if self.m._bin_exp == None: # we have the continous representation of generation
            c_type = 'C'     
        else: #we have the binary representation
            c_type = 'B'         


        #Forming the Strong Benders cut depending on the stage (relaxed or not), is the particular key
        next_key = len(self.m._node_tree[self.m._parent_id]['cuts']['SB'][c_type][self.m._current_id])
        self.m._node_tree[self.m._parent_id]['cuts']['SB'][c_type][self.m._current_id][next_key] = {}
        self.m._node_tree[self.m._parent_id]['cuts']['SB'][c_type][self.m._current_id][next_key]['pi'] = SB_pi['pi']
        self.m._node_tree[self.m._parent_id]['cuts']['SB'][c_type][self.m._current_id][next_key]['L'] = SB_L
        self.m._node_tree[self.m._parent_id]['cuts']['SB'][c_type][self.m._current_id][next_key]['gamma'] = SB_pi['gamma']
        self.m._node_tree[self.m._parent_id]['cuts']['SB'][c_type][self.m._current_id][next_key]['delta'] = SB_pi['delta']
        # ---------- For Strong Bender cuts ----------------

    

    def fix_sol_LP(self,ip=False):
        #We use this as part of the strong benders cuts and to get the worst-case distrib
        #Fixing optimal solutions for the whole problem to get dual solutions

        #Fixing inter-stage variables
        for gen in self.m._P_gen_out:
            if self.m._bin_exp == None: #these ones are already continuous
                self.m._P_gen_out[gen].ub,self.m._P_gen_out[gen].lb = self.m._node_tree[self.m._current_id]['x']['P_gen_out'][gen],self.m._node_tree[self.m._current_id]['x']['P_gen_out'][gen]

                self.m._P_gen_fish[gen].ub,self.m._P_gen_fish[gen].lb = self.m._node_tree[self.m._current_id]['z']['P_gen_fish'][gen],self.m._node_tree[self.m._current_id]['z']['P_gen_fish'][gen]
            else:
                for e in self.m._P_gen_out[gen]:
                    self.m._P_gen_out[gen][e].vtype = GRB.CONTINUOUS
                    self.m._P_gen_out[gen][e].ub,self.m._P_gen_out[gen][e].lb = self.m._node_tree[self.m._current_id]['x']['P_gen_out'][gen][e],self.m._node_tree[self.m._current_id]['x']['P_gen_out'][gen][e]
                    
                    self.m._P_gen_fish[gen][e].vtype = GRB.CONTINUOUS
                    self.m._P_gen_fish[gen][e].ub,self.m._P_gen_fish[gen][e].lb = self.m._node_tree[self.m._current_id]['z']['P_gen_fish'][gen][e],self.m._node_tree[self.m._current_id]['z']['P_gen_fish'][gen][e]
                    
        l_idx = 0
        for line in self.m._P_trans: #We fix to the solutions we have in the model (cannot access node tree because we only partially store them)
            self.m._z[line].vtype = GRB.CONTINUOUS
            self.m._z[line].ub,self.m._z[line].lb = np.round(self.m._z[line].X,0),np.round(self.m._z[line].X,0)
            
            self.m._a[line].vtype = GRB.CONTINUOUS
            self.m._a[line].ub,self.m._a[line].lb = np.round(self.m._a[line].X,0),np.round(self.m._a[line].X,0)
            
            if line in self.m._switchable_lines: #Only fix the partial variables for the lagrangian procedure
                self.m._z_fish[line].vtype = GRB.CONTINUOUS
                self.m._z_fish[line].ub,self.m._z_fish[line].lb = self.m._node_tree[self.m._current_id]['z']['z_fish'][line],self.m._node_tree[self.m._current_id]['z']['z_fish'][line]
                
                if self.m._dro['z_type']=='T': #Fixing additional variables
                    self.m._e[line].vtype = GRB.CONTINUOUS
                    self.m._e[line].ub,self.m._e[line].lb = np.round(self.m._e[line].X,0),np.round(self.m._e[line].X,0)
                    
                    #The rest of variables (planned fishing) are already fixed, but we need to make them continuous
                    for time in range(self.m._t+1,self.m._T):#one set of additional vectors for each stage t -> T (at 0 we don't add fishing constraints because we decide there)
                        self.m._z_t[time][line].vtype = GRB.CONTINUOUS
                l_idx += 1
            
            if line in self.m._failure_set: #Only fix the partial variables for the lagrangian procedure 
                #for a
                self.m._a_fish[line].vtype = GRB.CONTINUOUS
                self.m._a_fish[line].ub,self.m._a_fish[line].lb = self.m._node_tree[self.m._current_id]['r']['a_fish'][line],self.m._node_tree[self.m._current_id]['r']['a_fish'][line]
                
                #for limit out
                self.m._P_limit_out[line].vtype = GRB.CONTINUOUS
                self.m._P_limit_out[line].ub,self.m._P_limit_out[line].lb = self.m._node_tree[self.m._current_id]['z']['P_limit_fish'][line],self.m._node_tree[self.m._current_id]['z']['P_limit_fish'][line]
                
                #for limit fish
                self.m._P_limit_fish[line].vtype = GRB.CONTINUOUS
                self.m._P_limit_fish[line].ub,self.m._P_limit_fish[line].lb = self.m._node_tree[self.m._current_id]['z']['P_limit_fish'][line],self.m._node_tree[self.m._current_id]['z']['P_limit_fish'][line]
                

        if ip == False: #We want to keep them free to take values
            #fixing the local control variables
            for line in self.m._P_trans:
                self.m._P_trans[line].ub,self.m._P_trans[line].lb = self.m._P_trans[line].X,self.m._P_trans[line].X
            
            for node in self.m._theta:
                self.m._theta[node].ub,self.m._theta[node].lb = self.m._theta[node].X,self.m._theta[node].X
                self.m._delta_p[node].ub,self.m._delta_p[node].lb = self.m._delta_p[node].X,self.m._delta_p[node].X
                self.m._delta_m[node].ub,self.m._delta_m[node].lb = self.m._delta_m[node].X,self.m._delta_m[node].X

            #Fixing the lower approximation variables if they we are not at a terminal node
            if self.m._final == False:
                #We assume that we will never use it during an upper bounding procedure
                for m in self.m._phi:
                    self.m._phi[m].ub,self.m._phi[m].lb = self.m._phi[m].X,self.m._phi[m].X
                    for l in self.m._psi[m]:
                        self.m._psi[m][l].ub,self.m._psi[m][l].lb = self.m._psi[m][l].X,self.m._psi[m][l].X
                        self.m._rho[m][l].ub,self.m._rho[m][l].lb = self.m._rho[m][l].X,self.m._rho[m][l].X

        self.m.update()
    
    def get_pi(self):

        SB_pi = {}
        SB_pi['pi'] = [self.m.getConstrByName(f'Fishing_P_limit_{l}').Pi for l in self.m._P_limit_fish]
        
        SB_pi['gamma'] = [self.m.getConstrByName(f'Fishing_a_{l}').Pi for l in self.m._failure_set]
        
        #We arrange the vectors with the current stage vector, plus the future stages in the others

        #This is the delta from t-1
        SB_pi['delta'] = [self.m.getConstrByName(f'Fishing_z_{self.m._t-1}_{l}').Pi for l in self.m._z_fish]
        
        if self.m._dro['z_type'] == 'T': #We add the duals for all next fishing variables
            #we get the current stage fishing vars (t)
            SB_pi['delta'] += [self.m.getConstrByName(f'Fishing_z_{self.m._t}_{l}').Pi for l in self.m._z_out]
            #We get the next stages fishing vars (t+1 -> T)
            for time in range(self.m._t+1,self.m._T):
                SB_pi['delta'] += [self.m.getConstrByName(f'Fishing_z_{time}_{l}').Pi for l in self.m._z_t[time]]
        if self.m._bin_exp == None: # we have the continous representation of generation
            c_type = 'C'
            SB_pi['pi'] += [self.m.getConstrByName(f'Fishing_P_gen_{g}').Pi for g in self.m._P_gen_in]
        
        else: #we have the binary representation
            c_type = 'B'
            SB_pi['pi'] += [self.m.getConstrByName(f'Fishing_P_gen_{g}_{e}').Pi for g in self.m._P_gen_in for e in self.m._P_gen_in[g]]
        return SB_pi

    
    def push_Q_cuts(self):
        #Pushing Q cuts to reoptimize and obtain dual variables
        
        if 'SB' in self.m._dro['cut_types']: #We use Strong benders cuts
            #Since we can always have at least one cut used for any relaxed or binary cut, we have to add both cuts
            if self.m._dro['z_type'] == 'T':
                if self.m._S_benders != None: #We are in the relaxation, and thus, we have to remove the previous stage fishing vars for the cuts
                    delta_vector = self.m._delta_vec[len(self.m._switchable_lines):]
                else:
                    delta_vector = self.m._delta_vec
            else:
                delta_vector = self.m._delta_vec
            #What we have to handle, is how we build them
            for cut_id,cut_data in self.m._Q_SB_C_cuts.items(): #(*****) Do we need to multiply? 
                m,cut,pi,delta,a,gamma,lagrangian = cut_data['m'],cut_data['cut'],cut_data['pi'],cut_data['delta'],cut_data['a'],cut_data['gamma'],cut_data['lagrangian']
                
                if self.m._bin_exp == None: #We have a continuous model 
                    written_cut = SB_cut(pi,delta,gamma,lagrangian,self.m._var_vec,a.values,self.m._phi[m],self.m._psi[m],delta_vector)
                else: #We have a binary model (we have to expand the continuous variables)
                    full_pi,current_idx = [],0
                    #We expand pi (*****this can be vectorized with slicing*****)
                    for pi_idx in range(len(self.m._failure_set)):
                        full_pi.append(pi[pi_idx])
                        current_idx += 1
                    for g in range(self.m._gen_num):
                        for _ in range(self.m._E_gen[g]):
                            full_pi.append(pi[current_idx])
                        current_idx += 1
                    # for pi_idx in range(len(self.m._switchable_lines)):
                    #     full_pi.append(pi[current_idx+pi_idx])
                    written_cut = SB_cut(full_pi,delta,gamma,lagrangian,self.m._var_vec,a.values,self.m._phi[m],self.m._psi[m],delta_vector)

                #pushing the continuous cut
                self.m.addLConstr(written_cut,name=f'Q_SB_C_cut_{m}_{cut}_{cut_id}')
            
            if self.m._bin_exp != None:#By assumption, the binary cuts are only used whenever we have finished the strengtened cuts
                for cut_id,cut_data in self.m._Q_SB_B_cuts.items():
                    m,cut,pi,delta,a,gamma,lagrangian = cut_data['m'],cut_data['cut'],cut_data['pi'],cut_data['delta'],cut_data['a'],cut_data['gamma'],cut_data['lagrangian']
                    written_cut = SB_cut(pi,delta,gamma,lagrangian,self.m._var_vec,a.values,self.m._phi[m],self.m._psi[m],delta_vector)
                    # self.m.addConstr(self.m._phi[m]-gp.LinExpr(pi,self.m._var_vec)+gp.LinExpr(a,[self.m._psi[m][l] for l in self.m._psi[m]]) >= gamma_a+lagrangian,name=f'Q_SB_cut_{m}_{cut}_{cut_id}')
                    self.m.addLConstr(written_cut,name=f'Q_SB_B_cut_{m}_{cut}_{cut_id}')
        if 'I' in self.m._dro['cut_types']:
            for cut_id,cut_data in self.m._Q_I_cuts.items():
                m,cut,x_bar,a,lagrangian = cut_data['m'],cut_data['cut'],cut_data['x_bar'],cut_data['a'],cut_data['lagrangian']
                written_cut = I_cut(self.m._phi[m],lagrangian,self.m._full_vec,x_bar,a.values,self.m._psi[m])
                # self.m.addConstr(self.m._phi[m]-gp.LinExpr(pi,self.m._var_vec)+gp.LinExpr(a,[self.m._psi[m][l] for l in self.m._psi[m]]) >= gamma_a+lagrangian,name=f'Q_SB_cut_{m}_{cut}_{cut_id}')
                self.m.addLConstr(written_cut,name=f'Q_I_cut_{m}_{cut}_{cut_id}')

        self.m.update()

    def push_P_worst_n_sample(self):
        inconsistency_count = 0
        for m in self.m._ms:
            for l in self.m._psi[m]:
                if self.m._psi[m][l].X == self.m._bigM_Lin:
                    inconsistency_count += 1
        # if inconsistency_count > 0:
        #     print(f'!!! ----- Warning {inconsistency_count} psi vars reached their set UB of {self.m._bigM_Lin} ----- !!')
        #We compute the worst case distribution
        distrib_count = 0
        for m,_ in self.m._scenario_tree[self.m._t+1][self.m._current_id].items():
            a_tilde,a_all,probs,tot_prob = [],[],[],0
            #Probability coming from the Strong Benders cuts (if used)
            if 'SB' in self.m._dro['cut_types']:
                #getting countinuous cuts
                for cut_id,cut_data in self.m._Q_SB_C_cuts.items():
                    mm,cut,a = cut_data['m'],cut_data['cut'],cut_data['a']
                    if mm == m:
                        prob = self.m.getConstrByName(f'Q_SB_C_cut_{m}_{cut}_{cut_id}').Pi
                        a_all.append(a)
                        if prob > 0:
                            qnm = self.m._qs[m]
                            tot_prob += prob/qnm
                            a_tilde.append(a),probs.append(prob/qnm)
                            
                #if we have finished the relaxed, iterations, we move to the binary
                if self.m._bin_exp != None:
                    for cut_id,cut_data in self.m._Q_SB_B_cuts.items():
                        mm,cut,a = cut_data['m'],cut_data['cut'],cut_data['a']
                        if mm == m:
                            prob = self.m.getConstrByName(f'Q_SB_B_cut_{m}_{cut}_{cut_id}').Pi
                            a_all.append(a)
                            if prob > 0:
                                qnm = self.m._qs[m]
                                tot_prob += prob/qnm
                                a_tilde.append(a),probs.append(prob/qnm)

            if 'I' in self.m._dro['cut_types']:
                for cut_id,cut_data in self.m._Q_I_cuts.items():
                    mm,cut,a = cut_data['m'],cut_data['cut'],cut_data['a']
                    if mm == m:
                        prob = self.m.getConstrByName(f'Q_I_cut_{m}_{cut}_{cut_id}').Pi
                        a_all.append(a)
                        if prob > 0:
                            qnm = self.m._qs[m]
                            tot_prob += prob/qnm
                            a_tilde.append(a),probs.append(prob/qnm)

            if np.abs(tot_prob - 1) > 0.1:
                distrib_count += 1
            
            self.m._nm_data[self.m._current_id][m]['Worst_distrib'] = {'a_tilde':a_tilde,'probs':probs}

            #Sampling one of the a's
            if len(self.m._nm_data[self.m._current_id][m]['Worst_distrib']['a_tilde']) == 0:
                samp_idx = rnd.choices(range(len(a_all)))
                a_sample = a_all[samp_idx[0]]
            else:
                # print('We are fine in worst-case prob')
                a_sample = self.m._nm_data[self.m._current_id][m]['Worst_distrib']['a_tilde'][rnd.choices(range(len(self.m._nm_data[self.m._current_id][m]['Worst_distrib']['a_tilde'])),weights=self.m._nm_data[self.m._current_id][m]['Worst_distrib']['probs'])[0]]
            #Pushing the a into the node tree
            self.m._node_tree[self.m._current_id]['a'][m] = a_sample
            # print(self.m._current_id,'->',m,'|',self.m._a_in.values,'->',a_sample.values)
        return 

    def push_copy(self):
        mod_copy = copy.copy(self)
        mod_copy.m = self.m.copy()
        return mod_copy
    
    #Decision-dependent ambiguity set
    #Will take as inputs the current state of the system (states) and the Power generation (P_gen_out)

    #A_m(x_n), where m \in C(n), with probability q_{nm}
    #P({0,1}) : E[a] \leq A_m x_n + b_m
    #           e^{\top} a_n \geq n_a - K

    def constructAb(self,states):
        b = np.zeros((len(self.m._failure_set),1))
        #Constructing A and b (will depend on the current states and the lines subject to failure, and its already possible failure)
        #For those lines that will not be affected by assumption, A is 0, and b is 1

        #We leverage the knowledge of the maximum possible transmission
        A = [] #Cardinality should match the size of |switching|+|Transmission|+|Generation| The last two also have to consider their binary expansion
        #We order them by insertion order (since the failure set might not be ordered or sequential)
        for l,l_tag in enumerate(self.m._failure_set): #Rows
            a_l = np.zeros(len(self.m._failure_set))#Columns
            val = states[l]
            if val == 0:
                b[l] = self.m._b_inf+self.m._A_inf
            elif val == 1:
                b[l] = self.m._b_inf+self.m._A_inf-0.2
            elif val == 2:
                b[l] = self.m._b_inf+self.m._A_inf-0.3
            a_l[l] = -(self.m._A_inf)
            A.append(a_l)

        A = np.matrix(A)
        return A,b


