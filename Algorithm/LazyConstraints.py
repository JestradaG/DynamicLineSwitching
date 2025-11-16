from gurobipy import GRB
import copy
from Algorithm.HelperFunctions import get_failures
from Algorithm.CutConstructor import SB_cut,I_cut

'''
In this file we define the code that enables the use of 
lazy-constraints within the solution of the nodal_problem
'''

#This function takes care of the generation of cuts if we work with the underestimation problem
def cb_lower_approx(model,where):
    if where == GRB.Callback.POLLING:
        #Ignore polling callback
        pass
    if where == GRB.Callback.MIPSOL:
        #Taking partial solution to approximate cost-to-go function
        psi,phi,rho,P_limit = {},{},{},{} #Storing current psi variables value to compute worst-case a
        if model._dro['z_type'] == 'T':
            if model._S_benders != None: #We are in the relaxation, and thus, we have to remove the previous stage fishing vars for the cuts
                delta_vector = model._delta_vec[len(model._switchable_lines):]
            else:
                delta_vector = model._delta_vec
        else:
            delta_vector = model._delta_vec
        #Looping over the possible m \in C(n) (we can get it once and not recompute)
        for m,_ in model._scenario_tree[model._t+1][model._current_id].items():
            phi[m],psi[m],rho[m] = model.cbGetSolution(model._phi[m]),model.cbGetSolution(model._psi[m]),{}
            for l in model._failure_set:
                rho[m][l],P_limit[l] = model.cbGetSolution(model._rho[m][l]),model.cbGetSolution(model._P_limit_out[l])
            #Iterating over generated cuts
            
            #We gather the continuous and binary cuts 
            #(We assume that we first start some rounds of continuous and then binary) to avoid dealing with different coefficients for each bit
            if model._bin_exp == None:# If we are still in the relaxed stage, we only explore the continuous cuts
                if 'SB' in model._dro['cut_types']: #We use Strong benders cuts
                    m_SB_C_data = model._node_tree[model._current_id]['cuts']['SB']['C'][m]
                    for cut in range(len(m_SB_C_data)):
                        a,pi,delta,gamma,lagrangian = copy.copy(model._a_in),m_SB_C_data[cut]['pi'],m_SB_C_data[cut]['delta'],m_SB_C_data[cut]['gamma'],m_SB_C_data[cut]['L']
                        a = get_failures(a,model._K,gamma,psi[m],model._failure_set,model._line_num,model._a_SB_C_ref,model._SB_C_cut_count)
                        sb_cut = SB_cut(pi,delta,gamma,lagrangian,model._var_vec,a,model._phi[m],model._psi[m],delta_vector)
                        model.cbLazy(sb_cut)
                            
                        if model._register_cuts == True:
                            model._Q_SB_C_cuts[model._SB_C_cut_count] = {'m':m,'cut':cut,'a':a,'pi':pi,'delta':delta,'gamma':gamma,'lagrangian':lagrangian}
                            model._SB_C_cut_count += 1
                
            else: #We are in binary stage, we assume to continue with binary from this point onward
                #For continuous cuts, we have to repeat the coefficient for the generation, other elements stay put
                if 'SB' in model._dro['cut_types']: #We use stronger benders cuts
                    m_SB_C_data = model._node_tree[model._current_id]['cuts']['SB']['C'][m]
                    for cut in range(len(m_SB_C_data)):
                        a,pi,delta,gamma,lagrangian = copy.copy(model._a_in),m_SB_C_data[cut]['pi'],m_SB_C_data[cut]['delta'],m_SB_C_data[cut]['gamma'],m_SB_C_data[cut]['L']
                        full_pi,current_idx = [],0
                        #We expand pi to use continous cuts in binary mode (*****this can be vectorized with slicing*****)
                        for pi_idx in range(len(model._failure_set)):
                            full_pi.append(pi[pi_idx])
                            current_idx += 1
                        for g in range(model._gen_num):
                            for _ in range(model._E_gen[g]):
                                full_pi.append(pi[current_idx])
                            current_idx += 1
                        a = get_failures(a,model._K,gamma,psi[m],model._failure_set,model._line_num,model._a_SB_C_ref,model._SB_C_cut_count)

                        sb_cut = SB_cut(full_pi,delta,gamma,lagrangian,model._var_vec,a,model._phi[m],model._psi[m],delta_vector)

                        model.cbLazy(sb_cut)
                        if model._register_cuts == True:
                            model._Q_SB_C_cuts[model._SB_C_cut_count] = {'m':m,'cut':cut,'a':a,'pi':pi,'delta':delta,'gamma':gamma,'lagrangian':lagrangian}
                            model._SB_C_cut_count += 1
                    
                    #For binary cuts, we just directly use them as they are
                    m_SB_B_data = model._node_tree[model._current_id]['cuts']['SB']['B'][m]
                    for cut in range(len(m_SB_B_data)):
                        a,pi,delta,gamma,lagrangian = copy.copy(model._a_in),m_SB_B_data[cut]['pi'],m_SB_B_data[cut]['delta'],m_SB_B_data[cut]['gamma'],m_SB_B_data[cut]['L']
                        a = get_failures(a,model._K,gamma,psi[m],model._failure_set,model._line_num,model._a_SB_B_ref,model._SB_B_cut_count)
                        sb_cut = SB_cut(pi,delta,gamma,lagrangian,model._var_vec,a,model._phi[m],model._psi[m],delta_vector)
                        model.cbLazy(sb_cut)
                        if model._register_cuts == True:
                            model._Q_SB_B_cuts[model._SB_B_cut_count] = {'m':m,'cut':cut,'a':a,'pi':pi,'delta':delta,'gamma':gamma,'lagrangian':lagrangian}
                            model._SB_B_cut_count += 1
                            if model._dro['z_type'] == 'T':
                                model._Q_SB_B_cuts[model._SB_B_cut_count-1]['delta'] = delta

                if 'I' in model._dro['cut_types']: #We use Integer cuts (only in the binary stage)
                    m_I_data = model._node_tree[model._current_id]['cuts']['I'][m]
                    for cut in range(len(m_I_data)):
                        a,x_bar,lagrangian = copy.copy(model._a_in),m_I_data[cut]['x_bar'],m_I_data[cut]['L']
                        a = get_failures(a,model._K,None,psi[m],model._failure_set,model._line_num,model._a_I_ref,model._I_cut_count,'I',lagrangian)
                        i_cut = I_cut(model._phi[m],lagrangian,model._full_vec,x_bar,a,model._psi[m])
                        model.cbLazy(i_cut)
                        if model._register_cuts == True:
                            model._Q_I_cuts[model._I_cut_count] = {'m':m,'cut':cut,'a':a,'x_bar':x_bar,'lagrangian':lagrangian}
                            model._I_cut_count += 1

        # print('//////////////////////////////////////////')  
        model.update()
