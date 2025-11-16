import numpy as np
import gurobipy as gp

def SB_cut(pi,delta,gamma,lagrangian,var,a,phi,psi,delta_var):#A single function that creates the cut for use in the lazy and normal constraints
    gamma_a = np.dot(gamma,a)
    cut = phi +gp.LinExpr(pi,var) + gp.LinExpr(delta,delta_var) - gp.LinExpr(a,[psi[l] for l in psi])  >= (-gamma_a + lagrangian)
    # cut = (phi +gp.LinExpr(pi,var) - gp.LinExpr(a,[psi[l] for l in psi])) >= (-gamma_a + lagrangian)
    # print(cut)
    return cut

#I cut is pending
def I_cut(phi,phi_bar,x,x_bar,a,psi): #A single function that creates the integer cut
    #Putting everytihin in a single complete vector
    full_vars_bar = list(x_bar)
    full_vars = x
    cut = phi - 2*phi_bar*(gp.LinExpr(full_vars_bar,full_vars))+gp.LinExpr(np.ones(len(full_vars_bar))*phi_bar,full_vars) >= phi_bar - np.dot(np.ones(len(full_vars_bar))*phi_bar,full_vars_bar) - gp.LinExpr(a,[psi[l] for l in psi])
    #cut = phi - 2*phi_bar*(gp.LinExpr(full_vars_bar,full_vars)) >= phi_bar - np.dot(np.ones(len(full_vars_bar))*phi_bar,full_vars_bar) - gp.LinExpr(a,[psi[l] for l in psi])
    
    
    # cut = phi - 2*phi_bar*(gp.LinExpr(x_bar,x)+gp.LinExpr(delta_bar,delta)) >= phi_bar - np.dot(np.ones(len(x_bar))*phi_bar,x_bar)
    #cut = phi - 2*phi_bar*(gp.LinExpr(x_bar,x)+gp.LinExpr(delta_bar,delta)) + gp.LinExpr(np.ones(len(delta_bar))*phi_bar,delta) + gp.LinExpr(np.ones(len(x_bar))*phi_bar,x) >= phi_bar - np.dot(np.ones(len(x_bar))*phi_bar,x_bar) - np.dot(np.ones(len(delta_bar))*phi_bar,delta_bar) #- gp.LinExpr(a,[psi[l] for l in psi])
    # cut = phi - 2*phi_bar*(gp.LinExpr(x_bar,x)+ np.dot(a_bar,a)+gp.LinExpr(delta_bar,delta)) + gp.LinExpr(np.ones(len(x_bar))*phi_bar,x) + gp.LinExpr(np.ones(len(delta_bar))*phi_bar,delta) + np.dot(np.ones(len(a_bar))*phi_bar,a) >= phi_bar - np.dot(np.ones(len(x_bar))*phi_bar,x_bar) - np.dot(np.ones(len(delta_bar))*phi_bar,delta_bar) - np.dot(np.ones(len(a_bar))*phi_bar,a_bar) #- gp.LinExpr(a,[psi[l] for l in psi])
    #cut = phi - 2*phi_bar*(gp.LinExpr(x_bar,x)+gp.LinExpr(delta_bar,delta)) + gp.LinExpr(np.ones(len(x_bar))*phi_bar,x) + gp.LinExpr(np.ones(len(delta_bar))*phi_bar,delta)   >= phi_bar - np.dot(np.ones(len(x_bar))*phi_bar,x_bar) - np.dot(np.ones(len(delta_bar))*phi_bar,delta_bar) - gp.LinExpr(a,[psi[l] for l in psi])
    # print('agregando un fake integer cut')
    return cut

def B_cut(pi,gamma,lagrangian,var,a,phi,psi): #A constructor for the benders cuts (to use within the use of the restricted separation algorithm)
    gamma_a = np.dot(gamma,a)
    cut = (phi +gp.LinExpr(pi,var) - gp.LinExpr(a,[psi[l] for l in psi])) >= (-gamma_a + lagrangian)
    return cut

def L_cut(): #A constructor for the Lagrangian cuts (to use within the use of the restricted separation algorithm)
    cut = False
    return cut