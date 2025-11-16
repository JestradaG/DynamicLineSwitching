import numpy as np
import itertools


def Q_a(a):
    if a == 1: #medium risk
        probs = [0.20,0.60,0.20]
    else: #a == 0
        probs = [0.70,0.20,0.1]
    return probs

def construct_children(parent):
    #Contructing children based on the current parent node
    children = []
    children_probs = []
    fix = {}
    branch = {}
    probs = {}
    p_sum = parent.sum()
    for idx,i in enumerate(parent): #Looping over each lines' state
        if i == 2:
            fix[idx] = 2 
        else:
            branch[idx] = [0,1,2]
            probs[idx] = Q_a(i)
    
    joint_probs = list(itertools.product(*list(probs.values())))
    for i,scenario in enumerate(list(itertools.product(*list(branch.values())))):
        child = np.zeros(len(parent))
        joint_prob = np.prod(joint_probs[i])
        for f in fix.keys():
            child[f] = fix[f]
        for idx,node in enumerate(scenario):
            child[list(branch.keys())[idx]] = node    
        children.append(child)
        children_probs.append(joint_prob)
    return children,children_probs

#Constructing scenario tree T
#We discretize the states in three (low risk:0, medium risk:1, high risk:2)
#We assume that we have the same states at each state

def construct_T(T,num_lines):
    t = 0
    counter = 0
    a_state = np.zeros(num_lines) #All lines working
    scenario_tree = {t:{None:{counter:(a_state,1)}}}
    for t in range(1,T):
        scenario_tree[t] = {}
        for grandparent_id in scenario_tree[t-1]:
            for parent_id in scenario_tree[t-1][grandparent_id]:
                scenario_tree[t][parent_id] = {}
                parent,_ = scenario_tree[t-1][grandparent_id][parent_id]
                children,children_probs = construct_children(parent)
                for child,child_prob in zip(children,children_probs):
                    counter += 1
                    scenario_tree[t][parent_id][counter] = (child,child_prob)
    return scenario_tree

