
# Gibbs sampler for the simpler model
# infering C from A and S (baseline: C from S)
# Initializing 3 values for C

import numpy as np
import pandas as pd
import pickle
import os


def xtab(*cols):
    """
    Cross-tab two columns
    Both columns have to be np.arrays
    """
    if not all(len(col) == len(cols[0]) for col in cols[1:]):
        raise ValueError("all arguments must be same size")

    if len(cols) != 2:
        raise TypeError("xtab() requires 2 numpy arrays")

    fnx1 = lambda q: len(q.squeeze().shape)
    if not all([fnx1(col) == 1 for col in cols]):
        raise ValueError("all input arrays must be 1D")
    wt =1

    uniq_vals_all_cols, idx = zip( *(np.unique(col, return_inverse=True) for col in cols) )
    shape_xt = [uniq_vals_col.size for uniq_vals_col in uniq_vals_all_cols]
    xt = np.zeros(shape_xt)
    dtype_xt = 'float'
    np.add.at(xt, idx, wt)
    return  xt


def sampleCfromAS(c_sampled, a, S):
    """Sample clause type labels from speech acts and syntax
    c_sampled (np.arrray): clause type labels from last round
    a: speech act labels, given
    S: morpho-syntactic (+prosody) features
    Output:
        c_sampled: new clause type labels after this round of sim
        posterior_all: posterior of the labels
        likelihood_all: likelihood of the labels
    """
    #hyperparameters
    gamma_0 = 1
    beta_0 = 1 
    
    #counts
    a_uniq_values, a_counts = np.unique(a, return_counts=True) 
    a_c_counts = xtab(a,c_sampled)
    S_c_s_counts= np.array([xtab(c_sampled,s) for s in S])
    c_uniq_values, c_counts = np.unique(c_sampled, return_counts=True)
    
    
    posterior_all = []
    likelihood_all = []
    for n in range(len(c_sampled)):
        #identify current values
        current_a = a[n]
        current_c = c_sampled[n]
        current_S = [s[n] for s in S]

        # drop the current value of c (c_i) and recount
        c_counts[current_c]-=1
        a_c_counts[current_a, current_c]-=1
        for j in range(len(S)):
            S_c_s_counts[j, current_c, current_S[j]]-=1

        prior =  [((beta_0 + a_c_counts[current_a][c_x])/(beta_0*len(c_uniq_values)+ a_counts[current_a])) for c_x in range(len(c_uniq_values))]

        likelihood = []
        for c_x in range(len(c_uniq_values)):
            likelihood_s = [((gamma_0 + S_c_s_counts[s,c_x, current_S[s]])/(2*gamma_0+c_counts[c_x])) for s in range(len(S))]
            likelihood.append(np.prod(likelihood_s))
        likelihood = np.array(likelihood)
        
        likelihood_final = likelihood/np.sum(likelihood)
        posterior = (prior*likelihood)/np.sum(prior*likelihood)

        new_c = np.random.multinomial(1, posterior).argmax()

        #add new value back to the sampled array
        c_sampled[n]=new_c
        posterior_all.append(posterior)
        likelihood_all.append(likelihood_final)

        #increase counts 
        c_counts[new_c] +=1
        a_c_counts[current_a,new_c] +=1
        for j in range(len(S)):
            S_c_s_counts[j, new_c, current_S[j]]+=1

    return c_sampled, posterior_all,likelihood_all        

def simulate_a(delta,a):
    """mix noise into speech act labels
    Args:
        delta (int): proportion of noise in input
        a (np.array): speech act labels before simulation (e.g. a_true)

    Returns:
        a_sim: simuated speech act labels
    """
    a_sim = a
    a_uniq, counts = np.unique(a,return_counts = True)
    chance = np.random.choice(2, size=len(a), p = [delta/100,1-(delta/100)])
    for n in range(len(chance)):
        if chance[n] == 0:
            a_sim[n] = np.random.randint(len(a_uniq))
        else:
            a_sim[n] = a[n]
   
    return a_sim

def simulate_s(delta,s):
    s_sim = simulate_a(delta,s)    
    return s_sim

def sampleCfromS(c_sampled, S):
    """sample clause type labels from morpho-syntax (and prosody) labels

    Args:
        c_sampled (np.array): an array
        S (list of np.array): _description_

    Returns:
        _type_: _description_
    """
    #hyperparameters
    gamma_0 = 1
    beta_0 = 1 
    
    #counts
    S_c_s_counts= np.array([xtab(c_sampled,s) for s in S])
    c_uniq_values, c_counts = np.unique(c_sampled, return_counts=True)
    posterior_all = []
    likelihood_all = []
    for n in range(len(c_sampled)):

        current_c = c_sampled[n]
        current_S = [s[n] for s in S]

        # drop the current value of c (c_i) and recount
        c_counts[current_c]-=1
        for j in range(len(S)):
            S_c_s_counts[j, current_c, current_S[j]]-=1

        prior = [beta_0]*len(c_uniq_values)

        likelihood = []
     
        for c_x in range(len(c_uniq_values)):
            likelihood_s = [((gamma_0 + S_c_s_counts[s,c_x, current_S[s]])/(2*gamma_0+c_counts[c_x])) for s in range(len(S))]
            likelihood.append(np.prod(likelihood_s))
        likelihood = np.array(likelihood)
        
        likelihood_final = likelihood/np.sum(likelihood)

        posterior = (prior*likelihood)/np.sum(prior*likelihood)

        new_c = np.random.multinomial(1, posterior).argmax()

        c_sampled[n]=new_c
        posterior_all.append(posterior)
        likelihood_all.append(likelihood_final)

        #increase counts 
        c_counts[new_c] +=1
        for j in range(len(S)):
            S_c_s_counts[j, new_c, current_S[j]]+=1
    return c_sampled, posterior_all, likelihood_all       
