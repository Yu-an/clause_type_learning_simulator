"""
Simulating the learning of clause type categories
With unsurpervised learning method

python simulation.py \
    --output_dir 'A_without_prosody' \
    --rounds 12 \
    --mode 'baseline' \
    --noise 'yes' \
    --prosody 'yes'
"""

import gibbs
import numpy as np
import pickle
import argparse
import os
from tqdm import tqdm
import analysis

FEATURES = ['Subj', 'Obj', 'Verb', 'Aux', 'AuxInvert', 
     'InitFunction', 'PreVFunction', 'PostVFunction', "final_rise","VerbMorphology"]
def load_data():
    with open('input_data/training_data', 'rb') as f:
            a_true = np.load(f,allow_pickle=True)
            c_true = np.load(f,allow_pickle=True)
            s0 = np.load(f,allow_pickle=True)
            s1 = np.load(f,allow_pickle=True)
            s2 = np.load(f,allow_pickle=True)
            s3 = np.load(f,allow_pickle=True)
            s4 = np.load(f,allow_pickle=True)
            s5 = np.load(f,allow_pickle=True)
            s6 = np.load(f,allow_pickle=True)
            s7 = np.load(f,allow_pickle=True)
            s8 = np.load(f,allow_pickle=True)
            s9 = np.load(f,allow_pickle=True)
    print('data is loaded!')
    return a_true, c_true, s0,s1,s2,s3,s4,s5,s6,s7,s8,s9

def train_baseline_model(c_init, S:list, iter_dir):
    c_sampled= c_init
    # sample c from morpho-syntactic (+prosody) features
    for k in tqdm(range(0, 5000)):
        c_sampled, posterior_all, likelihood_all = gibbs.sampleCfromS(c_sampled, S)
        if ((k+1) % 1000) == 0:
            c_filename = iter_dir+f"/iter_"+str(k+1)
            os.makedirs(os.path.dirname(c_filename),exist_ok=True)
            with open(c_filename, "wb") as g:
                np.save(g,c_sampled)
            print(k+1, " round have finished simulation")

def train_target_model(c_init, a, S:list,iter_dir):
    c_sampled= c_init
    # sample c from speech act and morpho-syntactic (prosody) info    
    for m in tqdm(range(0, 5000)):
        c_sampled, posterior_all, likelihood_all = gibbs.sampleCfromAS(c_sampled, a, S)
        if ((m+1) % 1000) == 0:
            c_filename = iter_dir+f"/iters/iter_"+str(m+1)
            os.makedirs(os.path.dirname(c_filename),exist_ok=True)
            with open(c_filename, "wb") as g:
                np.save(g,c_sampled)
            print(f'{m+1} iteration has finished')

def train_noisy_S_models_simplied(rounds,output_dir,c_init,c_true,a_true,S):
    """mix noise in morphosyntactic features (simplifed)
    noise mixing method (non-product):
    - Each feature noise at delta, for delta in range(0,110,10) level
    - Other features full knowledge (0% noise) in all 
    Reason for non-product:
    - Full permutation will vary 10**10 times; could do but might not be necessary
    """
    for s_x in S:
        for delta in range(0,110,10):
            s_x_sim = gibbs.simulate_s(delta,s_x)
            
            for i in tqdm(range(rounds)):
                iter_dir = f'{output_dir}/sims/baseline_rounds/{delta}_percent_noise/round_{str(i+1)}'
                os.makedirs(os.path.dirname(iter_dir),exist_ok=True)
                train_baseline_model(c_init, S_sim, output_dir)
            print(f'baseline model with noise level {delta} finished training!')
            for i in tqdm(range(rounds)):                        
                iter_dir = f'{output_dir}/sims/target_rounds/{delta}_percent_noise/round_{str(i+1)}'
                os.makedirs(os.path.dirname(iter_dir),exist_ok=True)
                train_target_model(c_true, a_true, S_sim, output_dir)
            print('target model finished training!')   
    

def train_models_with_parameters(args):
    output_dir = 'outputs/'+ args.output_dir +'/'
    rounds = args.rounds
    prosody = args.prosody
    noise_source = args.noise_source
    
    
    readme = '##Simulation Report\n\n'
    readme+= '###Parameters:\n'
    a_true, c_true, s0,s1,s2,s3,s4,s5,s6,s7,s8,s9 = load_data()
    for feature in FEATURES:
        if feature =='final_rise':
            if prosody:
                print('Prosodic feature will be used!')
                S = [s0,s2,s3,s4,s5,s6,s7,s8,s9]
                readme+=f'- {feature}\n'                
        else:
            readme+=f'- {feature}\n'
            S = [s0,s2,s3,s4,s5,s6,s7,s9]
            
    readme += '###Training specifications\n'
    if rounds:
        rounds = int(rounds)        
        readme += f'- {rounds} rounds of sampling were performed, each round with 5000 iterations;\n'
    else:
        rounds = 10
        readme += '- 10 rounds of sampling were performed, , each round with 5000 iterations;\n'    
    print(rounds,' rounds will be trained!')
    
    # Model initialization
    c_init = np.random.randint(3, size=len(c_true))
    
    # Model training
    readme += '## Model specification'
    if noise_source == 'a':
        deltas = range(0,110,10)
        for delta in tqdm(deltas):
            a_sim = gibbs.simulate_a(delta,a_true)
            for i in tqdm(range(rounds)):
                iter_dir = f'{output_dir}/sims/baseline_rounds/{delta}_percent_noise/round_{str(i+1)}'
                os.makedirs(os.path.dirname(iter_dir),exist_ok=True)
                train_baseline_model(c_init, S, iter_dir)
            print(f'baseline model with noise level {delta} finished training!')
            for i in tqdm(range(rounds)):                        
                iter_dir = f'{output_dir}/sims/target_rounds/{delta}_percent_noise/round_{str(i+1)}'
                os.makedirs(os.path.dirname(iter_dir),exist_ok=True)
                train_target_model(c_true, a_sim, S, output_dir)
            print('target model finished training!')                                
        readme += '- speech act labels were mixed with noise;\n'
    elif noise_source =='S':
        train_noisy_S_models_simplied(rounds,output_dir,c_init,c_true,a_true,S)
        readme += '- morpho-syntax labels were mixed with noise;\n'
        readme += '-- noise-mixing method: all but one feature mix with noise\n'
    elif ('a' in noise_source) and ('S' in noise_source):
        
        readme += '- both speech act and morpho-syntax labels were mixed with noise;\n'
    else:        
        readme += '- labels were not mixed with noise;\n'
        for i in tqdm(range(rounds)):
            iter_dir = f'{output_dir}/sims/baseline_rounds/round_{str(i+1)}'
            os.makedirs(os.path.dirname(iter_dir),exist_ok=True)                                        
            train_baseline_model(c_init, S, iter_dir)
        print('baseline model finished training!')
        for i in tqdm(range(rounds)):
            iter_dir = f'{output_dir}/sims/target_rounds/round_{str(i+1)}'
            os.makedirs(os.path.dirname(iter_dir),exist_ok=True)                                        
            train_target_model(c_true, a_true, S, iter_dir)
        print('target model finished training!')
    
                
    with open(output_dir + 'README.md','w') as f:
        f.write(readme)
    
def main():
    parser = argparse.ArgumentParser(
        description="Read model training parameters"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="name of the output folder; will be placed under 'output/' dir"
    )
    parser.add_argument(
        "--rounds",
        required=False,
        help="how many rounds of simulation to be run, default 10"
    )
    parser.add_argument(
        "--prosody",
        required=False,
        help="with or without prosody, default without"
    )
    parser.add_argument(
        "--noise_source",
        required=False,
        help="source of noise, 'a' for speech act, 'S' for morpho-syn, default no noise"
    )
    parser.add_argument(
        "--mode",
        required=False,
        help="possible values: baseline, target; both will be trained if not specified"
    )
    args = parser.parse_args()
    
    train_models_with_parameters(args)
    

if __name__ == "__main__": 
    main()


