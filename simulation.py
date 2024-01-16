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

def train_baseline_model(round_number,c_init, S:list, output_dir):
    c_sampled= c_init
    # sample c from morpho-syntactic (+prosody) features
    for k in tqdm(range(0, 5000)):
        c_sampled, posterior_all, likelihood_all = gibbs.sampleCfromS(c_sampled, S)
        if ((k+1) % 1000) == 0:
            c_filename = output_dir+f"sims/baseline_rounds/round_{str(round_number+1)}/iter_"+str(k+1)
            os.makedirs(os.path.dirname(c_filename),exist_ok=True)
            with open(c_filename, "wb") as g:
                np.save(g,c_sampled)
            print(k+1, " round have finished simulation")

def train_target_model(round_num,c_init, a_sim, delta, S_sim:list,output_dir):
    c_sampled= c_init
    # sample c from speech act and morpho-syntactic (prosody) info
    
    for m in tqdm(range(0, 5000)):
        c_sampled, posterior_all, likelihood_all = gibbs.sampleCfromAS(c_sampled, a_sim, S_sim)
        if ((m+1) % 1000) == 0:
            c_filename = output_dir+f"sims/target_rounds/round_{str(round_num+1)}/level_{delta}/iters/iter_"+str(m+1)
            os.makedirs(os.path.dirname(c_filename),exist_ok=True)
            with open(c_filename, "wb") as g:
                np.save(g,c_sampled)
            print(f'{m+1} iteration for noise level {delta} at round {round} has finished')


def train_models_with_parameters(args):
    output_dir = 'outputs/'+ args.output_dir +'/'
    rounds = args.rounds
    prosody = args.prosody
    noise_source = args.noise_source
    mode = args.mode
    
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
    
    
    readme += '## Model specification'
    if noise_source == 'a':
        deltas = range(0,110,10)        
        readme += '- speech act labels were mixed with noise;\n'
    elif noise_source =='S':
        deltas = range(0,110,10)        
        readme += '- morpho-syntax labels were mixed with noise;\n'
    else:
        deltas = [0]
        readme += '- labels were not mixed with noise;\n'
    print(len(deltas),' noise level will be sampled!')
    
    if mode =='baseline':
        readme += '- only baseline model (infer with morpho-syntax features) was trained;\n'
        for i in tqdm(range(rounds)):
            train_baseline_model(i,c_init, S, output_dir)
        print('baseline model finished training!')
    elif mode == 'target':
        readme += '- only target model was trained; baseline model might be available;\n'
        for i in tqdm(range(rounds)):
            for delta in deltas:
                a_sim = gibbs.simulate_a(delta,a_true)
                train_target_model(i,c_true, a_sim, delta, S, output_dir)
        print('target model finished training!')
    else:
        for i in tqdm(range(rounds)):
            train_baseline_model(i,c_true, S, output_dir)
        print('baseline model finished training!')
        for i in tqdm(range(rounds)):
            for delta in deltas:
                train_target_model(i,c_true, a_true, delta, S, output_dir)
        print('target model finished training!')
        readme += '- both baseline and target model were trained;\n'
                
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


