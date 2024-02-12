from simulation import train_models_with_parameters
import pickle
import os
import json
import argparse

parser = argparse.ArgumentParser(
        description="Read model training parameters"
    )
parser.add_argument(
    "--conf_file",
    required=True,
    help="Configs to use under configs/ dir, e.g. with_prosody.json"
)
args = parser.parse_args()
conf_file = args.conf_file
with open('configs/'+conf_file) as f:
    configs = json.loads(f.read())

input_data = configs.get("input_data",'input_')
output_dir = 'outputs/'+ configs.get("output_dir")
rounds = configs.get('rounds',10)
prosody = configs.get('prosody','no')
noise_source = configs.get('noise_source','a')
os.makedirs(output_dir,exist_ok=True)
if configs.get("deltas"):
    start= configs.get("deltas").get("start",0)
    end = configs.get('deltas').get('end',110)
    interval = configs.get('deltas').get('interval',10)
    deltas = range(start,end,interval)
else:
    deltas = range(0,110,10)
baseline = configs.get("baseline",None)

with open(os.path.join(output_dir,'config.pkl'),'w') as f:
    json.dump(configs,f)
print('[DONE] Configs loaded, copied to ', output_dir)

train_models_with_parameters(input_data,output_dir, rounds,prosody,noise_source,deltas, baseline)


    