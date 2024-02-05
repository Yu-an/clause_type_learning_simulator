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
    help="e.g. configs/with_prosody.json"
)
args = parser.parse_args()
conf_file = args.conf_file
with open('configs/'+conf_file) as f:
    configs = json.loads(f.read())

output_dir = 'outputs/'+ configs.get("output_dir").strip('/')+'/'
os.makedirs(os.path.dirname(output_dir),exist_ok=True)
if deltas:
    start= configs.get("deltas").get("start",0)
    end = configs.get('deltas').get('end',110)
    interval = configs.get('deltas').get('interval',10)
    deltas = range(start,end,interval)
else:
    deltas = range(0,110,10)
baseline = configs.get("baseline",None)

with open(output_dir+'config.pkl','wb') as f:
    pickle.dumps(configs,f)
print('[DONE] Configs loaded, copied to ', output_dir)

train_models_with_parameters(input_data,output_dir, rounds,prosody,noise_source,deltas, baseline)


    