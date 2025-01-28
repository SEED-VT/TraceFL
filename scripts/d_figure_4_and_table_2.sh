echo "      ****************** Differential Privacy (DP) Enabled FL (Figure-4 and Table-2) ******************"


device=cpu # cpu, cuda, mps  # mps is only for M chip macs 
client_gpu=1 
key_start='artifact_exp'
num_clients=10 # change this 
clients_per_round=4 # change this
num_rounds=2 # increase this
dirichlet_alpha=0.3
model="resnet18" # change model, 
dataset="mnist" # change dataset
dp_noise=0.001 # change differential privacy noise
dp_clip=15 # change differential privacy clip

python -m tracefl.main --multirun device=$device client_resources.gpus=$client_gpu  exp_key=$key_start model.name=$model dataset.name=$dataset num_clients=$num_clients num_rounds=$num_rounds clients_per_round=$clients_per_round dirichlet_alpha=$dirichlet_alpha noise_multiplier=$dp_noise clipping_norm=$dp_clip

