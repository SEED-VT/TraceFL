echo "      ****************** Varying Data Distribution Figure-3 ******************"


device=cuda # cpu, cuda, mps  # mps is only for M chip macs 
client_gpu=1 
key_start='artifact_exp'
num_clients=10 # change this 
clients_per_round=4 # change this
num_rounds=2 # increase this
# dirichlet_alpha="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1"
dirichlet_alpha=0.3 # change this
model="resnet18" # change model, 
dataset="mnist" # change dataset

python -m tracefl.main --multirun device=$device client_resources.gpus=$client_gpu  exp_key=$key_start model.name=$model dataset.name=$dataset num_clients=$num_clients num_rounds=$num_rounds clients_per_round=$clients_per_round dirichlet_alpha=$dirichlet_alpha 