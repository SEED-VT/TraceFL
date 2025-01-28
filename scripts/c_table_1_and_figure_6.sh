echo "      ****************** TraceFL’s Localization Accuracy in Mispredictions (Table-1) and (Figure-6) ******************"


device=cpu # cpu, cuda, mps  # mps is only for M chip macs 
client_gpu=1 
key_start='artifact_exp'
num_clients=10 # change this 
clients_per_round=10 # change this
num_rounds=3 # increase this
model="resnet18" # change model, 
dataset="mnist" # change dataset
dist_type="non_iid_dirichlet"
dirichlet_alpha=0.7
faulty_clients_ids=["0"]
label2flip={1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:0,12:0,13:0}



python -m tracefl.main --multirun device=$device client_resources.gpus=$client_gpu  exp_key=$key_start model.name=$model dataset.name=$dataset num_clients=$num_clients num_rounds=$num_rounds clients_per_round=$clients_per_round dirichlet_alpha=$dirichlet_alpha faulty_clients_ids=$faulty_clients_ids label2flip=$label2flip


