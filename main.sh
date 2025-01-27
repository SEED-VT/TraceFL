client_gpu=0.2
device=cuda # cpu, cuda, mps  # mps is only for M chip macs 



#
# key_start='image_classification_exp'
# num_clients=100
# clients_per_round=10
# dirichlet_alpha="0.3"


# echo "      ****************** Image Classification Experiments ******************"
# img_models="resnet18"
# img_datasets="mnist"
# num_rounds=10
# python -m tracefl.main --multirun device=$device client_gpu=$client_gpu  exp_key=$key_start model.name=$img_models dataset.name=$img_datasets num_clients=$num_clients num_rounds=$num_rounds clients_per_round=$clients_per_round dirichlet_alpha=$dirichlet_alpha | tee -a logs/exp_$key_start.log




# faulty clients experiments

# num_rounds=5
# num_clients=10
# clients_per_round=10
# key_start="faulty_dirichlet"
# dist_type="non_iid_dirichlet"
# label2flip={1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:0,12:0,13:0} 
# max_server_data_size=2048
# faulty_clients_ids=["0"]
# dirichlet_alpha=0.7
# check_prov_cache=True


# echo "      ****************** Medical Image Faulty Clients Experiments ******************"
# model_names="densenet121"
# # dataset_names="pathmnist,organamnist,cifar10,mnist"
# dataset_names="mnist"
# python -m tracefl.main --multirun device=$device client_gpu=$client_gpu data_dist.dist_type=$dist_type dirichlet_alpha=$dirichlet_alpha label2flip=$label2flip noise_rate=$label2flip  faulty_clients_ids=$faulty_clients_ids exp_key=$key_start model.name=$model_names dataset.name=$dataset_names num_clients=$num_clients num_rounds=$num_rounds clients_per_round=$clients_per_round | tee -a logs/exp$key_start.log





echo "      ****************** 2 Differential Privacy Experiments ******************"
num_rounds=2
num_clients=10 #100
clients_per_round=2 #10
dp_noise=0.001 #0.012
dp_clip=15
client_gpu=1
python -m tracefl.main --multirun device=$device client_gpu=$client_gpu exp_key=DP-text model.name="openai-community/openai-gpt" dataset.name="dbpedia_14" num_clients=$num_clients num_rounds=$num_rounds clients_per_round=$clients_per_round dirichlet_alpha=0.3 noise_multiplier=$dp_noise clipping_norm=$dp_clip | tee -a logs/train_dp_exp_text.log






