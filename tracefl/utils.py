"""Define any utility function.

They are not directly relevant to  the other (more FL specific) python modules. For
example, you may define here things like: loading a model from a checkpoint, saving
results, plotting.
"""


from sklearn import metrics
import numpy as np


def get_backend_config(cfg): 

    client_resources = {"num_cpus": cfg.client_resources.cpus}
    if cfg.device == "cuda":
        client_resources['num_gpus'] = cfg.client_resources.gpus 
            

    backend_config = {
        "client_resources": client_resources,
        "init_args": {"num_cpus": cfg.total_cpus,
                "num_gpus": cfg.total_gpus},
    }
    return backend_config



def compute_importance(n_elements, decay_factor=0.9):
    # Generate indices for the elements
    indices = np.arange(n_elements)
    importance = decay_factor ** indices[::-1]
    return importance.tolist()


def set_exp_key(cfg):

    if cfg.noise_multiplier == -1 and cfg.clipping_norm == -1:

        key = (f"{cfg.exp_key}-"
               f"{cfg.model.name}-{cfg.dataset.name}-"
               f"faulty_clients[{cfg.faulty_clients_ids}]-"
               f"noise_rate{cfg.noise_rate}-"
               f"TClients{cfg.data_dist.num_clients}-"
               f"{cfg.strategy.name}-(R{cfg.strategy.num_rounds}"
               f"-clientsPerR{cfg.strategy.clients_per_round})"
               f"-{cfg.data_dist.dist_type}{cfg.data_dist.dirichlet_alpha}"
               f"-batch{cfg.data_dist.batch_size}-epochs{cfg.client.epochs}-"
               f"lr{cfg.client.lr}"
               )
        return key
    elif cfg.noise_multiplier >= 0 and cfg.clipping_norm >= 0:
        dp_key = (f"DP-(noise{cfg.noise_multiplier}+clip{cfg.clipping_norm})-{cfg.exp_key}-"
                  f"{cfg.model.name}-{cfg.dataset.name}-"
                  f"faulty_clients[{cfg.faulty_clients_ids}]-"
                  f"noise_rate{cfg.noise_rate}-"
                  f"TClients{cfg.data_dist.num_clients}-"
                  f"{cfg.strategy.name}-(R{cfg.strategy.num_rounds}"
                  f"-clientsPerR{cfg.strategy.clients_per_round})"
                  f"-{cfg.data_dist.dist_type}{cfg.data_dist.dirichlet_alpha}"
                  f"-batch{cfg.data_dist.batch_size}-epochs{cfg.client.epochs}-"
                  f"lr{cfg.client.lr}"
                  )
        return dp_key
    else:
        raise ValueError("Invalid config")


def get_prov_eval_metrics(labels, predicted_labels):
    
    accuracy = metrics.accuracy_score(labels, predicted_labels)
    

    answer = {    
        "Accuracy": accuracy,
    }

    return answer
