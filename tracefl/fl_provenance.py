import logging
import time
import torch 
from diskcache import Index
from tracefl.dataset import get_clients_server_data
from tracefl.fl_provenance_modules import FederatedProvTrue
from tracefl.utils import get_prov_eval_metrics
from tracefl.models import test_neural_network
from tracefl.neuron_provenance import NeuronProvenance
from tracefl.differential_testing import round_lambda_fed_debug_func

class FederatedProvFalse(FederatedProvTrue):
    def __init__(self, cfg, round_key, central_test_data, t=None):
        """
        Initialize a FederatedProvFalse instance.

        Parameters
        ----------
        cfg : object
            Configuration object.
        round_key : str
            Key identifying the training round.
        central_test_data : object
            Central test dataset.
        t : optional
            Additional parameter.
        """
        super().__init__(cfg, round_key, central_test_data, t)
    
    def _sanityCheck(self):
        """
        Perform a sanity check on the test dataset.

        Returns
        -------
        float or None
            The accuracy on the subset of test data if non-empty; otherwise, None.

        Raises
        ------
        AssertionError
            If the sanity check fails.
        """
        if len(self.subset_test_data) == 0:
            return None

        acc = test_neural_network(self.train_cfg.model.arch, {
                                  'model': self.prov_global_model}, self.subset_test_data)["accuracy"]
        logging.info(f"Sanity check: {acc}")
        assert int(acc) == 0, "Sanity check failed"
        
        return acc

    def _setParticipatingClientsLabels(self) -> None:
        """
        Aggregate and set the labels for participating clients.

        Returns
        -------
        None
        """
        labels = set()
        for c in self.client2model.keys():
            labels = labels.union(self.ALLROUNDSCLIENTS2CLASS[c])

        self.participating_clients_labels = list(labels)
        logging.debug(
            f"participating_clients_labels: {self.participating_clients_labels}"
        )

    def _eval_and_extract_wrong_preds(self, test_data):
        """
        Evaluate the model on test data and extract indices and labels for incorrect predictions.

        Parameters
        ----------
        test_data : object
            Test dataset to be evaluated.

        Returns
        -------
        tuple
            A tuple containing:
            - torch.Tensor: Indices of incorrectly predicted samples.
            - torch.Tensor: Actual labels of the samples.
            - torch.Tensor: Predicted labels of the samples.
        """
        d = test_neural_network(self.train_cfg.model.arch, {
                                'model': self.prov_global_model}, test_data)  # type: ignore
        self.loss = d["loss"]
        self.acc = d["accuracy"]
        logging.debug(f"Accuracy on test data: {self.acc}")
        return d["eval_incorrect_indices"], d["eval_actual_labels"], d["eval_predicted_labels"]
    
    def _selectProvenanceData(self, central_test_data, min_per_label: int = 10) -> None:
        """
        Select a subset of test data for provenance analysis based on incorrect predictions.

        Parameters
        ----------
        central_test_data : object
            The complete central test dataset.
        min_per_label : int, optional
            Minimum number of samples per label to select (default is 10).

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If no faulty clients are specified in the configuration.
        """        
        label2flip = self.train_cfg.label2flip
        assert len(self.train_cfg.faulty_clients_ids) > 0, "No faulty clients"
        all_wrong_i, dataset_lablels, predicted_labels = self._eval_and_extract_wrong_preds(central_test_data)
        logging.info(f"Total wrong predictions: {len(all_wrong_i)}")
        logging.info(f'label2flip {label2flip}')
        logging.info(f'Actual labels : {dataset_lablels}')
        logging.info(f'Predicted labels : {predicted_labels}')
        selected_wrong_indices = []
        for index_i in all_wrong_i:
            predicted_label = int(predicted_labels[index_i]) 
            true_label =  int(dataset_lablels[index_i])

            if predicted_label in  [int(l) for l in label2flip.values()]  and true_label in [int(l) for l in label2flip.keys()]:# and true_label in label2flip.keys():
                selected_wrong_indices.append(index_i)

        selected_wrong_indices  = selected_wrong_indices[:min_per_label]    
        self.subset_test_data = central_test_data.select(selected_wrong_indices)
        logging.info(f'Selected Actual labels : {dataset_lablels[selected_wrong_indices]}')
        logging.info(f'Selected Predicted labels : {predicted_labels[selected_wrong_indices]}')

        if len(self.subset_test_data) == 0:
            logging.info("No correct predictions found")


    def _computeEvalMetrics(self, input2prov) :
        """
        Compute evaluation metrics based on provenance information.

        Parameters
        ----------
        input2prov : list of dict
            List of provenance data dictionaries.

        Returns
        -------
        dict
            Dictionary of evaluation metrics.
        """
        data_loader = torch.utils.data.DataLoader(self.subset_test_data, batch_size=1)
        
        target_labels = [data['label'].item() for data in data_loader]

        client2class = {c: self.ALLROUNDSCLIENTS2CLASS[c] for c in self.client2model}

        logging.debug(f"client2class: {client2class}")

        correct_tracing = 0

        true_labels = []
        predicted_labels = []

        for idx, prov_r in enumerate(input2prov):
            traced_client = prov_r["traced_client"]
            client2prov = prov_r["client2prov"]

            target_l = target_labels[idx]
            responsible_clients = [f"{c}" for c in self.train_cfg.faulty_clients_ids]

            res_c_string = ','.join(map(str, [f"c{c}" for c in responsible_clients]))

            logging.info(
                f'            *********** Input Label: {target_l}, Responsible Client(s): {res_c_string}  *************')

            if traced_client in responsible_clients:
                logging.info(
                    f"     Traced Client: c{traced_client} || Tracing = Correct")
                correct_tracing += 1
                predicted_labels.append(1)
                true_labels.append(1)
            else:
                logging.info(

                    f"     Traced Client: c{traced_client} || Tracing = Wrong")
                predicted_labels.append(0)
                true_labels.append(1)

            c2nk_label = {f'c{c}': client2class[c].get(  # type: ignore
                target_l, 0) for c in client2prov.keys()}
            c2nk_label = {c: v for c, v in c2nk_label.items() if v > 0}

            client2prov_score = {f'c{c}': round(
                p, 2) for c, p in client2prov.items()}
            logging.info(f"    TraceFL Clients Contributions Rank:     {client2prov_score}")
            logging.info('\n')

        eval_metrics = get_prov_eval_metrics(true_labels, predicted_labels)
        a = correct_tracing / len(input2prov)
        assert a == eval_metrics['Accuracy'], "Accuracy mismatch"
        return eval_metrics

    def run(self): 
        """
        Execute provenance analysis and return the results.

        Returns
        -------
        dict
            Dictionary containing provenance analysis results.
        """        
        r = self._sanityCheck()
        if r is None:
            prov_result = {'Error': "No data found for provenance analysis"}
            return prov_result
            

        sart_time = time.time()
        nprov = NeuronProvenance(cfg=self.prov_cfg, arch=self.train_cfg.model.arch, test_data=self.subset_test_data, gmodel=self.prov_global_model,  # type: ignore
                                 c2model=self.client2model, num_classes=self.train_cfg.dataset.num_classes, c2nk=self.client2num_examples)  # type: ignore
        input2prov = nprov.computeInputProvenance()
        eval_metrics = self._computeEvalMetrics(input2prov)
        end_time = time.time()

        logging.info(f"[Round {self.round_id}] TraceFL Localization Accuracy = {eval_metrics['Accuracy'] *100} || Total Inputs Used In Prov: {len(self.subset_test_data)} || GM_(loss, acc) ({self.loss},{self.acc})")

        fed_debug_result = {}

        if self.prov_cfg.model.arch == 'cnn':
            temp_dict =  round_lambda_fed_debug_func(self.prov_cfg, self.round_key)
            fed_debug_result['FedDebug Accuracy'] = temp_dict['eval_metrics']['accuracy']
            fed_debug_result['FedDebug avg_fault_localization_time'] = temp_dict['avg_fault_localization_time']
            fed_debug_result['FedDebug avg_input_gen_time'] = temp_dict['avg_input_gen_time']

        prov_result = {
            "clients": list(self.client2model.keys()),
            "data_points": len(self.subset_test_data),
            "eval_metrics": eval_metrics,
            "test_data_acc": self.acc,
            "test_data_loss": self.loss,
            "avg_prov_time_per_input": (end_time - sart_time)/ len(self.subset_test_data),
            "round_id": self.round_id,
            'prov_layers': self.save_prov_layers,
        }
        prov_result = {**prov_result, **fed_debug_result}
        return prov_result


def _get_round_keys(fl_key, train_cache_path):
    """
    Retrieve round keys from the training cache that correspond to a given federated learning key.

    Parameters
    ----------
    fl_key : str
        Federated learning configuration key.
    train_cache_path : str
        Path to the training cache.

    Returns
    -------
    list
        List of round keys.

    Raises
    ------
    ValueError
        If no round keys are found.
    """
    training_cache = Index(train_cache_path)
    r_keys = []
    for k in training_cache.keys():
        if fl_key == k:
            continue
        elif fl_key in k and len(k) > len(fl_key):
            r_keys.append(k)

    if len(r_keys) == 0:
        raise ValueError(
            f"No rounds found for key {fl_key}. Please check the training cache.")
    return r_keys


def _checkAlredyDone(fl_config_key: str, results_cache):
    """
    Check if provenance results are already computed and stored in the results cache.

    Parameters
    ----------
    fl_config_key : str
        Federated learning configuration key.
    results_cache : diskcache.Index
        Results cache.

    Returns
    -------
    list
        List of provenance results if available; otherwise, an empty list.
    """    
    if fl_config_key in results_cache:
        d = results_cache[fl_config_key]
        return d["round2prov_result"]
    return []


def _roundLambdaProv(cfg, round_key, central_test_data):
    """
    Compute provenance for a given round using the appropriate provenance module.

    Parameters
    ----------
    cfg : object
        Configuration object.
    round_key : str
        Key for the training round.
    central_test_data : object
        Central test dataset.

    Returns
    -------
    dict
        Provenance result dictionary.
    """    
    if len(cfg.faulty_clients_ids) > 0:
        round_prov = FederatedProvFalse(
            cfg, round_key, central_test_data, t=None)
    else:
        round_prov = FederatedProvTrue(
            cfg, round_key, central_test_data, t=None)
    try:
        prov_result_dict = round_prov.run()
    except Exception as e:
        logging.error(
            f"Error in running provenance for round {round_key}. Error: {e}")
        prov_result_dict = {'Error': e}

    return prov_result_dict


def _run_and_save_prov_result_in_cache(cfg):
    """
    Run provenance analysis for all rounds and save the results in the results cache.

    Parameters
    ----------
    cfg : object
        Configuration object.

    Returns
    -------
    None
    """    
    round2prov_result = []
    train_cache_path = cfg.storage.dir + cfg.storage.train_cache_name
    prov_results_cache = Index(
        cfg.storage.dir + cfg.storage.results_cache_name)

    if cfg.check_prov_cache:
        round2prov_result = _checkAlredyDone(cfg.exp_key, prov_results_cache)
        # print(f"round2prov_result {round2prov_result}")
        if len(round2prov_result) > 0:
            logging.info(
                f">> Done.Provenance of key {cfg.exp_key} is already done.")
            return round2prov_result

    logging.info(f"Starting provenance analysis for {cfg.exp_key}...")
    rounds_keys = _get_round_keys(cfg.exp_key, train_cache_path)

    central_test_data = get_clients_server_data(cfg)['server_testdata']

    # total test data size
    logging.info(f"Total test data size: {len(central_test_data)}")
    start_time = time.time()
    round2prov_result = [_roundLambdaProv(
        cfg, round_key, central_test_data) for round_key in rounds_keys]

    end_time = time.time()
    avg_prov_time_per_round = -1

    if len(rounds_keys) > 0:
        avg_prov_time_per_round = (end_time - start_time) / len(rounds_keys)
    
    prov_results_cache[cfg.exp_key] = {
        "round2prov_result": round2prov_result, "prov_cfg": cfg, "training_cache_path": train_cache_path, "avg_prov_time_per_round": avg_prov_time_per_round}


def _get_all_train_cfgs_from_train_cache(cache):
    """
    Retrieve all training configurations from the training cache.

    Parameters
    ----------
    cache : diskcache.Index
        Training cache.

    Returns
    -------
    list
        List of training configuration objects.
    """    
    cfgs = []
    filter_keys = [key for key in cache.keys() if key.find("-round:") == -1]

    for key in filter_keys:
        if key.find('Temp--') != -1:
            continue
        logging.info(f"key: {key}")

        if "train_cfg" in cache[key] and cache[key]['complete']:
            cfgs.append(cache[key]["train_cfg"])
    return cfgs


def run_full_cache_provenance(cfg):
    """
    Execute provenance analysis for all training configurations in the training cache.

    Parameters
    ----------
    cfg : object
        Configuration object.

    Returns
    -------
    None
    """    
    all_train_cfgs = _get_all_train_cfgs_from_train_cache(
        Index(cfg.storage.dir + cfg.storage.train_cache_name))
    for train_cfg in all_train_cfgs:
        train_cfg.parallel_processes = cfg.parallel_processes
        train_cfg.check_prov_cache = cfg.check_prov_cache
        try:
            _run_and_save_prov_result_in_cache(train_cfg)
        except Exception as e:
            logging.error(f"Error: {e}")
            logging.error(
                f"Error in provenance experiment: {train_cfg.exp_key}")


def given_key_provenance(cfg):
    """
    Run provenance analysis for a specific key and save the results in the cache.

    Parameters
    ----------
    cfg : object
        Configuration object.

    Returns
    -------
    None
    """
    _run_and_save_prov_result_in_cache(cfg)
