"""Fed_Debug Differential Testing."""
import copy
import gc
import itertools
import logging
import time
import torch
from typing import Any, Dict
import torch.nn.functional as F
from diskcache import Index
from torch.nn.init import kaiming_uniform_
from torchvision.transforms import Compose, Normalize, RandomHorizontalFlip, Resize
from tracefl.models import initialize_model
from tracefl.neuron_activation import get_neurons_activations


def _make_all_subsets_of_size_n(s, n):
    """
    Generate all subsets of a given set with a specified size.

    Parameters
    ----------
    s : iterable
        An iterable containing the elements from which to form subsets.
    n : int
        The size of each subset to generate. Must be less than the length of `s`.

    Returns
    -------
    list of set
        A list where each element is a set representing a subset of `s` of size `n`.

    Raises
    ------
    AssertionError
        If `n` is not less than the length of `s`.
    """   
    assert n < len(s)
    l_of_subsets = list(itertools.combinations(s, n))
    l_of_lists = [set(sub) for sub in l_of_subsets]
    return l_of_lists


class InferenceGuidedInputs:
    """
    Generate random inputs based on feedback from multiple client models.

    Parameters
    ----------
    clients2models : dict
        Dictionary mapping client identifiers to their corresponding models.
    input_shape : tuple
        The shape of the input tensor to generate.
    random_generator_func : callable
        Function used to generate random values for the inputs.
    transform_func : callable or None
        Function to transform the generated input. If None, no transformation is applied.
    k_gen_inputs : int, optional
        The number of inputs to generate (default is 10).
    min_nclients_same_pred : int, optional
        Minimum number of clients that should have the same prediction for the input to be accepted (default is 3).
    time_delta : int, optional
        Time interval (in seconds) used in the input generation process (default is 60).
    faster_input_generation : bool, optional
        Flag to select a faster input generation method (default is False).
    """
    def __init__(
        self,
        clients2models,
        input_shape,
        random_generator_func,
        transform_func,
        k_gen_inputs=10,
        min_nclients_same_pred=3,
        time_delta=60,
        faster_input_generation=False,
    ):
        self.clients2models = clients2models
        self.min_nclients_same_pred = min_nclients_same_pred
        self.same_seqs_set = set()
        self.k_gen_inputs = k_gen_inputs
        self.size = 1024
        self.random_inputs = []
        self.input_shape = input_shape
        self.time_delta = time_delta
        self.random_generator_func = random_generator_func
        self.transform = transform_func
        self.faster_input_generation = faster_input_generation

    def _get_random_input(self):
        """
        Generate a single random input tensor.

        Returns
        -------
        torch.Tensor
            A tensor representing a random input, optionally transformed and with an added batch dimension.
        """
        img = torch.empty(self.input_shape)
        self.random_generator_func(img)
        if self.transform is not None:
            return self.transform(img).unsqueeze(0)
        return img.unsqueeze(0)

    def _simple_random_inputs(self):
        """
        Generate a list of random inputs using a simple random input generation method.

        Returns
        -------
        tuple
            A tuple containing:
                - A list of generated random input tensors.
                - The time taken (in seconds) to generate these inputs.
        """
        start = time.time()
        random_inputs = [self._get_random_input() for _ in range(self.k_gen_inputs)]
        return random_inputs, time.time() - start

    def get_inputs(self):
        """
        Generate and return random inputs based on the configuration and client feedback.

        Returns
        -------
        tuple
            A tuple containing:
                - A list of generated input tensors.
                - The time taken (in seconds) to generate each input on average.
        """
        if self.faster_input_generation:
            logging.info("Faster input generation generation")
            return self._simple_random_inputs()
        else:
            if len(self.clients2models) < 10:
                return self._simple_random_inputs()
            else:
                return self._generate_feedback_random_inputs1()

    def _predict_func(self, model, input_tensor):
        """
        Predict the class for the given input using the provided model.

        Parameters
        ----------
        model : torch.nn.Module
            The model used to perform prediction.
        input_tensor : torch.Tensor
            The input tensor to classify.

        Returns
        -------
        int
            The predicted class label as an integer.
        """
        model.eval()
        logits = model(input_tensor)
        preds = torch.argmax(F.log_softmax(logits, dim=1), dim=1)
        pred = preds.item()
        return pred

    # # feedback loop to create diverse set of inputs
    def _generate_feedback_random_inputs1(self):
        """
        Generate random inputs using a feedback loop that accepts inputs when a sufficient number of client models
        agree on the prediction.

        Returns
        -------
        tuple
            A tuple containing:
                - A list of accepted input tensors.
                - The total time taken (in seconds) to generate the inputs.
        """
        
        print("Feedback Random inputs")

        def _append_or_not_func(input_tensor):
            preds = [
                self._predict_func(m, input_tensor)
                for m in self.clients2models.values()
            ]
            for ci1, pred1 in enumerate(preds):
                seq = set()
                seq.add(ci1)
                for ci2, pred2 in enumerate(preds):
                    if ci1 != ci2 and pred1 == pred2:
                        seq.add(ci2)

                s = ",".join(str(p) for p in seq)
                if s not in same_prediciton and len(seq) >= self.min_nclients_same_pred:
                    same_prediciton.add(s)
                    random_inputs.append(input_tensor)
                    return

        timeout = 60
        random_inputs = []
        same_prediciton = set()
        start = time.time()
        while len(random_inputs) < self.k_gen_inputs:
            img = self._get_random_input()
            if self.min_nclients_same_pred > 1:
                _append_or_not_func(img)
            else:
                random_inputs.append(img)

            if time.time() - start > timeout:
                timeout += 60
                self.min_nclients_same_pred -= 1
                print(
                    f">> Timeout: Number of distinct inputs: {len(random_inputs)}, \
                        /so decreasing the min_nclients_same_pred to \
                        /{self.min_nclients_same_pred} and trying again with \
                        /timeout of {timeout} seconds"
                )
        return random_inputs, time.time() - start


class FaultyClientLocalization:
    """
    Perform faulty client localization using neuron activation analysis.

    Parameters
    ----------
    client2model : dict
        Dictionary mapping client identifiers to their corresponding models.
    generated_inputs : list
        List of input tensors used for neuron activation extraction.
    use_gpu : bool
        Flag indicating whether to perform computations on GPU.
    """
    def __init__(self, client2model, generated_inputs, use_gpu) -> None:
        self.generated_inputs = generated_inputs
        self.use_gpu = use_gpu
        self.clients2randominputs_neurons_activations: Dict[str, Any] = {}
        self.client2layeracts: Dict[str, Any] = {}
        self._update_neuron_coverage_func(client2model)
        self.clientids = set(client2model.keys())

        self.all_clients_combinations = _make_all_subsets_of_size_n(
            self.clientids, len(self.clientids) - 1
        )  # resetting for next random iteration

    def _update_neuron_coverage_func(self, client2model):
        """
        Update neuron activations for each client model over the generated inputs.

        Parameters
        ----------
        client2model : dict
            Dictionary mapping client identifiers to their models.
        """
        device = torch.device("cpu")
        if self.use_gpu:
            device = torch.device("cuda")

        for client_id, model in client2model.items():
            model = model.to(device)
            outs = [
                get_neurons_activations(model, img.to(device))
                for img in self.generated_inputs
            ]
            self.clients2randominputs_neurons_activations[client_id] = [
                all_acts for all_acts, _ in outs
            ]
            self.client2layeracts[client_id] = [layer_acts for _, layer_acts in outs]

            model = model.to(torch.device("cpu"))
            gc.collect()
            torch.cuda.empty_cache()

    def run_fault_localization(self, na_t, num_bugs):
        """
        Run the fault localization algorithm to identify faulty client(s) based on neuron activation thresholds.

        Parameters
        ----------
        na_t : float
            The neuron activation threshold.
        num_bugs : int
            The number of faulty clients (bugs) expected.

        Returns
        -------
        list
            A list of sets, each set containing the identifiers of potential faulty clients for each input.
        """
        faulty_clients_on_gen_inputs = []
        for i in range(len(self.generated_inputs)):
            potential_faulty_clients = None
            # for the given input i find "num_bugs" number of faulty clients
            for _ in range(num_bugs):
                benign_clients_ids = self._find_normal_clients_seq_v1(i, na_t)
                potential_faulty_clients = self.clientids - benign_clients_ids
                self._update_clients_combinations(potential_faulty_clients)

            faulty_clients_on_gen_inputs.append(potential_faulty_clients)

            self.all_clients_combinations = _make_all_subsets_of_size_n(
                self.clientids, len(self.clientids) - 1
            )  # resetting for next generated input

        assert len(faulty_clients_on_gen_inputs) == len(self.generated_inputs)
        return faulty_clients_on_gen_inputs

    def _update_clients_combinations(self, potential_faulty_clients):
        """
        Update the client combinations based on the currently identified potential faulty clients.

        Parameters
        ----------
        potential_faulty_clients : set
            A set of client identifiers that are potentially faulty.
        """
        remaining_clients = self.clientids - potential_faulty_clients
        self.all_clients_combinations = _make_all_subsets_of_size_n(
            remaining_clients, len(remaining_clients) - 1
        )

    def _find_normal_clients_seq_v1(self, input_id, na_t):
        """
        Identify normal clients for a given input based on their neuron activations.

        Parameters
        ----------
        input_id : int
            Index of the generated input to analyze.
        na_t : float
            The neuron activation threshold.

        Returns
        -------
        set
            A set of client identifiers considered as normal (non-faulty).
        """
        client2_na = {
            cid: c2na[input_id] > na_t
            for cid, c2na in self.clients2randominputs_neurons_activations.items()
        }
        clients_ids = self._get_clients_ids_with_highest_common_neurons(client2_na)
        return clients_ids

    def _get_clients_ids_with_highest_common_neurons(self, clients2neurons2boolact):
        """
        Determine the client identifiers with the highest number of common neuron activations.

        Parameters
        ----------
        clients2neurons2boolact : dict
            Dictionary mapping client identifiers to tensors of boolean neuron activations.

        Returns
        -------
        set
            A set of client identifiers that have the highest common neuron activations.
        """
        select_neurons = self._torch_intersection(clients2neurons2boolact) == False

        clients_neurons2boolact = {
            cid: t[select_neurons] for cid, t in clients2neurons2boolact.items()
        }

        count_of_common_neurons = [
            (
                self._torch_intersection(
                    {cid: clients_neurons2boolact[cid] for cid in comb}
                )
                == True
            )
            .sum()
            .item()
            for comb in self.all_clients_combinations
        ]
        highest_number_of_common_neurons = max(count_of_common_neurons)
        val_index = count_of_common_neurons.index(highest_number_of_common_neurons)
        val_clients_ids = self.all_clients_combinations[val_index]
        return val_clients_ids

    def _torch_intersection(self, client2tensors):
        """
        Compute the element-wise logical AND (intersection) across tensors from multiple clients.

        Parameters
        ----------
        client2tensors : dict
            Dictionary mapping client identifiers to boolean tensors.

        Returns
        -------
        torch.Tensor
            A tensor representing the element-wise intersection of the input tensors.
        """
        
        intersct = True
        for _k, v in client2tensors.items():
            intersct = intersct * v
            torch.cuda.synchronize()
        return intersct


def _get_transforms_for_diff_testing(dname):
    """
    Get data transforms for differential testing based on the dataset name.

    Parameters
    ----------
    dname : str
        Name of the dataset (e.g., 'cifar10' or 'mnist').

    Returns
    -------
    dict
        A dictionary with keys 'train' and 'test' mapping to the corresponding transformation functions.
    """
    
    transform_dict = {"train": None, "test": None}
    if dname == "cifar10":
        transform_dict["train"] = Compose(
            [
                Resize((32, 32)),
                RandomHorizontalFlip(),
                Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        transform_dict["test"] = Compose(  # type: ignore
            [
                Resize((32, 32)),
                Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
    elif dname == "mnist":
        transform_dict["train"] = Compose(
            [Resize((32, 32)), Normalize((0.1307,), (0.3081,))]
        )

        transform_dict["test"] = Compose(
            [Resize((32, 32)), Normalize((0.1307,), (0.3081,))]
        )

    return transform_dict


class FedDebug:
    """
    Main class to run the debugging analysis for federated learning.

    Parameters
    ----------
    cfg : Any
        Configuration object containing necessary parameters and paths.
    round_key : str
        Key identifying the current federated learning round.
    """
    def __init__(self, cfg, round_key) -> None:
        self.cfg = cfg
        self.round_key = round_key
        self.random_generator = kaiming_uniform_
        self._extract_round_id()
        self._load_training_config()
        self._initialize_and_load_models()
        self.na_threshold = cfg.neuron_activation_threshold
    
    def _extract_round_id(self) -> None:
        """
        Extract and set the round identifier from the given round key.
        """
        self.round_id = self.round_key.split(":")[-1]

    def _load_training_config(self) -> None:
        """
        Load the training configuration from the training cache based on the experiment key in the configuration.
        """
        self.training_cache = Index(
            self.cfg.storage.dir + self.cfg.storage.train_cache_name
        )
        exp_dict = self.training_cache[self.cfg.exp_key]
        self.train_cfg = exp_dict["train_cfg"]  # type: ignore
        self.num_bugs = len(self.train_cfg.faulty_clients_ids)  # type: ignore
        self.input_shape = tuple(exp_dict["input_shape"])  # type: ignore
        self.transform_func = _get_transforms_for_diff_testing(
            dname=self.train_cfg.dataset.name
        )["test"]

    def _initialize_and_load_models(self) -> None:
        """
        Initialize and load the client models using the parameters from the training configuration and cache.
        """
        round2ws = self.training_cache[self.round_key]
        self.client2num_examples = round2ws["client2num_examples"]  # type: ignore

        self.client2model = {}
        for cid, ws in round2ws["client2ws"].items():  # type: ignore
            cmodel = initialize_model(
                self.train_cfg.model.name, self.train_cfg.dataset
            )["model"]
            cmodel.load_state_dict(ws)  # type: ignore
            cmodel = cmodel.cpu().eval()  # type: ignore
            self.client2model[cid] = cmodel

    def _get_fault_localization_accuracy(self, predicted_faulty_clients_on_each_input):
        """
        Compute the fault localization accuracy based on predicted faulty clients and ground truth.

        Parameters
        ----------
        predicted_faulty_clients_on_each_input : list
            A list of sets where each set contains the predicted faulty client identifiers for a generated input.

        Returns
        -------
        float
            The average fault localization accuracy as a percentage.
        """        
        true_faulty_clients = set([f"{c}" for c in  self.train_cfg.faulty_clients_ids])
        logging.info(f"True Malicious client(s) {true_faulty_clients}")
        detection_acc = 0
        for pred_faulty_clients in predicted_faulty_clients_on_each_input:
            logging.info(f"-- Potential Malicious client(s) {pred_faulty_clients}")

            correct_localize_faults = len(
                true_faulty_clients.intersection(pred_faulty_clients)
            )
            acc = (correct_localize_faults / len(true_faulty_clients)) * 100
            detection_acc += acc
        fault_localization_acc = detection_acc / len(
            predicted_faulty_clients_on_each_input
        )
        return fault_localization_acc

    def _help_run(self, k_gen_inputs, na_threshold, use_gpu):
        """
        Helper method to run the faulty client localization and input generation processes.

        Parameters
        ----------
        k_gen_inputs : int
            Number of inputs to generate.
        na_threshold : float
            Neuron activation threshold used for localization.
        use_gpu : bool
            Flag to indicate whether to use GPU for computations.

        Returns
        -------
        tuple
            A tuple containing:
                - List of potential faulty client sets for each input.
                - Average input generation time per input.
                - Average fault localization time per input.
        """        
        
        print(">  Running FaultyClientLocalization ..")
        generate_inputs = InferenceGuidedInputs(
            self.client2model,
            self.input_shape,
            random_generator_func=self.random_generator,
            transform_func=self.transform_func,
            min_nclients_same_pred=3,
            k_gen_inputs=k_gen_inputs,
            faster_input_generation=self.cfg.faster_input_generation,
        )

        selected_inputs, input_gen_time = generate_inputs.get_inputs()
        

        start = time.time()
        faultyclientlocalization = FaultyClientLocalization(
            self.client2model, selected_inputs, use_gpu=use_gpu
        )

        potential_faulty_clients_for_each_input = (
            faultyclientlocalization.run_fault_localization(
                na_threshold, num_bugs=self.num_bugs
            )
        )
        fault_localization_time = (time.time() - start) / len(selected_inputs)
        return (
            potential_faulty_clients_for_each_input,
            input_gen_time/len(selected_inputs), # average
            fault_localization_time,
        )


    def run(self, k_gen_inputs=10, use_gpu=True) -> Dict[str, any]:  # type: ignore
        """
        Run the debugging analysis for the current federated learning round.

        Parameters
        ----------
        k_gen_inputs : int, optional
            Number of random inputs to generate for analysis (default is 10).
        use_gpu : bool, optional
            Flag indicating whether to use GPU for computations (default is True).

        Returns
        -------
        dict
            A dictionary containing:
                - 'clients': list of client identifiers.
                - 'eval_metrics': dictionary with evaluation metrics (e.g., accuracy).
                - 'avg_fault_localization_time': average time taken for fault localization per input.
                - 'avg_input_gen_time': average time taken for input generation per input.
                - 'round_id': identifier for the current round.
        """
        
        predicted_faulty_clients, input_gen_time, fault_localization_time = (
            self._help_run(k_gen_inputs, self.na_threshold, use_gpu)
        )

        fault_localization_acc = self._get_fault_localization_accuracy(
            predicted_faulty_clients
        )

        eval_metrics = {"accuracy": fault_localization_acc}

        logging.info(f"Fault Localization Accuracy: {fault_localization_acc}")

        debug_result = {
            "clients": list(self.client2model.keys()),
            "eval_metrics": eval_metrics,
            "avg_fault_localization_time": fault_localization_time,
            "avg_input_gen_time": input_gen_time,
            "round_id": self.round_id,
        }

        return debug_result


def _get_round_keys_and_central_test_data(fl_key, train_cache_path):
    """
    Retrieve round keys from the training cache excluding the central test data key.

    Parameters
    ----------
    fl_key : str
        The federated learning experiment key to exclude.
    train_cache_path : str
        The path to the training cache.

    Returns
    -------
    list
        A list of round keys that correspond to federated learning rounds (excluding `fl_key`).
    """    
    
    training_cache = Index(train_cache_path)
    r_keys = []
    for k in training_cache.keys():
        if fl_key == k:
            continue
        elif k.find(fl_key) != -1 and len(k) > len(fl_key):
            r_keys.append(k)
    return r_keys


def round_lambda_fed_debug_func(cfg, round_key):
    """
    Lambda function to perform debugging for a specific round.

    Parameters
    ----------
    cfg : Any
        Configuration object containing necessary parameters.
    round_key : str
        Key identifying the current federated learning round.

    Returns
    -------
    dict
        A dictionary containing the debugging results for the specified round.
    """    
    round_debug = FedDebug(cfg, round_key)
    debug_result_dict = round_debug.run()
    return debug_result_dict


def run_fed_debug_differential_testing(cfg, store_in_cache=True):
    """
    Run the debugging analysis for differential testing in federated learning.

    Parameters
    ----------
    cfg : Any
        Configuration object containing parameters, paths, and experiment keys.
    store_in_cache : bool, optional
        If True, store the results in the debug cache (default is True).

    Returns
    -------
    dict or None
        A dictionary containing:
            - 'round2debug_result': list of debugging results for each round.
            - 'debug_cfg': the configuration used for debugging.
            - 'training_cache_path': path to the training cache.
            - 'avg_debug_time_per_round': average debugging time per round.
        Returns None if no rounds are found.
    """
    train_cache_path = cfg.storage.dir + cfg.storage.train_cache_name
    debug_results_cache = Index(cfg.storage.dir + cfg.storage.fed_debug_cache_name)

    round2debug_result = []

    if len(round2debug_result) > 0:
        logging.info(">> Debugging is already done.")
        return round2debug_result

    rounds_keys = _get_round_keys_and_central_test_data(cfg.exp_key, train_cache_path)
    logging.debug(f"rounds_keys {rounds_keys}")

    start_time = time.time()

    round2debug_result = [
        round_lambda_fed_debug_func(cfg, round_key) for round_key in rounds_keys
    ]

    end_time = time.time()

    if len(rounds_keys) == 0:
        logging.warning("Unable to get any model. Something is wrong.")
        return None

    avg_debug_time_per_round = (end_time - start_time) / len(rounds_keys)

    final_result_dict = {
        "round2debug_result": round2debug_result,
        "debug_cfg": cfg,
        "training_cache_path": train_cache_path,
        "avg_debug_time_per_round": avg_debug_time_per_round,
    }


    cfg.exp_key = f"fed_debug_{cfg.exp_key}"

    if store_in_cache:
        debug_results_cache[cfg.exp_key] = final_result_dict

        logging.info(
            f"Debugging results saved for {cfg.exp_key}, \
            / avg Debugging time per round: {avg_debug_time_per_round} seconds"
        )
    return final_result_dict


def eval_na_threshold(cfg):
    """
    Evaluate the impact of different neuron activation thresholds on debugging accuracy.

    Parameters
    ----------
    cfg : Any
        Configuration object containing:
            - 'storage': information about cache directories and file names.
            - 'threshold_variation_exp_key': key for threshold variation experiments.
            - 'threshold_exps_keys': list of experiment keys for threshold evaluation.
            - 'neuron_act_thresholds': list of neuron activation threshold values to evaluate.
            - Other parameters required for running the debugging analysis.

    Returns
    -------
    None
        The function prints the cache of threshold evaluation results.
    """
    
    debug_results_cache = Index(cfg.storage.dir + cfg.storage.results_cache_name)
    na_cached_results_dict = debug_results_cache.get(
        cfg.threshold_variation_exp_key, {}
    )

    for exp_key in cfg.threshold_exps_keys:
        cfg.exp_key = exp_key
        na2acc = {}
        for na in cfg.neuron_act_thresholds:
            temp_cfg = copy.deepcopy(cfg)
            temp_cfg.neuron_activation_threshold = na
            r2results = run_fed_debug_differential_testing(
                temp_cfg, store_in_cache=False
            )["round2debug_result"]

            all_accs = [r["eval_metrics"]["accuracy"] for r in r2results]
            avg_accs = sum(all_accs) / len(all_accs)

            logging.info(
                f"Neuron Activation threshold {na} "
                f"and average malicious client localization accuracy is {avg_accs}."
            )
            na2acc[na] = avg_accs

        na_cached_results_dict[cfg.exp_key] = {
            "cfg": debug_results_cache[exp_key]["debug_cfg"],
            "na2acc": na2acc,
        }

    debug_results_cache[cfg.threshold_variation_exp_key] = na_cached_results_dict

    print(debug_results_cache)

