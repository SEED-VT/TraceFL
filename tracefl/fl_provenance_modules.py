from typing import Dict, List
import torch
import logging
from diskcache import Index
import time
from tracefl.models import test_neural_network, initialize_model
from tracefl.neuron_provenance import NeuronProvenance, getAllLayers
from tracefl.utils import get_prov_eval_metrics

class FederatedProvTrue:
    def __init__(self, prov_cfg, round_key: str, server_test_data, t) -> None:
        """
        Initialize a FederatedProvTrue instance.

        Parameters
        ----------
        prov_cfg : object
            Configuration for provenance, including storage and experiment parameters.
        round_key : str
            Identifier for the current training round.
        server_test_data : object
            Test data from the central server.
        t : any
            Additional parameter required for initialization.
        """    
        
        self.t = t
        self.prov_cfg = prov_cfg
        self.round_key = round_key
        self._extractRoundID()
        self._loadTrainingConfig()
        self._initializeAndLoadModels()
        self._setParticipatingClientsLabels()
        self._selectProvenanceData(server_test_data)

    def _modelInitializeWrapper(self):
        """
        Initialize and return a new model instance.

        Returns
        -------
        torch.nn.Module
            A new model instance initialized with the specified architecture.
        """
        m = initialize_model(self.train_cfg.model.name,
                             self.train_cfg.dataset)  # type: ignore
        return m['model']

    def _extractRoundID(self) -> None:
        """
        Extract the round ID from the round key and assign it to an instance attribute.

        Returns
        -------
        None
        """
        self.round_id = self.round_key.split(":")[-1]

    def _loadTrainingConfig(self) -> None:
        """
        Load the training configuration from the cache specified in the provenance configuration.

        The method initializes the training cache, retrieves the experiment dictionary using the
        experiment key, and sets the training configuration and client-to-class mapping.

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If the experiment key is not found in the training cache.
        """
        self.training_cache = Index(
            self.prov_cfg.storage.dir + self.prov_cfg.storage.train_cache_name)
        exp_dict = self.training_cache[self.prov_cfg.exp_key]
        self.train_cfg = exp_dict["train_cfg"]  # type: ignore
        self.ALLROUNDSCLIENTS2CLASS = exp_dict["client2class"]  # type: ignore

    def _initializeAndLoadModels(self) -> None:
        """
        Initialize and load the global and client-specific models for the current training round.

        The method retrieves model weights from the training cache using the round key,
        initializes the global model and each client model, sets them to CPU and evaluation mode,
        and records the layers used for provenance.

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If required keys are missing in the training cache data.
        """
        logging.info(
            f'\n\n             ----------Round key  {self.round_key} -------------- \n')
        round2ws = self.training_cache[self.round_key]
        self.client2num_examples = round2ws["client2num_examples"]
        self.prov_global_model = self._modelInitializeWrapper()
        self.prov_global_model.load_state_dict(
            round2ws["gm_ws"])  # type: ignore
        self.prov_global_model = self.prov_global_model.cpu().eval()  # type: ignore

        self.save_prov_layers = set(
            [type(l) for l in getAllLayers(self.prov_global_model)])
        logging.debug(f"Layers used in Provenance: {self.save_prov_layers}")

        self.client2model = {}
        for cid, ws in round2ws["client2ws"].items():  # type: ignore
            cmodel = self._modelInitializeWrapper()
            cmodel.load_state_dict(ws)  # type: ignore
            cmodel = cmodel.cpu().eval()  # type: ignore
            self.client2model[cid] = cmodel

    def _setParticipatingClientsLabels(self) -> None:
        """
        Aggregate and set the union of labels from all participating client models.

        The method iterates over the client models and collects unique labels from the
        client-to-class mapping.

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

    def _evalAndExtractCorrectPredsTransformer(self, test_data):
        """
        Evaluate the global model on test data and extract indices and labels of correctly predicted samples.

        Parameters
        ----------
        test_data : object
            The test dataset to be evaluated.

        Returns
        -------
        tuple
            A tuple containing:
            - torch.Tensor: Indices of correctly predicted samples.
            - torch.Tensor: Actual labels corresponding to the evaluated samples.
        """
        d = test_neural_network(self.train_cfg.model.arch, {
                                'model': self.prov_global_model}, test_data)  # type: ignore
        self.loss = d["loss"]
        self.acc = d["accuracy"]
        logging.debug(f"Accuracy on test data: {self.acc}")
        return d["eval_correct_indices"], d["eval_actual_labels"]

    def _balanceDatasetByLabel(self, correct_indices: torch.Tensor, dataset_labels: torch.Tensor, min_per_label: int) -> torch.Tensor:
        """
        Balance the dataset by selecting a minimum number of samples per label from correctly predicted indices.

        Parameters
        ----------
        correct_indices : torch.Tensor
            Indices corresponding to correctly predicted samples.
        dataset_labels : torch.Tensor
            Labels for the dataset corresponding to the indices.
        min_per_label : int
            Minimum number of samples required for each label.

        Returns
        -------
        torch.Tensor
            A tensor of indices representing a balanced subset of the dataset.

        Raises
        ------
        ValueError
            If the number of correct predictions for a label is less than min_per_label.
        """         
        balanced_indices = []
        logging.debug(
            f'participating_clients_labels {self.participating_clients_labels}')
        for l in self.participating_clients_labels:
            selected_labels = dataset_labels[correct_indices]
            temp_bools = selected_labels == l
            temp_correct_indxs = correct_indices[temp_bools]
            if len(temp_correct_indxs) >= min_per_label:
                balanced_indices.append(temp_correct_indxs[:min_per_label])
        if len(balanced_indices) > 0:
            balanced_indices = torch.cat(balanced_indices)

        return balanced_indices

    def _selectProvenanceData(self, central_test_data, min_per_label: int = 2) -> None:
        """
        Select a subset of test data for provenance analysis based on balanced, correctly predicted samples.

        Parameters
        ----------
        central_test_data : object
            The complete test dataset from the central server.
        min_per_label : int, optional
            Minimum number of samples per label required for the subset (default is 2).

        Returns
        -------
        None
        """
        all_correct_i, dataset_lablels = self._evalAndExtractCorrectPredsTransformer(
            central_test_data)

        # correct_ds = Dataset.from_dict(central_test_data[all_correct_i])

        balanced_indices = self._balanceDatasetByLabel(
            all_correct_i, dataset_lablels, min_per_label)  # type: ignore

        self.subset_test_data = central_test_data.select(balanced_indices)

        if len(self.subset_test_data) == 0:
            logging.info("No correct predictions found")

    def _sanityCheck(self):
        """
        Perform a sanity check on the selected test data by verifying prediction accuracy.

        Returns
        -------
        float
            The accuracy on the subset of test data.

        Raises
        ------
        ValueError
            If the subset test data is empty.
        AssertionError
            If the evaluated accuracy does not equal 1.
        """
        if len(self.subset_test_data) == 0:
            raise ValueError("No correct predictions found")

        acc = test_neural_network(self.train_cfg.model.arch, {
                                  'model': self.prov_global_model}, self.subset_test_data)["accuracy"]
        logging.info(f"Sanity check: {acc}")
        assert int(acc) == 1, "Sanity check failed"
        
        return acc

    def _computeEvalMetrics(self, input2prov: List[Dict]) -> Dict[str, float]:
        """
        Compute evaluation metrics for provenance data by comparing traced client predictions with actual labels.

        Parameters
        ----------
        input2prov : list of dict
            List containing dictionaries with provenance information for each input.

        Returns
        -------
        dict
            A dictionary containing evaluation metrics, including accuracy.
        """
        data_loader = torch.utils.data.DataLoader(  # type: ignore
            self.subset_test_data, batch_size=1)
        target_labels = [data['label'].item() for data in data_loader]

        client2class = {
            c: self.ALLROUNDSCLIENTS2CLASS[c] for c in self.client2model}

        logging.debug(f"client2class: {client2class}")

        correct_tracing = 0

        true_labels = []
        predicted_labels = []

        for idx, prov_r in enumerate(input2prov):
            traced_client = prov_r["traced_client"]
            client2prov = prov_r["client2prov"]

            target_l = target_labels[idx]
            responsible_clients = [
                cid for cid, c_labels in client2class.items() if target_l in c_labels]

            res_c_string = ','.join(map(str, [f"c{c}" for c in responsible_clients]))

            logging.info(
                f'            *********** Input Label: {target_l}, Responsible Client(s): {res_c_string}  *************')

            if target_l in client2class[traced_client]:
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
        return eval_metrics

    def run(self) -> Dict[str, any]:  # type: ignore
        """
        Execute the provenance analysis process and return the results.

        Returns
        -------
        dict
            A dictionary containing provenance analysis results, including:
              - List of participating clients.
              - Number of data points used.
              - Evaluation metrics.
              - Test data accuracy and loss.
              - Total provenance computation time.
              - Round identifier.
              - Layers used for provenance.
        """        
        r = self._sanityCheck()
        if r is None:
            prov_result = {
                "clients": list(self.client2model.keys()),
                "data_points": len(self.subset_test_data),
                "eval_metrics": {},
                "test_data_acc": self.acc,
                "test_data_loss": self.loss,
                "prov_time": -1,
                "round_id": self.round_id,
                'prov_layers': self.save_prov_layers,
            }

        sart_time = time.time()
        nprov = NeuronProvenance(cfg=self.prov_cfg, arch=self.train_cfg.model.arch, test_data=self.subset_test_data, gmodel=self.prov_global_model,  # type: ignore
                                 c2model=self.client2model, num_classes=self.train_cfg.dataset.num_classes, c2nk=self.client2num_examples)  # type: ignore
        input2prov = nprov.computeInputProvenance()
        eval_metrics = self._computeEvalMetrics(input2prov)
        end_time = time.time()

        logging.info(f"[Round {self.round_id}] TraceFL Localization Accuracy = {eval_metrics['Accuracy']*100} || Total Inputs Used In Prov: {len(self.subset_test_data)} || GM_(loss, acc) ({self.loss},{self.acc})")
        
        prov_result = {
            "clients": list(self.client2model.keys()),
            "data_points": len(self.subset_test_data),
            "eval_metrics": eval_metrics,
            "test_data_acc": self.acc,
            "test_data_loss": self.loss,
            "prov_time": end_time - sart_time,
            "round_id": self.round_id,
            'prov_layers': self.save_prov_layers,
        }

        return prov_result

