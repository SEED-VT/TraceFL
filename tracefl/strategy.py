"""Optionally define a custom strategy.

Needed only when the strategy is not yet implemented in Flower or because you want to
extend or modify the functionality of an existing strategy.
"""
import gc
import logging
import flwr as fl
from tracefl.client import set_parameters
from tracefl.models import initialize_model


class FedAvgSave(fl.server.strategy.FedAvg):
    """A custom federated averaging strategy that saves client and global model state.

    This strategy extends the Flower FedAvg strategy to store individual client model updates
    and the aggregated global model weights in a cache.
    """
    def __init__(self, cfg, cache, *args, **kwargs):
        """
        Initialize the FedAvgSave strategy.

        Parameters
        ----------
        cfg : object
            Configuration object containing experiment settings.
        cache : diskcache.Index
            Cache object for storing model states.
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.cache = cache

    def aggregate_fit(self, server_round, results, failures):
        """
        Aggregate client model updates and save the aggregated state.

        Parameters
        ----------
        server_round : int
            The current server round.
        results : list
            A list of tuples, each containing the client identifier and the fit result.
        failures : list
            A list of client failures.

        Returns
        -------
        tuple
            A tuple containing:
            - aggregated_parameters : fl.common.Parameters or None
                The aggregated model parameters.
            - aggregated_metrics : dict
                Aggregated metrics computed during the fit aggregation.
        """
        round_dict = {}
        round_dict["client2ws"] = {
            fit_res.metrics["cid"]: self.get_state_dict_from_parameters(
                fit_res.parameters
            )
            for _, fit_res in results
        }

        client_ids = round_dict["client2ws"].keys()

        logging.info(f"participating clients: {client_ids}")

        # client2num_examples save in round_dict from results
        round_dict["client2num_examples"] = {
            fit_res.metrics["cid"]: fit_res.num_examples for _, fit_res in results
        }
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        round_key = f"{self.cfg.exp_key}-round:{server_round}"

        if aggregated_parameters is not None:
            round_dict["gm_ws"] = self.get_state_dict_from_parameters(
                aggregated_parameters
            )
            self.cache[round_key] = round_dict
            round_dict.clear()
        gc.collect()
        return aggregated_parameters, aggregated_metrics

    def get_state_dict_from_parameters(self, parameters):
        """
        Convert Flower parameters to a state_dict for a PyTorch model.

        Parameters
        ----------
        parameters : fl.common.Parameters
            Model parameters in Flower's format.

        Returns
        -------
        dict
            A state_dict representing the model parameters.
        """
        ndarr = fl.common.parameters_to_ndarrays(parameters)
        temp_net = initialize_model(self.cfg.model.name, self.cfg.dataset)["model"]
        set_parameters(temp_net, ndarr)
        return temp_net.state_dict()
