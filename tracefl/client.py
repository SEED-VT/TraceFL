"""Define your client class and a function to construct such clients.

Please overwrite `flwr.client.NumPyClient` or `flwr.client.Client` and create a function
to instantiate your client.
"""
import gc
import flwr as fl
import torch
from tracefl.models import train_neural_network

class FlowerClient(fl.client.NumPyClient):
    """
    Flower client for training a CNN model.

    Parameters
    ----------
    config : dict
        A configuration dictionary containing keys such as:
        - "client_data_train": The local training dataset.
        - "model_dict": A dictionary that includes the model instance under the key "model".
        - "arch": The model architecture identifier.
        - "device": The computation device (e.g., "cpu" or "cuda").
        - "cid": The client identifier.
    """

    def __init__(self, config):
        """
        Initialize the FlowerClient instance with the provided configuration.

        Parameters
        ----------
        config : dict
            Configuration parameters as described in the class docstring.
        """
        self.config = config

    def fit(self, parameters, config):
        """
        Train the model on the local dataset and return updated parameters.

        Parameters
        ----------
        parameters : list of numpy.ndarray
            Model parameters received from the server.
        config : dict
            Configuration dictionary for local training containing keys such as:
            - "batch_size": The batch size for training.
            - "lr": Learning rate for training.
            - "local_epochs": The number of epochs for local training.

        Returns
        -------
        tuple
            A tuple containing:
            - parameters : list of numpy.ndarray
                The updated model parameters after local training.
            - nk_client_data_points : int
                The number of batches processed (computed as the integer division
                of the size of the training dataset by the batch size).
            - client_train_dict : dict
                A dictionary with training details, which includes at least the key "cid"
                and other training metrics from the training process.
        """
        
        nk_client_data_points = int(len(self.config["client_data_train"])/config["batch_size"])
        
        model_dict = self.config["model_dict"]

        set_parameters(model_dict["model"], parameters=parameters)
        train_dict = train_neural_network(
            {
                'arch': self.config["arch"],
                "lr": config["lr"],
                "epochs": config["local_epochs"],
                "batch_size": config["batch_size"],
                "model_dict": model_dict,
                "train_data": self.config["client_data_train"],
                "device": self.config["device"],
            }
        )

        model_dict["model"] = model_dict["model"].cpu()
        parameters = get_parameters(model_dict["model"])
        client_train_dict = {"cid": self.config["cid"]} | train_dict
        gc.collect()
        return parameters, nk_client_data_points, client_train_dict


def get_parameters(model):
    """
    Extract and return the parameters of a PyTorch model as a list of NumPy arrays.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model from which to extract the parameters.

    Returns
    -------
    list of numpy.ndarray
        A list containing the model parameters converted to NumPy arrays.
    """
    model = model.cpu()
    return [val.cpu().detach().clone().numpy() for _, val in model.state_dict().items()]


def set_parameters(net, parameters):
    """
    Set the parameters of a PyTorch model from a list of NumPy arrays.

    Parameters
    ----------
    net : torch.nn.Module
        The PyTorch model whose parameters will be updated.
    parameters : list of numpy.ndarray
        A list of parameters (as NumPy arrays) to load into the model.

    Returns
    -------
    None
    """
    net = net.cpu()
    params_dict = zip(net.state_dict().keys(), parameters)
    new_state_dict = {k: torch.from_numpy(v) for k, v in params_dict}
    net.load_state_dict(new_state_dict, strict=True)
