import logging
import torch
import torch.nn.functional as F
import transformers
import transformers.models.bert.modeling_bert as modeling_bert
from tracefl.utils import compute_importance
from tracefl.models import test_neural_network


class NeuronProvenance:
    def __init__(self, cfg, arch, test_data, gmodel, c2model, num_classes, c2nk):
        """
        Initialize a NeuronProvenance instance.

        Parameters
        ----------
        cfg : object
            Configuration object.
        arch : str
            Model architecture type.
        test_data : object
            Test dataset.
        gmodel : torch.nn.Module
            Global model.
        c2model : dict
            Dictionary mapping client identifiers to their models.
        num_classes : int
            Number of classes.
        c2nk : dict
            Dictionary mapping client identifiers to number of examples.
        """
        self.arch = arch
        self.cfg = cfg
        self.test_data = test_data
        self.gmodel = gmodel
        self.c2model = c2model
        self.num_classes = num_classes
        self.device = self.cfg.device
        self.c2nk = c2nk  # client2num_examples
        self.client_ids = list(self.c2model.keys())
        self.layer_importance = compute_importance(len(getAllLayers(gmodel)))
        
        if cfg.enable_beta == False:
            self.layer_importance = [1 for _ in range(len(getAllLayers(gmodel)))]        
            
        logging.info(f'client ids: {self.client_ids}')
        self.pk = {
            cid: self.c2nk[cid] / sum(self.c2nk.values()) for cid in self.c2nk.keys()}
        if self.cfg.client_weights_normalization:
            logging.debug('>> clients weights are normaized')
            self._inplaceScaleClientWs()

    def _checkAnomlies(self, t):
        """
        Check the tensor for inf or NaN values.

        Parameters
        ----------
        t : torch.Tensor
            The tensor to be checked.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If inf or NaN values are detected in the tensor.
        """
        inf_mask = torch.isinf(t)
        nan_mask = torch.isnan(t)
        if inf_mask.any() or nan_mask.any():
            logging.error(f"Inf values: {torch.sum(inf_mask)}")
            logging.error(f"NaN values: {torch.sum(nan_mask)}")
            logging.error(f"Total values: {torch.numel(t)}")
            raise ValueError("Anomalies detected in tensor")


    def _calculateLayerContribution(self, global_neurons_outputs: torch.Tensor, global_layer_grads: torch.Tensor, client2outputs, layer_id: int):
        """
        Calculate the contribution of each client for a specific layer.

        Parameters
        ----------
        global_neurons_outputs : torch.Tensor
            Global neurons outputs for a given input.
        global_layer_grads : torch.Tensor
            Gradients of the global layer.
        client2outputs : dict
            Dictionary mapping client IDs to their outputs.
        layer_id : int
            Identifier of the layer.

        Returns
        -------
        dict
            Dictionary mapping client IDs to their contribution for the layer.
        """
        client2avg = {cid: 0.0 for cid in self.client_ids}
        self._checkAnomlies(global_neurons_outputs)
        self._checkAnomlies(global_layer_grads)
        global_layer_grads = global_layer_grads.flatten()
        
        for cli in self.client_ids:
            cli_acts = client2outputs[cli].to(self.device).flatten()
            self._checkAnomlies(cli_acts)
            cli_part =  torch.dot(cli_acts, global_layer_grads)           
            
            client2avg[cli] = cli_part.item() * self.layer_importance[layer_id]
            cli_acts = cli_acts.cpu()

        # cli with max contribution
        max_contributor = max(client2avg, key=client2avg.get) # type: ignore
        logging.debug(f"Max contributor: {max_contributor}")
        return client2avg

    def _mapClientLayerContributions(self, layer_id: int):
        """
        Map the contributions of clients for a specific layer.

        Parameters
        ----------
        layer_id : int
            Identifier of the layer.

        Returns
        -------
        dict
            Dictionary mapping input indices to client contributions.
        """
        client2layers = {cid: getAllLayers(cm)
                         for cid, cm in self.c2model.items()}
        global_neurons_inputs = self.global_neurons_inputs_outputs_batch[layer_id][0]
        global_neurons_inputs = global_neurons_inputs.to(self.device)
        global_neurons_outputs = self.global_neurons_inputs_outputs_batch[layer_id][1]

        # check if global_neurons_outputs is a tuple
        if isinstance(global_neurons_outputs, tuple) or isinstance(global_neurons_outputs, list):
            assert len(
                global_neurons_outputs) == 1, f"Expected 1 element in tuple, got {len(global_neurons_outputs)}"
            global_neurons_outputs = global_neurons_outputs[0]
        global_neurons_outputs = global_neurons_outputs.to(self.device)

        c2l = {cid: client2layers[cid][layer_id] for cid in self.client_ids}
        clinet2outputs = {c: self._evaluateLayer(
            l, global_neurons_inputs) for c, l in c2l.items()}

        input2client2contribution = {}
        for input_id in range(len(self.test_data)):  # for per input in the batch
            logging.debug(
                f"Mapping client contributions for input {input_id} for layer {layer_id}")
            c2out_per_input = {
                cid: clinet2outputs[cid][input_id] for cid in self.client_ids}

            glayer_grads = torch.squeeze(
                self.inputs2layer_grads[input_id][layer_id][1]).to(self.device)

            c2contribution = self._calculateLayerContribution(
                global_neurons_outputs=global_neurons_outputs[input_id], global_layer_grads=glayer_grads, client2outputs=c2out_per_input, layer_id=layer_id)
            input2client2contribution[input_id] = c2contribution

        return input2client2contribution

    def _inplaceScaleClientWs(self):
        """
        Scale client model weights based on the number of data points per client.

        Returns
        -------
        None
        """
        logging.debug(
            "Scaling client weights based on the number of data points each client has.")
        
        logging.debug(f'Total clients in c2nk: {len(self.c2nk)}')
        logging.debug(f'Total clients in c2model: {len(self.c2model)}')

        ids1 = list(self.c2model.keys())
        ids2 = list(self.c2nk.keys())
        logging.debug(f'ids1: {ids1}')
        logging.debug(f'ids2: {ids2}')

        for cid in self.c2model.keys():
            scale_factor = self.c2nk[cid] / sum(self.c2nk.values())
            logging.debug(
                f"Scaling client {cid} by {scale_factor}, nk = {self.c2nk[cid]}")
            with torch.no_grad():
                for cparam in self.c2model[cid].parameters():
                    cparam.data = scale_factor * cparam.data
                self.c2model[cid] = self.c2model[cid].eval().cpu()


    def _captureLayerIO(self):
        """
        Capture the inputs and outputs of all layers in the global model.

        Returns
        -------
        None
        """
        hook_manager = HookManager()
        glayers = getAllLayers(self.gmodel)
        # logging.debug(f"all layers in global model: {glayers}")
        logging.debug(f"Total layers in global model: {len(glayers)}")
        hooks_forward = [hook_manager.insertForwardHook(
            layer) for layer in glayers]
        self.gmodel.eval()
        self.gmodel = self.gmodel.to(self.device)
        test_neural_network(self.arch, {'model': self.gmodel}, self.test_data,
                        batch_size=len(self.test_data))
        hook_manager.removeHooks(hooks_forward)
        self.global_neurons_inputs_outputs_batch = hook_manager.forward_hooks_storage
        hook_manager.clearStorages()

    def _captureLayerGradients(self):
        """
        Capture the gradients for each layer of the global model for the test data.

        Returns
        -------
        None
        """
        self.inputs2layer_grads = []
        for m_input in torch.utils.data.DataLoader(self.test_data, batch_size=1): # type: ignore
            hook_manager = HookManager()
            setGradientsofModel(self.arch, self.gmodel,
                                m_input, self.device, hook_manager)
            self.inputs2layer_grads.append(hook_manager.backward_hooks_storage)
            hook_manager.clearStorages()

    def _evaluateLayer(self, client_layer: torch.nn.Module, global_neurons_inputs: torch.Tensor) -> torch.Tensor:
        """
        Evaluate a client's layer using the global model's layer inputs.

        Parameters
        ----------
        client_layer : torch.nn.Module
            The client's layer.
        global_neurons_inputs : torch.Tensor
            The inputs captured from the global model for the layer.

        Returns
        -------
        torch.Tensor
            The output of the client's layer.
        """
        client_layer = client_layer.eval().to(device=self.device)
        client_neurons_outputs = client_layer(global_neurons_inputs)

        if isinstance(client_neurons_outputs, tuple) or isinstance(client_neurons_outputs, list):
            client_neurons_outputs = client_neurons_outputs[0].cpu()
        else:
            client_neurons_outputs = client_neurons_outputs.cpu()

        client_layer = client_layer.cpu()
        return client_neurons_outputs

    def _aggregateClientContributions(self, input_id: int, layers2prov):
        """
        Aggregate the contributions of all clients for a given input across layers.

        Parameters
        ----------
        input_id : int
            The index of the input.
        layers2prov : list
            List of client contributions per layer.

        Returns
        -------
        dict
            Dictionary mapping client IDs to aggregated contribution for the input.
        """
        client2totalcont = {c: 0.0 for c in self.client_ids}
        clients_prov = [i2prov[input_id] for i2prov in layers2prov]
        for c2lsum in clients_prov:
            for cid in self.client_ids:
                client2totalcont[cid] += c2lsum[cid]
        return client2totalcont

    def _normalizeContributions(self, contributions):
        """
        Normalize client contributions using softmax.

        Parameters
        ----------
        contributions : dict
            Dictionary mapping client IDs to contributions.

        Returns
        -------
        dict
            Dictionary mapping client IDs to normalized contributions.
        """
        conts = F.softmax(torch.tensor(list(contributions.values())), dim=0)
        client2prov = {cid: v.item() for cid, v in zip(self.client_ids, conts)}
        return dict(sorted(client2prov.items(), key=lambda item: item[1], reverse=True))

    def _aggregateInputContributions(self, layers2prov):
        """
        Aggregate client contributions for each input across all layers.

        Parameters
        ----------
        layers2prov : list
            List of client contributions per layer.

        Returns
        -------
        list
            List of dictionaries mapping each input to its provenance data.
        """
        input2prov = []
        for input_id in range(len(self.test_data)):
            client_conts = self._aggregateClientContributions(
                input_id, layers2prov)
            normalized_conts = self._normalizeContributions(client_conts)
            traced_client = max(
                normalized_conts, key=normalized_conts.get)  # type: ignore
            input2prov.append({
                "traced_client": traced_client,
                "client2prov": normalized_conts
            })
        return input2prov

    def computeInputProvenance(self):
        """
        Compute the provenance of each input by aggregating client contributions across layers.

        Returns
        -------
        list
            List of dictionaries mapping each input to its provenance data.
        """
        self._captureLayerIO()
        self._captureLayerGradients()

        layers2prov = []
        for layer_id in range(len(self.global_neurons_inputs_outputs_batch)):
            client2cont = self._mapClientLayerContributions(layer_id)
            layers2prov.append(client2cont)

        input2prov = self._aggregateInputContributions(layers2prov)
        return input2prov


class HookManager:
    def __init__(self):
        """
        Initialize a HookManager instance.
        """
        self.forward_hooks_storage = []
        self.backward_hooks_storage = []

    def insertForwardHook(self, layer):
        """
        Insert a forward hook into the specified layer.

        Parameters
        ----------
        layer : torch.nn.Module
            The layer to attach the hook to.

        Returns
        -------
        handle
            The hook handle.
        """
        def forward_hook(module, input_tensor, output_tensor):
            try:
                input_tensor = input_tensor[0]
                input_tensor = input_tensor.detach()
            except Exception as e:
                pass

            input_tensor = input_tensor.detach()
            output_tensor = output_tensor
            self.forward_hooks_storage.append((input_tensor, output_tensor))

        hook = layer.register_forward_hook(forward_hook)
        return hook

    def insertBackwardHook(self, layer):
        """
        Insert a backward hook into the specified layer.

        Parameters
        ----------
        layer : torch.nn.Module
            The layer to attach the hook to.

        Returns
        -------
        handle
            The hook handle.
        """
        def backward_hook(module, input_tensor, output_tensor):
            try:
                input_tensor = input_tensor[0]
                output_tensor = output_tensor[0]
                input_tensor = input_tensor.detach()
                output_tensor = output_tensor.detach()

            except Exception as e:
                # logging.debug(f"Error processing input in backward hook: {e}")
                pass
            try:
                input_tensor = input_tensor.detach()
            except Exception as e:
                pass
            try:
                output_tensor = output_tensor.detach()
            except Exception as e:
                pass

            self.backward_hooks_storage.append((input_tensor, output_tensor))

        hook = layer.register_full_backward_hook(backward_hook)
        return hook

    def clearStorages(self):
        """
        Clear the storage for forward and backward hooks.

        Returns
        -------
        None
        """
        self.forward_hooks_storage = []
        self.backward_hooks_storage = []

    def removeHooks(self, hooks):
        """
        Remove the provided hooks.

        Parameters
        ----------
        hooks : list
            List of hook handles to be removed.

        Returns
        -------
        None
        """
        for hook in hooks:
            hook.remove()

#   ==================================================== Helpers ==================================================================


def setGradientsofModel(arch, net, text_input_tuple, device, hook_manager):
    """
    Set up hooks to capture gradients of the model during backward pass.

    Parameters
    ----------
    arch : str
        Model architecture ("transformer" or "cnn").
    net : torch.nn.Module
        The model.
    text_input_tuple : dict or tensor
        Input data for the model.
    device : str or torch.device
        Device on which to perform computation.
    hook_manager : HookManager
        Instance of HookManager to manage hooks.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the model architecture is not supported.
    """
    if arch == "transformer":
        _setGradientsTransformerModel(
            net, text_input_tuple, device, hook_manager)
    elif arch == "cnn":
        _setGradientsCNNModel(net, text_input_tuple, device, hook_manager)
    else:
        raise ValueError(f"Model architecture {arch} not supported")

def _setGradientsCNNModel(net, input_for_model, device, hook_manager):
    """
    Set up hooks and compute gradients for a CNN model.

    Parameters
    ----------
    net : torch.nn.Module
        The CNN model.
    input_for_model : dict
        Dictionary containing input data under key 'pixel_values'.
    device : str or torch.device
        Device on which to perform computation.
    hook_manager : HookManager
        Instance of HookManager to manage hooks.

    Returns
    -------
    None
    """
    net.zero_grad()
    all_layers = getAllLayers(net)
    all_hooks = [hook_manager.insertBackwardHook(
        layer) for layer in all_layers]

    net = net.to(device)
    img_input = input_for_model['pixel_values']
    outs = net(img_input.to(device))
    logits = outs  # Access the logits from the output object

    prob, predicted = torch.max(logits, dim=1)
    logits[0, predicted].backward()  # computing the gradients
    hook_manager.removeHooks(all_hooks)
    hook_manager.backward_hooks_storage.reverse()


def _setGradientsTransformerModel(net, text_input_tuple, device, hook_manager):
    """
    Set up hooks and compute gradients for a transformer model.

    Parameters
    ----------
    net : torch.nn.Module
        The transformer model.
    text_input_tuple : dict
        Dictionary containing input data for the transformer.
    device : str or torch.device
        Device on which to perform computation.
    hook_manager : HookManager
        Instance of HookManager to manage hooks.

    Returns
    -------
    None
    """
    # Insert hooks for capturing backward gradients of the transformer model
    net.zero_grad()
    all_layers = getAllLayers(net)
    all_hooks = [hook_manager.insertBackwardHook(
        layer) for layer in all_layers]

    net.to(device)

    # Assume text_input_tuple is already on the correct device and prepared
    text_input_tuple = {k: torch.tensor(v, device=device).unsqueeze(
        0) for k, v in text_input_tuple.items() if k in ["input_ids", "token_type_ids", "attention_mask"]}

    outs = net(**text_input_tuple)

    logits = outs.logits  # Access the logits from the output object

    prob, predicted = torch.max(logits, dim=1)
    predicted = predicted.cpu().detach().item()
    logits[0, predicted].backward()  # computing the gradients
    hook_manager.removeHooks(all_hooks)
    hook_manager.backward_hooks_storage.reverse()


def getAllLayers(net):
    """
    Retrieve all layers from the model using a BERT-specific extraction.

    Parameters
    ----------
    net : torch.nn.Module
        The model.

    Returns
    -------
    list
        List of layers extracted from the model.
    """
    layers = getAllLayersBert(net)
    return layers #[len(layers)-1:len(layers)]


def getAllLayersBert(net):
    """
    Retrieve all layers from the model that are instances of specific layer types.

    Parameters
    ----------
    net : torch.nn.Module
        The model.

    Returns
    -------
    list
        List of layers that are instances of Linear, Conv2d, LayerNorm, Conv1D, or BertLayer.
    """
    layers = []
    for layer in net.children():
        if isinstance(layer, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.LayerNorm, transformers.pytorch_utils.Conv1D, modeling_bert.BertLayer)):
            layers.append(layer)
        elif len(list(layer.children())) > 0:
            temp_layers = getAllLayersBert(layer)
            layers = layers + temp_layers
    return layers



