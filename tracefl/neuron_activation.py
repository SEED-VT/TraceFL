"""The file contains the code to get the activations of all neurons."""
import torch
import torch.nn.functional as F


def _get_all_layers_in_neural_network(net):
    """
    Retrieve all layers of the neural network that are either Conv2d or Linear.

    Parameters
    ----------
    net : torch.nn.Module
        The neural network model.

    Returns
    -------
    list
        A list of layers (torch.nn.Module) that are either Conv2d or Linear.
    """
    layers = []
    for layer in net.children():
        if len(list(layer.children())) == 0 and (
            isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear)
        ):
            layers.append(layer)
        if len(list(layer.children())) > 0:
            temp_layers = _get_all_layers_in_neural_network(layer)
            layers = layers + temp_layers
    return layers


global Hooks_Storage
Hooks_Storage = []


def _get_input_and_output_of_layer(self, input_t, output_t):
    """
    Capture and store the output of a layer during the forward pass.

    Parameters
    ----------
    self : torch.nn.Module
        The layer for which the hook is registered.
    input_t : tuple
        The input tensor(s) provided to the layer.
    output_t : torch.Tensor
        The output tensor produced by the layer.

    Returns
    -------
    None
    """
    global Hooks_Storage
    assert (
        len(input_t) == 1
    ), f"Hook, {self.__class__.__name__} Expected 1 input, got {len(input_t)}"
    Hooks_Storage.append(output_t.detach())


def _insert_hooks_func(layers):
    """
    Insert forward hooks into the specified layers.

    Parameters
    ----------
    layers : list
        A list of layers (torch.nn.Module) into which hooks will be inserted.

    Returns
    -------
    list
        A list of hook handles.
    """
    all_hooks = []
    for layer in layers:
        h = layer.register_forward_hook(_get_input_and_output_of_layer)
        all_hooks.append(h)
    return all_hooks


def _scale_func(out, rmax=1, rmin=0):
    """
    Scale the output tensor to a specified range.

    Parameters
    ----------
    out : torch.Tensor
        The tensor to be scaled.
    rmax : float, optional
        The maximum value of the scaled range (default is 1).
    rmin : float, optional
        The minimum value of the scaled range (default is 0).

    Returns
    -------
    torch.Tensor
        The scaled tensor.
    """
    output_std = (out - out.min()) / (out.max() - out.min())
    output_scaled = output_std * (rmax - rmin) + rmin
    return output_scaled


def _my_eval_neurons_activations(model, x):
    """
    Evaluate the neuron activations of the model for a given input.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model.
    x : torch.Tensor
        The input tensor.

    Returns
    -------
    tuple
        A tuple containing:
            - A 1D tensor of concatenated flattened activations from all layers.
            - A list of tensors with the activations of each layer.
    """
    global Hooks_Storage
    layer2output = []
    all_layers = _get_all_layers_in_neural_network(model)

    layers = all_layers  # [1:]

    hooks = _insert_hooks_func(layers)
    model(x)  # forward pass and everthing is stored in Hooks_Storage
    for l_id in range(len(layers)):
        outputs = F.relu(Hooks_Storage[l_id])
        scaled_outputs = _scale_func(outputs)
        layer2output.append(scaled_outputs)

    _ = [h.remove() for h in hooks]  # remove the hooks
    Hooks_Storage = []
    return torch.cat([out.flatten() for out in layer2output]), layer2output


def get_neurons_activations(model, img):
    """
    Get the activations of all neurons in the model for the given input image.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model.
    img : torch.Tensor
        The input image tensor.

    Returns
    -------
    tuple
        A tuple containing:
            - A 1D tensor of concatenated flattened activations from all layers.
            - A list of tensors with the activations of each layer.
    """
    r = _my_eval_neurons_activations(model, img)
    return r
