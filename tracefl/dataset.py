"""Handle basic dataset creation.

In case of PyTorch it should return dataloaders for your dataset (for both the clients
and the server). If you are using a custom dataset class, this module is the place to
define it. If your dataset requires to be downloaded (and this is not done
automatically -- e.g. as it is the case for many dataset in TorchVision) and
partitioned, please include all those functions and logic in the
`dataset_preparation.py` module. You can use all those functions from functions/methods
defined here of course.
"""

import logging
from tracefl.dataset_preparation import ClientsAndServerDatasets
from diskcache import Index

def mdedical_dataset2labels(dname):
    """
    Retrieve the mapping of class indices to label names for a given dataset.

    Parameters
    ----------
    dname : str
        The name of the dataset for which the label mapping is requested.

    Returns
    -------
    dict or None
        A dictionary mapping integer class indices to string labels if `dname` is recognized.
        Returns None if the dataset name is not recognized.
    """
    if dname == 'pathmnist':
        return {0: 'Adipose', 1: 'Background', 2: 'Debris', 3: 'Lymphocytes', 4: 'Mucus', 5: 'Smooth Muscle', 6: 'Normal Colon Mucosa', 7: 'Cancer-associated Stroma', 8: 'Colorectal Adenocarcinoma'}
    else:
        return None

def get_clients_server_data(cfg):
    """
    Obtain and cache datasets for clients and server based on the provided configuration.

    This function checks for an existing dataset in the cache using a unique key derived
    from the configuration. If the dataset is present in the cache and caching is enabled,
    it loads the dataset from the cache. Otherwise, it creates the dataset using
    `ClientsAndServerDatasets`, caches it, and then returns it.

    Parameters
    ----------
    cfg : object
        A configuration object that must contain attributes for storage paths and data
        distribution parameters, as well as a flag to check the dataset cache.

    Returns
    -------
    dict
        A dictionary containing the datasets for both clients and server.
    """
    
    ds_dict = {}
    cache_path = cfg.storage.dir + cfg.storage.fl_datasets_cache
    cache = Index(cache_path)

    dataset_key = f"-"
    for k, v in cfg.data_dist.items():
        dataset_key += f"{k}:{v}-"

    if dataset_key in cache.keys()  and cfg.check_dataset_cache:
        logging.info(
            f"\nLoading dataset from cache {cache_path}: {dataset_key}\n")
        ds_dict   = cache[dataset_key]       
    else:
        ds_prep = ClientsAndServerDatasets(cfg)
        ds_dict = ds_prep.get_data()
        cache[dataset_key] = ds_dict
        logging.info(f"Saving dataset to cache {cache_path}: {dataset_key}")
    
    return ds_dict


def load_central_server_test_data(cfg):
    """
    Load the test data intended for the central server.

    This function retrieves the complete dataset using `ClientsAndServerDatasets`
    and then extracts the subset corresponding to the server's test data.

    Parameters
    ----------
    cfg : object
        A configuration object required to initialize the dataset creation process.

    Returns
    -------
    object
        The test dataset for the central server.
    """
    d_obj = ClientsAndServerDatasets(cfg).get_data()
    return d_obj["server_testdata"]


def convert_client2_faulty_client(ds, label2flip, target_label_col= 'label'):
    """
    Transform a client's dataset to simulate a faulty client by flipping specified labels.

    This function iterates over the dataset, and for each example, if the label (accessed
    via the `target_label_col` key) exists in the `label2flip` mapping, the label is replaced
    by its corresponding flipped value. The function also counts the occurrences of each label
    after the transformation.

    Parameters
    ----------
    ds : Dataset
        The dataset to be modified. It is expected to support the `map` method and iteration,
        and its elements must be mutable mappings containing at least the key specified by
        `target_label_col`.
    label2flip : dict
        A dictionary mapping original label values to the flipped label values.
    target_label_col : str, optional
        The key in each example dict corresponding to the target label. Default is 'label'.

    Returns
    -------
    dict
        A dictionary with the following keys:
        
        - 'ds': The transformed dataset with flipped labels.
        - 'label2count': A dictionary mapping each label to the count of its occurrences
          in the transformed dataset.
    """
    def flip_label(example):
        label = None
        try:
            label = example[target_label_col].item()
        except:
            label = example[target_label_col]
        if label in label2flip:
            example[target_label_col] = label2flip[label]  
        return example
    
    ds =  ds.map(flip_label).with_format("torch")
    label2count = {}

    for example in ds:
        label = example[target_label_col].item()
        if label not in label2count:
            label2count[label] = 0
        label2count[label] += 1

    return {'ds': ds, 'label2count' : label2count}
