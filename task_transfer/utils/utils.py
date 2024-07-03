import hashlib
import itertools as it
from collections import Iterable, Mapping, OrderedDict

import torch


# from nnfabrik:
# https://github.com/sinzlab/nnfabrik/blob/ea4f5148c943741e45d937fe7ee681978b4224f7/nnfabrik/utility/dj_helpers.py#L58-L97
def make_hash(obj):
    """
    Given a Python object, returns a 32 character hash string to uniquely identify
    the content of the object. The object can be arbitrary nested (i.e. dictionary
    of dictionary of list etc), and hashing is applied recursively to uniquely
    identify the content.
    For dictionaries (at any level), the key order is ignored when hashing
    so that {"a":5, "b": 3, "c": 4} and {"b": 3, "a": 5, "c": 4} will both
    give rise to the same hash. Exception to this rule is when an OrderedDict
    is passed, in which case difference in key order is respected. To keep
    compatible with previous versions of Python and the assumed general
    intentions, key order will be ignored even in Python 3.7+ where the
    default dictionary is officially an ordered dictionary.
    Args:
        obj - A (potentially nested) Python object
    Returns:
        hash: str - a 32 charcter long hash string to uniquely identify the object.
    """
    hashed = hashlib.md5()

    if isinstance(obj, str):
        hashed.update(obj.encode())
    elif isinstance(obj, OrderedDict):
        for k, v in obj.items():
            hashed.update(str(k).encode())
            hashed.update(make_hash(v).encode())
    elif isinstance(obj, Mapping):
        for k in sorted(obj, key=str):
            hashed.update(str(k).encode())
            hashed.update(make_hash(obj[k]).encode())
    elif isinstance(obj, Iterable):
        for v in obj:
            hashed.update(make_hash(v).encode())
    else:
        hashed.update(str(obj).encode())

    return hashed.hexdigest()


def dict_product(d, insert_hash=True):
    """
    Generates the cartesian product of a dictionary of lists and returns a list of dictionaries.

    Each dictionary in the resulting list represents a unique combination of the input lists' values.
    Optionally, a unique hash identifier can be added to each dictionary.

    Args:
        d (dict): A dictionary where each key maps to a list of values.
                Example: {'color': ['red', 'blue'], 'size': ['small', 'medium', 'large']}
        insert_hash (bool): A flag to determine whether to include a unique hash identifier
                            (keyed by 'id') in each output dictionary. Default is True.

    Returns:
        list: A list of dictionaries, each representing a unique combination of the input lists' values.
            If insert_hash is True, each dictionary will include an 'id' key with a unique hash value.

    Example:
        input_dict = {
            'color': ['red', 'blue'],
            'size': ['small', 'medium', 'large']
        }

        dict_product(input_dict)
        # Output:
        # [
        #     {'color': 'red', 'size': 'small', 'id': 'some_hash'},
        #     {'color': 'red', 'size': 'medium', 'id': 'some_hash'},
        #     {'color': 'red', 'size': 'large', 'id': 'some_hash'},
        #     {'color': 'blue', 'size': 'small', 'id': 'some_hash'},
        #     {'color': 'blue', 'size': 'medium', 'id': 'some_hash'},
        #     {'color': 'blue', 'size': 'large', 'id': 'some_hash'}
        # ]

    Note:
        The order of keys in the output dictionaries matches the order of keys in the input dictionary.
    """
    result = []
    # Generate the cartesian product of the lists in the dictionary
    for values in it.product(*d.values()):
        # Create a dictionary by zipping the keys and the generated values
        product_dict = {key: value for key, value in zip(d.keys(), values)}
        # Add a unique identifier hash to the dictionary
        if insert_hash:
            product_dict["id"] = make_hash(product_dict)
        # Append the dictionary to the result list
        result.append(product_dict)

    return result


def are_models_equal(model1, model2):
    """
    Check if two PyTorch models have the exact same parameter values.

    This function compares the parameters of two nn.Module models to determine
    if they are exactly the same. It does this by iterating over the parameters
    of both models and checking for equality.

    Parameters:
    model1 (nn.Module): The first model to compare.
    model2 (nn.Module): The second model to compare.

    Returns:
    bool: True if the models have the exact same parameters, False otherwise.

    Example:
    >>> model1 = nn.Linear(10, 1)
    >>> model2 = nn.Linear(10, 1)
    >>> model2.load_state_dict(model1.state_dict())
    >>> are_models_equal(model1, model2)
    True
    """
    # Check if both models have the same number of parameters
    # If they don't, they cannot be the same
    if len(list(model1.parameters())) != len(list(model2.parameters())):
        return False

    # Iterate through the parameters of both models
    for param1, param2 in zip(model1.parameters(), model2.parameters()):
        # Check if the parameters are equal
        # torch.equal checks if two tensors have the same size and elements
        if not torch.equal(param1, param2):
            return False

    # If all parameters are equal, the models are considered the same
    return True
