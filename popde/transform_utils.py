# Functions for transforming input data
import numpy as np


def apply_transf(column, fnc):
    """
    Apply a named transformation to a single array

    'log' or 'ln' : take log of data
    'stdize' :
    fnc convertible to float : multiply by constant
    """
    column = np.asarray(column)
    if fnc in ['log', 'ln']:
        return np.log(column)
    elif fnc == 'exp':
        return np.exp(column)
    elif fnc == 'stdize':
        return column / np.std(column)
    else:
        try:  # Convert possible string to float
            return column * float(fnc)
        except:
            raise ValueError(f'I got an unknown transformation function {fnc} !')


def transform_data(data, transf):
    """
    Cycle over data dimensions applying a named transformation to each
    """
    assert len(data.shape) == 2  # only works for 'tabular' data
    assert len(transf) == data.shape[1]
    transf_data = np.zeros_like(data)
    for dim, (col, fnc) in enumerate(zip(data.T, transf)):  # iterate over columns
        if fnc in ('None', 'none'):  # no-op
             transf_data[:, dim] = col
        else:
             transf_data[:, dim] = apply_transf(col, fnc)

    return transf_data


def reverse_transform(data, transf, stds, rescale):
    """
    Reverse the input process: rescale, de-standardize, transform
    """
    assert len(data.shape) == 2  # only works for 'tabular' data
    invdict = {'log': 'exp',
               'ln': 'exp',
               'exp': 'log'
               }
    if rescale is not None:
        data = transform_data(data, 1. / np.array(rescale))
    if stds is not None:  # restore original variances
        data = transform_data(data, np.array(stds))
    inv_transf = [invdict[t] if t in invdict else t
                  for t in transf]
    return transform_data(data, inv_transf)

