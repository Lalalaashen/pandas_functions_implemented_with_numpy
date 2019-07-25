# -*- coding: utf-8 -*-
# @Time     : 2019.07
# @Author   : Eason
# @FileName : pandas_functions.py
import numpy as np


# pandas->diff
def pct_change(xs, n, fill_value=np.nan) -> np.ndarray:
    '''
    >>> x = np.arange(1, 10, 1)
    >>> pct_change(x, 1)
    '''
    return xs / shift(xs, n, fill_value) - 1


# pandas->diff
def diff(xs, n, fill_value=np.nan) -> np.ndarray:
    '''
    >>> x = np.arange(1, 10, 1)
    >>> diff(x, 1)
    '''
    return xs - shift(xs, n, fill_value)


# pandas->shift
def shift(xs, n, fill_value=np.nan) -> np.ndarray:
    '''
    >>> x = np.arange(1, 10, 1)
    >>> shift(x, 1)
    >>> shift(x, -1)
    '''
    if n >= 0:
        return np.r_[np.full(n, fill_value), xs[:-n]]
    else:
        return np.r_[xs[-n:], np.full(-n, fill_value)]


def rolling(arr, window, np_func, fill_value=np.nan, forward=True) -> np.ndarray:
    '''
    >>> x = np.arange(1, 10, 1)
    >>> rolling(x, 2, np.min)
    >>> rolling(x, 2, np.max)
    >>> rolling(x, 2, np.max, forward=False)
    '''
    shape = arr.shape[:-1] + (arr.shape[-1] - window + 1, window)
    strides = arr.strides + (arr.strides[-1],)
    roll = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
    if forward:
        return np.append(np.zeros(window - 1) + fill_value, np_func(roll, axis=1))
    else:
        return np.append(np_func(roll, axis=1), np.zeros(window - 1) + fill_value)
