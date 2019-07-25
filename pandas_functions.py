# -*- coding: utf-8 -*-
# @Time     : 2019.07
# @Author   : Eason
# @FileName : pandas_functions.py
import numpy as np


# pandas->shift
def shift(xs, n, fill_value=np.nan) -> np.ndarray:
    if n >= 0:
        return np.r_[np.full(n, fill_value), xs[:-n]]
    else:
        return np.r_[xs[-n:], np.full(-n, fill_value)]


# pandas->rolling
def rolling(arr, window) -> np.ndarray:
    shape = arr.shape[:-1] + (arr.shape[-1] - window + 1, window)
    strides = arr.strides + (arr.strides[-1],)
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
