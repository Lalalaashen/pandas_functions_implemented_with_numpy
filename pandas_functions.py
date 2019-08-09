# -*- coding: utf-8 -*-
# @Time     : 2019.07
# @Author   : Eason
# @FileName : pandas_functions.py


# 实现类似pandas的shift
def shift_(xs, n, fill_value=None):
    import numpy as np

    fill_value = np.nan if fill_value is None else fill_value
    return np.r_[np.full(n, fill_value), xs[:-n]] if n >= 0 else np.r_[xs[-n:], np.full(-n, fill_value)]


# 实现类似于pandas的rolling
def rolling_(arr, window, func1d, fill_value=None, fill_before=True):
    import numpy as np

    shape = arr.shape[:-1] + (arr.shape[-1] - window + 1, window)
    strides = arr.strides + (arr.strides[-1],)
    roll = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)

    roll_value = np.apply_along_axis(func1d=func1d, axis=1, arr=roll)

    fill_value = np.nan if fill_value is None else fill_value
    if len(roll_value.shape) > 1:
        fill_value = np.zeros((window - 1, roll_value.shape[1])) + fill_value
        return np.vstack((fill_value, roll_value)) if fill_before else np.vstack((roll_value, fill_value))
    else:
        fill_value = np.zeros(window - 1) + fill_value
        return np.append(fill_value, roll_value) if fill_before else np.append(roll_value, fill_value)


# pandas->diff_
def pct_change_(xs, n, fill_value=None): return xs / shift_(xs, n, fill_value) - 1


# pandas->diff_
def diff_(xs, n, fill_value=None): return xs - shift_(xs, n, fill_value)
