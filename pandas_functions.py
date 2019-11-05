# -*- coding: utf-8 -*-
"""
This module is designed to implement faster pandas functions
"""

import numpy as np


def shift_(xs: np.ndarray, n: int, fill_value=None, out=None, dtype=None):
    """pandas->shift"""
    fill_value = 0 if fill_value is None else fill_value
    dtype = np.float64 if dtype is None else dtype
    if out is None:
        out = np.empty_like(xs, dtype=dtype)
    if n >= 0:
        out[:n] = fill_value
        out[n:] = xs[:-n]
    else:
        out[:n] = xs[-n:]
        out[n:] = fill_value
    return out


def pct_change_(xs: np.ndarray, n: int, fill_value=None, out=None):
    """pandas->pct_change, but NOT in percent"""
    assert n > 0, 'n should be positive'
    fill_value = 0 if fill_value is None else fill_value
    denominator = shift_(xs, n, fill_value)
    non_zero_flag = denominator != 0
    if out is not None:
        out[~non_zero_flag] = 1
        np.true_divide(xs, denominator, out=out, where=non_zero_flag)
        out -= 1
        return out
    else:
        return np.true_divide(
            xs, denominator,
            out=np.full_like(denominator, fill_value + 1, dtype=np.float64, order='F'),
            where=non_zero_flag
        ) - 1


def diff_(xs: np.ndarray, n: int, fill_value=None, out=None, dtype=None):
    """pandas->diff"""
    assert n > 0, 'n should be positive'
    dtype = np.float64 if dtype is None else dtype
    if out is not None:
        out[:] = xs - shift_(xs, n, fill_value, dtype=dtype)
        return out
    else:
        return xs - shift_(xs, n, fill_value, dtype=dtype)


def ob_gain_(ask_prc: np.ndarray, bid_prc: np.ndarray, n: int, out=None, dtype=None):
    """主买、主卖收益"""
    dtype = np.float64 if dtype is None else dtype
    if out is None:
        out = np.zeros_like(ask_prc, dtype=dtype)
    else:
        out[:] = 0.0
    if n >= 0:  # 过去的收益
        ask_prc_former = shift_(ask_prc, n, fill_value=ask_prc[:n], dtype=np.float64)
        bid_prc_former = shift_(bid_prc, n, fill_value=bid_prc[:n], dtype=np.float64)
        ask_prc_later = ask_prc
        bid_prc_later = bid_prc
    else:  # 未来的收益
        ask_prc_former = ask_prc
        bid_prc_former = bid_prc
        ask_prc_later = shift_(ask_prc, n, fill_value=ask_prc[n:], dtype=np.float64)
        bid_prc_later = shift_(bid_prc, n, fill_value=bid_prc[n:], dtype=np.float64)

    up_flag = (bid_prc_later > ask_prc_former) & (ask_prc_former > 0)
    down_flag = (bid_prc_former > ask_prc_later) & (ask_prc_later > 0)

    out[up_flag] = bid_prc_later[up_flag] / ask_prc_former[up_flag] - 1
    out[down_flag] = ask_prc_later[down_flag] / bid_prc_former[down_flag] - 1

    return out


def _rolling_window(xs: np.ndarray, window: int, stride=1):
    """Auxiliary functions"""
    assert window > 1, 'window should be larger than 1'
    shape = xs.shape[:-1] + (xs.shape[-1] - (window - 1) * stride, window)
    strides = xs.strides + (xs.strides[-1] * stride,)
    return np.lib.stride_tricks.as_strided(xs, shape=shape, strides=strides)


def _fill_out(xs: np.ndarray, window: int, out=None, fill_value=None, fill_before=True, dtype=None):
    """Auxiliary functions"""
    assert window > 1, 'window should be larger than 1'
    fill_value = 0 if fill_value is None else fill_value
    dtype = np.float64 if dtype is None else dtype
    if out is None:
        out = np.empty_like(xs, dtype=dtype, order='F')
    if fill_before:
        offset = window - 1
        out[:offset] = fill_value
    else:
        offset = 0
        out[out.shape[0] - (window - 1):] = fill_value
    return offset, out


def rolling_min(xs: np.ndarray, window: int, out=None, fill_value=None, fill_before=True, dtype=None):
    """rolling min"""
    offset, out = _fill_out(xs, window, out, fill_value, fill_before, dtype)
    roll = _rolling_window(xs, window)
    out[offset:xs.shape[0] - (window - 1) + offset] = roll.min(axis=1)
    return out


def rolling_max(xs: np.ndarray, window: int, out=None, fill_value=None, fill_before=True, dtype=None):
    """rolling max"""
    offset, out = _fill_out(xs, window, out, fill_value, fill_before, dtype)
    roll = _rolling_window(xs, window)
    out[offset:xs.shape[0] - (window - 1) + offset] = roll.max(axis=1)
    return out


def rolling_std(xs: np.ndarray, window: int, out=None, fill_value=None, fill_before=True, dtype=None):
    """rolling std"""
    offset, out = _fill_out(xs, window, out, fill_value, fill_before, dtype)
    roll = _rolling_window(xs, window)
    out[offset:xs.shape[0] - (window - 1) + offset] = roll.std(axis=1)
    return out


def rolling_pct(xs: np.ndarray, window: int, out=None, fill_value=None, fill_before=True, dtype=None):
    """rolling percent position, but NOT in percent"""
    offset, out = _fill_out(xs, window, out, fill_value, fill_before, dtype)
    roll = _rolling_window(xs, window)
    left = np.count_nonzero(roll < roll[:, -1].reshape((-1, 1)), axis=1)
    right = np.count_nonzero(roll <= roll[:, -1].reshape((-1, 1)), axis=1)
    out[offset:xs.shape[0] - (window - 1) + offset] = (right + left + (right > left)) * 0.5 / roll.shape[1]
    return out


def rolling_sum(xs: np.ndarray, window: int, out=None, fill_value=None, fill_before=True, dtype=None) -> np.ndarray:
    """rolling sum, xs and out can be the same array"""
    integrate_x = np.cumsum(xs)
    offset, out = _fill_out(xs, window, out, fill_value, fill_before, dtype)
    out[offset] = integrate_x[window - 1]
    out[offset + 1:xs.shape[0] - (window - 1) + offset] = integrate_x[window:] - integrate_x[:-window]
    return out


def rolling_mean(xs, window, out=None, fill_value=None, fill_before=True, dtype=None) -> np.ndarray:
    """rolling mean, xs and out can be the same array"""
    integrate_x = np.cumsum(xs)
    offset, out = _fill_out(xs, window, out, fill_value, fill_before, dtype)
    out[offset] = integrate_x[window - 1] / window
    out[offset + 1:xs.shape[0] - (window - 1) + offset] = (integrate_x[window:] - integrate_x[:-window]) / window
    return out


def rolling_std_reinforced(xs, window, out=None, fill_value=None, fill_before=True, dtype=None, integrate_x=None):
    """rolling std, xs and out can be the same array"""
    # Float Window
    window_float = np.array(window, dtype=np.float64)
    # Integrate xs
    if integrate_x is None:
        integrate_x = np.cumsum(xs, dtype=np.float64)

    offset, out_std = _fill_out(xs, window, out, fill_value, fill_before, dtype)
    offset, out_mean = _fill_out(xs, window, None, fill_value, fill_before, dtype)
    start_idx = offset
    end_idx = xs.shape[0] - (window - 1) + offset
    # mean
    out_mean[start_idx] = integrate_x[window - 1] / window_float
    out_mean[start_idx + 1:end_idx] = (integrate_x[window:] - integrate_x[:-window]) / window_float
    # shortcut
    avg_1_n = out_mean[start_idx:end_idx - 1]
    avg_2_np1 = out_mean[start_idx + 1:end_idx]
    x_1 = xs[:-window]
    x_np1 = xs[window:]
    # var
    out_std[start_idx] = np.sum(xs[:window] ** 2) / window_float - out_mean[offset] ** 2

    numerator_1 = (x_np1 + x_1 - avg_2_np1 - avg_1_n) * (x_np1 - x_1 + avg_1_n - avg_2_np1)
    temp = 2 * (integrate_x[window - 1:-1] - integrate_x[:-window]) - (avg_1_n + avg_2_np1) * (window_float - 1)
    numerator_2 = (avg_1_n - avg_2_np1) * temp
    out_std[start_idx + 1:end_idx] = (numerator_1 + numerator_2) / window_float

    np.cumsum(out_std[start_idx:end_idx], out=out_std[start_idx:end_idx])
    out_std[out_std < 0] = 0
    np.sqrt(out_std, out=out_std)

    return out_std


def diff_between(x1: np.ndarray, x2: np.ndarray, dm: int, val_1=0, val_2=0, out=None):
    """
    get difference between x1 and x2
    :param x1:
    :param x2:
    :param dm: 1-4
    :param val_1: fill value for dm = 2-3
    :param val_2: fill value for dm = 1-4
    :param out:
    :return:
    """
    if dm == 1:
        if out is None:
            out = np.empty_like(x1, dtype=np.float64, order='F')
        out[:] = x1 - x2
        return out
    elif dm == 2:
        if out is None:
            out = np.full_like(x2, val_1, dtype=np.float64, order='F')
        zero_flag = x2 == 0
        out[zero_flag] = val_1
        out[zero_flag & (x1 == 0)] = 1
        return np.true_divide(x1, x2, out=out, where=~zero_flag)
    elif dm == 3:
        if out is None:
            out = np.full_like(x1, val_1, dtype=np.float64, order='F')
        zero_flag = x1 == 0
        out[zero_flag] = val_1
        out[zero_flag & (x2 == 0)] = 1
        return np.true_divide(x2, x1, out=out, where=~zero_flag)
    elif dm == 4:
        num = x1 - x2
        den = x1 + x2
        if out is None:
            out = np.full_like(den, val_2, dtype=np.float64, order='F')
        non_zero_flag = den != 0
        out[~non_zero_flag] = val_2
        return np.true_divide(num, den, out=out, where=den != 0)
    else:
        raise NotImplementedError
