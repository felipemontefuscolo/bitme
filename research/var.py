import argparse

import datetime

import os

import math
import pandas as pd
from matplotlib.pyplot import *
from scipy.ndimage.interpolation import shift

# this script is just to understand how AR model works
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.vector_ar.var_model import VAR

from segmented_ols import seg_lin_reg

np.random.seed(446)


def diff(x, n, remove_nan=True):
    r = x - shift(x, n, cval=np.NaN)
    if remove_nan:
        r = r[~np.isnan(r)]
    return r


def hat_func(n):
    assert n % 2 == 1
    x = np.arange(n).astype('float64')
    y = np.arange(n).astype('float64')
    for i in range(n / 2, n):
        y[i] = n - 1 - i
    return x, y


def n_hat(n, num_hats):
    assert num_hats > 0
    y = hat_func(n)[1]
    for i in range(num_hats - 1):
        a = 0.5 if i % 2 == 0 else 1.
        y = np.append(y[:-1], a * hat_func(n)[1])
    return np.arange(len(y)), y


def sin_(n, n_osc=3.):
    x = np.arange(n).astype('float64')
    T = float((n - 1)) / float(n_osc)
    print('period = ' + str(T))
    y = np.cos(2. * np.pi / T * x * (1. + 0.00 * x))
    return x, y


# for VAR model
def test_forecast(x, Y, len_for_prediction, n_pred, *args, **kwargs):
    limit = 3 * np.max(Y)
    N = len(x)
    assert len(x) == len(Y)
    assert len(Y) > 0

    assert n_pred < N
    assert len_for_prediction < N - n_pred

    pred = list()
    for i in range(len_for_prediction, N - n_pred + 1):
        print("{} out of {}".format(i, N - n_pred + 1))
        input = Y[i - len_for_prediction: i]
        model = VAR(input)
        model_fit = model.fit(*args, **kwargs)
        if i == len_for_prediction:
            pred = list(model_fit.forecast(input, n_pred))
        else:
            pred += [list(model_fit.forecast(input, n_pred)[-1])]
    pred = np.array(pred)
    pred[np.abs(pred) > limit] = float('nan')
    return x[len_for_prediction:], pred


# for AR model
def test_predict(x, Y, len_for_prediction, n_pred, *args, **kwargs):
    limit = 3 * np.max(Y)
    N = len(x)
    assert len(x) == len(Y)
    assert len(Y) > 0

    assert n_pred < N
    assert len_for_prediction < N - n_pred

    pred = list()
    for i in range(len_for_prediction, N - n_pred + 1):
        print("{} out of {}".format(i, N - n_pred + 1))
        input = Y[i - len_for_prediction: i]
        model = AR(input)
        model_fit = model.fit(*args, **kwargs)
        # tmp = list(model_fit.predict(start=len(input), end=len(input)+n_pred-1, dynamic=True))
        if i == len_for_prediction:
            pred = list(model_fit.predict(start=len(input), end=len(input) + n_pred - 1, dynamic=True))
        else:
            pred += [list(model_fit.predict(start=len(input), end=len(input) + n_pred - 1, dynamic=True))[-1]]
        # pred = pred[:-len(tmp) + 1] + tmp
    pred = np.array(pred)
    pred[np.abs(pred) > limit] = float('nan')
    return x[len_for_prediction:], pred


def test_last_slope_forecast(x, Y, len_for_prediction, n_pred, decay=0.01, verbose=False):
    assert len(x) == len(Y)
    assert len(Y) > max(2, len_for_prediction)
    N = len(x)

    if verbose:
        print('testing last slope forecast')

    i0 = max(2, len_for_prediction)
    for i in range(i0, N - n_pred + 1):
        if verbose:
            print("{} out of {}".format(i, N - n_pred + 1))
        xx = x[i - len_for_prediction:i]
        yy = Y[i - len_for_prediction:i]
        seg = seg_lin_reg(xx, yy, tol=0.4, variance=100)[-1]

        fit = seg[2] * np.arange(i, i + n_pred, 1.0) + seg[1]
        avg = np.mean(Y[i - len_for_prediction:i])
        exp = np.exp(-decay * np.abs(seg[2]) * np.arange(0., n_pred, 1.0))
        final = fit * exp + avg * (1. - exp)

        if i == i0:
            pred = list(final)
        else:
            pred += [final[-1]]
    pred = np.array(pred)
    return x[len_for_prediction:], pred


def segs_to_np(segs):
    for i in range(len(segs)):
        segs[i] = [segs[i][0].stop - segs[i][0].start, segs[i][1], segs[i][2]]
    return np.array(segs)


def test_var_slope_forecast(x, Y, len_for_prediction, n_pred, decay=0.01, verbose=False):
    assert len(x) == len(Y)
    assert len(Y) > max(2, len_for_prediction)
    N = len(x)

    if verbose:
        print('testing last slope forecast')

    i0 = max(2, len_for_prediction)
    for i in range(i0, N - n_pred + 1):
        if verbose:
            print("{} out of {}".format(i, N - n_pred + 1))
        xx = x[i - len_for_prediction:i]
        yy = Y[i - len_for_prediction:i]
        segs = segs_to_np(seg_lin_reg(xx, yy, tol=0.4, variance=100))

        if len(segs) > 1:
            model = VAR(segs)
            model_fit = model.fit(trend='nc')
            seg = model_fit.forecast(segs, 1)[0]
        else:
            seg = segs[0]

        fit = seg[2] * np.arange(i, i + n_pred, 1.0) + seg[1]
        avg = np.mean(Y[i - len_for_prediction:i])
        exp = np.exp(-decay * np.abs(seg[2]) * np.arange(0., n_pred, 1.0))
        final = fit * exp + avg * (1. - exp)

        if i == i0:
            pred = list(final)
        else:
            pred += [final[-1]]
    pred = np.array(pred)
    return x[len_for_prediction:], pred


def artificial_data():
    N = 301
    # x, y = n_hat(N, 6)
    x, y = sin_(N, 5)
    y = y + 0.01 * np.random.normal(0., .5, len(y))
    z = y * y
    Y = np.matrix([y, z]).transpose().tolist()

    # ======================
    title('single prediction')
    model = VAR(Y)
    model_fit = model.fit(maxlags=15, ic='aic')
    pred = model_fit.forecast(Y[-model_fit.k_ar:], N)
    xx = np.arange(N, N + len(pred))
    assert (len(pred) == N)
    # print(model_fit.k_ar)
    # print(model_fit.params)
    plot(x, Y)
    plot(xx, pred, '--')
    show()
    # # =======================================

    title('dynamic prediction')
    xx, pred = test_forecast(x, Y, len_for_prediction=100, n_pred=100, maxlags=15, ic='aic')
    plot(x, Y)
    plot(xx, pred, '--')
    show()


def get_args():
    parser = argparse.ArgumentParser(description='VAR (Vector autoregression) forecast')
    parser.add_argument('-f', '--file', type=str, help='csv filename with candles data', required=True)
    parser.add_argument('-b', '--begin', type=str, help='begin time')
    parser.add_argument('-e', '--end', type=str, help='end time')

    args = parser.parse_args()

    if not os.path.isfile(args.file):
        raise ValueError("invalid file {}".format(args.file))

    if args.begin:
        args.begin = pd.Timestamp(args.begin)
    if args.end:
        args.end = pd.Timestamp(args.end)

    return args


# def timeparser(s):
#    # type: (str) -> pd.datetime
#    return pd.datetime.strptime(str(s), '%Y-%m-%dT%H:%M:%S')


def read_data(args):
    timeparser = lambda s: pd.datetime.strptime(str(s), '%Y-%m-%dT%H:%M:%S')
    Y = pd.DataFrame(pd.read_csv(args.file, parse_dates=True, index_col='time', date_parser=timeparser))

    if args.begin and args.end:
        Y = Y.loc[args.begin:args.end]
    elif args.begin:
        Y = Y.loc[args.begin:]
    elif args.end:
        Y = Y.loc[:args.end]
    return Y


def error(Y, xx, pred, steps_ahead):  # return average percent
    # x and xx must be indexes
    # this error penalizes more when the trend of the prediction is wrong.

    n = len(xx)
    total_error_penalized = 0
    total_error_non_pen = 0
    for i in xx:
        i = int(i)
        ii = i - int(xx[0])
        wrong_trend = np.sign(Y[i] - Y[i - steps_ahead]) != np.sign(pred[ii] - Y[i - steps_ahead])
        factor = 10 if wrong_trend else 1
        total_error_penalized += factor * np.abs(pred[ii] - Y[i]) / max(np.abs(Y[i]), 1) / n
        total_error_non_pen += np.abs(pred[ii] - Y[i]) / max(np.abs(Y[i]), 1) / n

    return 100 * total_error_penalized, 100 * total_error_non_pen


def real_data(len_for_prediction=None, n_pred_ahead=None):
    args = get_args()
    print(args.file)

    # plot VAR prediction ============================================
    title("VAR - using only the curve")

    Y = read_data(args)
    x = np.arange(len(Y.index))
    if not n_pred_ahead:
        n_pred_ahead = len(x) / 8
    if not len_for_prediction:
        len_for_prediction = n_pred_ahead

    # very bad results, forget it
    if False:
        xx, pred = test_forecast(x, Y.values, len_for_prediction=len_for_prediction, n_pred=n_pred_ahead,
                                 maxlags=30, ic='aic', trend='nc', verbose=False)
        Y = Y[['open', 'high', 'low', 'close']]
        pred = pred[:, [0, 1, 2, 3]]
        plot(x, Y.values)
        print(len(x), len(pred))
        plot(xx, pred, '--')
        show()
    # ==================================================================

    if True:
        title("AR - using the close price curve")
        xx, pred = test_predict(x, Y['close'].values, len_for_prediction=len_for_prediction, n_pred=n_pred_ahead,
                                maxlags=30, ic='aic', trend='nc', verbose=False)
        plot(x, Y['close'].values)
        plot(xx, pred, '--')
        print(error(Y['close'].values, xx, pred, n_pred_ahead))
        show()

    # ========================================================================================

    if False:
        tol = 0.4
        title('last slope forecast with tol={}'.format(tol))
        xx, pred = test_last_slope_forecast(x, Y['close'].values, len_for_prediction=len_for_prediction, n_pred=n_pred_ahead,
                                            verbose=True)
        plot(x, Y['close'].values)
        plot(xx, pred, '--')
        segs = seg_lin_reg(x, Y['close'].values, tol, 100)
        for seg in segs:
            print(seg[0])
            xx = x[seg[0]]
            yy = seg[2] * xx + seg[1]
            plot(xx, yy)
        print(error(Y['close'].values, xx, pred, n_pred_ahead))
        show()

    # seems bad too
    if False:
        tol = 0.4
        title('var slope forecast with tol={}'.format(tol))
        xx, pred = test_var_slope_forecast(x, Y['close'].values, len_for_prediction=len_for_prediction, n_pred=n_pred_ahead,
                                           verbose=True)
        plot(x, Y['close'].values)
        plot(xx, pred, '--')
        segs = seg_lin_reg(x, Y['close'].values, tol, 100)
        for seg in segs:
            print(seg[0])
            xx = x[seg[0]]
            yy = seg[2] * xx + seg[1]
            plot(xx, yy)
        print(error(Y['close'].values, xx, pred, n_pred_ahead))
        show()

    sys.exit(0)


if __name__ == '__main__':
    if False:
        artificial_data()
    else:
        sys.exit(real_data(len_for_prediction=300, n_pred_ahead=10))
