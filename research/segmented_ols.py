import numpy as np
import sys
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
from scipy.stats import stats
import statsmodels.api as sm
import math
import pandas as pd


def seg_lin_reg(x, y, tol=0.5, variance=None):
    # if |ynew - ypredicted| > tol*std, new a new segment is started
    # return list of (slice, alpha, beta)
    assert len(x) == len(y)
    assert len(x) > 0

    var_y_global = math.sqrt(np.var(y)) if not variance else variance

    i0 = 0
    mean_x0 = x[i0]
    mean_y0 = y[i0]
    var0 = 0.
    beta0 = 0.
    alpha0 = mean_y0
    result = []

    i = 1
    safe_counter = 0
    while i < len(x):
        safe_counter += 1
        assert safe_counter <= 2 * len(x)
        n = float(i - i0 + 1)
        x_new = x[i]
        y_new = y[i]
        mean_x = mean_x0 + (x_new - mean_x0) / n
        mean_y = mean_y0 + (y_new - mean_y0) / n
        dx = x_new - mean_x
        dy = y_new - mean_y
        var = _update_var(var0, n, dx)
        beta = _update_beta(beta0, n, dx, dy, var0, var)  # slope
        alpha = mean_y - beta * mean_x  # intercept
        y_exp = alpha0 + beta0 * x_new
        #print('i={}, i0={}, y_new={}, y_exp={}, condition={}, tol={}'
        #      .format(i, i0, y_new, y_exp, abs(y_new - y_exp) / max(abs(y_exp), 1.e-8), tol))
        # plot(x, y, 'o')
        # plot(x[i0:i+1], beta0*x[i0:i+1] + alpha0)
        # plot([x_new, x_new], [y_new, y_exp])
        # show()
        if i-i0 > 1 and abs(y_new - y_exp) > tol * var_y_global:
            result += [(slice(i0, i, 1), alpha0, beta0)]
            i0 = i-1
            mean_x0 = x[i0]
            mean_y0 = y[i0]
            var0 = 0.
            beta0 = 0.
            alpha0 = mean_y0
            continue
        else:
            mean_x0 = mean_x
            mean_y0 = mean_y
            var0 = var
            beta0 = beta
            alpha0 = alpha
        if i == len(x)-1:
            result += [(slice(i0, i+1, 1), alpha0, beta0)]
            break
        i += 1
    return result


# non normalized variance
def _update_var(var0, n, dx):
    return var0 + dx * dx * n / (n - 1.)


# slope
def _update_beta(beta0, n, dx, dy, var0, varx_new):
    return (dx * dy * n / (n - 1) + beta0 * var0) / varx_new
    # return beta0


def _assert_eq(x, y):
    assert y + 1.e-10 >= x >= y - 1.e-10


def _test1():
    np.random.seed(0)
    x = np.linspace(0., 10., 41)
    y1 = 2. - 1.5 * x  # (2,-1)
    y2 = 2. * x - 5.  # (3, 1)
    y3 = -x + 4.  # (5, -1)
    y4 = 2. * x - 11.
    y = np.array(x)
    y[np.where(x < 2)] = y1[np.where(x < 2)]
    y[np.where((x >= 2) & (x < 3))] = y2[np.where((x >= 2) & (x < 3))]
    y[np.where((x >= 3) & (x < 5))] = y3[np.where((x >= 3) & (x < 5))]
    y[np.where(5 <= x)] = y4[np.where(5 <= x)]
    # plot(x, y, 'o')
    # show()
    n = len(x)
    var_x0 = np.var(x[:-1]) * (n - 1.)
    var_y0 = np.var(y[:-1]) * (n - 1.)
    mean_x = np.mean(x[:-1]) + (x[-1] - np.mean(x[:-1])) / n
    mean_y = np.mean(y[:-1]) + (y[-1] - np.mean(y[:-1])) / n
    dx = x[-1] - mean_x
    dy = y[-1] - mean_y
    _assert_eq(np.var(x) * n, _update_var(var_x0, n, dx))
    _assert_eq(np.var(y) * n, _update_var(var_y0, n, dy))
    beta0 = np.cov(x[:-1], y[:-1], bias=True)[0][1] / np.var(x[:-1])
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    #print(slope)
    #print(np.cov(x, y, bias=True)  [0][1] / np.var(x))
    print('slope exact = {}, computed = {}'.format(slope, _update_beta(beta0, n, dx, dy, var_x0, np.var(x) * n)))
    print('intercept exact = {}, computed = {}'.format(intercept, mean_y - slope * mean_x))
    segs = seg_lin_reg(x, y, 0.0001)
    assert len(segs) == 4
    _assert_eq(segs[0][1], 2.), _assert_eq(segs[0][2], -1.5)
    _assert_eq(segs[1][1], -5.), _assert_eq(segs[1][2], 2.)
    _assert_eq(segs[2][1], 4.), _assert_eq(segs[2][2], -1.)
    _assert_eq(segs[3][1], -11.), _assert_eq(segs[3][2], 2.)

    plot_segments(x, y, 0.0001)

    # test spikes
    y[17] = 2
    y[7] = 0
    y[-1] = 7
    y[-2] = 6
    plot_segments(x, y, 0.0001)


def _test2(rand_err=1., tol=0.1):
    np.random.seed(0)
    x = np.linspace(0., 10., 41) + 0.*np.random.normal(0., .01, 41)
    y1 = 2. - 1.5 * x  # (2,-1)
    y2 = 2. * x - 5.  # (3, 1)
    y3 = -x + 4.  # (5, -1)
    y4 = 2. * x - 11.
    y = np.array(x)
    y[np.where(x < 2)] = y1[np.where(x < 2)]
    y[np.where((x >= 2) & (x < 3))] = y2[np.where((x >= 2) & (x < 3))]
    y[np.where((x >= 3) & (x < 5))] = y3[np.where((x >= 3) & (x < 5))]
    y[np.where(5 <= x)] = y4[np.where(5 <= x)]
    y += np.random.normal(0., rand_err, 41)

    plot_segments(x, y, tol, title='tol={}'.format(tol))


def _test3(rand_err=0., tol=0.1):
    np.random.seed(0)
    x = np.linspace(0., 10., 41) + 0.*np.random.normal(0., .01, 41)
    y = x * (10. - x)
    y += np.random.normal(0., rand_err, 41)
    plot_segments(x, y, tol, title='tol={}'.format(tol))


def _test4(tol=0.1):
    y = pd.DataFrame(pd.read_csv('data/bitmex_1day.csv', parse_dates=True, index_col='time'))['close'].values
    x = np.arange(len(y))
    plot_segments(x, y, tol, title='tol={}'.format(tol))


def plot_segments(x, y, tol, title=None):
    segs = seg_lin_reg(x, y, tol)
    #plot(x, y, 'o')
    plot(x, y, '-')
    for seg in segs:
        # print(seg[0])
        xx = x[seg[0]]
        yy = seg[2] * xx + seg[1]
        plot(xx, yy)
    if title:
        plt.title(title)
    show()


if __name__ == '__main__':
    # _test1()
    # _test2(rand_err=1., tol=0.1)
    # _test2(rand_err=1., tol=.5)
    # _test2(rand_err=1., tol=1.)
    #
    # _test2(rand_err=2., tol=0.1)
    # _test2(rand_err=2., tol=.5)
    # _test2(rand_err=2., tol=1.)
    #
    # _test3(rand_err=1., tol=0.1)
    # _test3(rand_err=1., tol=.5)
    # _test3(rand_err=1., tol=1.)
    _test4(tol=0.4)
    sys.exit(0)
