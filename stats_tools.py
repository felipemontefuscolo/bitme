from scipy.optimize import brute
from sklearn.linear_model import Ridge
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
from sklearn import linear_model


def get_best_pdq_arma(data, max_p=5, max_q=5):
    grid = (slice(1, max_p + 1, 1), slice(1, max_q + 1, 1))
    return [int(x) for x in brute(_objfunc_arma, grid, args=[data], finish=None)]


def _objfunc_arma(order, data):
    print("order = " + str(order))
    try:
        fit = ARMA(data, order=order).fit(transparams=False, disp=0)
    except ValueError as e:
        print("value error in objfunc")
        return 999999.

    if np.isnan(fit.aic):
        return 9999999.
    else:
        print("obtained aic = " + str(fit.aic))
        return fit.aic


def get_best_pdq_sarimax(data, max_p=7, max_d=3, max_q=7):
    grid = (slice(1, max_p + 1, 1), slice(0, max_d + 1, 1), slice(1, max_q + 1, 1))
    return [int(x) for x in brute(_objfunc_sarimax, grid, args=[data], finish=None)]


def _objfunc_sarimax(order, data):
    print("order = " + str(order))
    try:
        fit = SARIMAX(data, order=order).fit(transparams=False, disp=0)
    except:
        print("value error in objfunc")
        return 999999.

    if np.isnan(fit.aic):
        return 9999999.
    else:
        print("obtained aic = " + str(fit.aic))
        return fit.aic
