from matplotlib.pyplot import *
from scipy.ndimage.interpolation import shift

# this script is just to understand how AR model works
from statsmodels.tsa.vector_ar.var_model import VAR


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
    y = np.cos(2. * np.pi / T * x * (1. + 0.00*x))
    return x, y


def forecast(Y, var_results, Npred):
    N = len(Y)
    pred = list(model_fit.forecast(Y[:var_results.k_ar], Npred))
    for i in range(var_results.k_ar + 1, N - Npred):
        pred += [list(model_fit.forecast(Y[:i], Npred)[-1])]
    return pred

N = 301
# x, y = n_hat(N, 6)
x, y = sin_(N, 5)
y = y + 0.5 * np.random.normal(0., .5, len(y))
z = y*y
Y = np.matrix([y, z]).transpose().tolist()

# plot AR prediction ======================
model = VAR(Y)
model_fit = model.fit(maxlags=15, ic='aic')
pred = model_fit.forecast(Y[-model_fit.k_ar:], N)
xx = np.arange(N, N + len(pred))
assert(len(pred) == N)
print(model_fit.k_ar)
print(model_fit.params)
plot(x, Y)
plot(xx, pred, '--')
show()
# # =======================================

pred = forecast(Y, model_fit, 100)
xx = np.arange(model_fit.k_ar + 1, model_fit.k_ar + 1 + len(pred))
plot(x, Y)
plot(xx, pred, '--')
show()

