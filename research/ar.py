from statsmodels.tsa.ar_model import AR
from statsmodels.graphics.tsaplots import *
from matplotlib.pyplot import *
import scipy.signal as sg  # for scipy.signal.welch
from scipy.ndimage.interpolation import shift

# this script is just to understand how AR model works


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


N = 301
# x, y = n_hat(N, 6)
x, y = sin_(N, 5)
y = y + 0.5 * np.random.normal(0., .5, len(y))

#y = diff(y, 15)
#x = np.arange(len(y)).astype('float64')

# plot AR prediction ======================
model = AR(y)
model_fit = model.fit()
pred = model_fit.predict(start=model_fit.k_ar, end=int(2 * N), dynamic=False)
xx = np.arange(model_fit.k_ar, model_fit.k_ar + len(pred))

print(model_fit.k_ar)
print(model_fit.params)
plot(x, y)
plot(xx, pred)
show()
# # =======================================

# # plot SARIMAX prediction =================
# best_pdq = get_best_pdq_sarimax(y, max_d=2, max_q=1)
# print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA best pdq = " + str(best_pdq))
# model = SARIMAX(y, order=best_pdq)
# model_fit = model.fit(disp=0)
# pred = model_fit.predict(start=N/2, end=int(2 * N), dynamic=False)
# xx = np.arange(N/2, N/2 + len(pred))
# #sys.exit(0)
# print(model_fit.params)
# plot(x, y)
# plot(xx, pred)
# show()
# # =======================================

#per = sg.periodogram(y if len(y) % 2 == 0 else y[1:], fs=x[1], scaling='spectrum')
per = sg.welch(y, fs=x[1], nperseg=N if N%2==0 else N-1, detrend='linear')
# remove 0 freqs
freqs = per[0][1:]
periods = (1. / np.array(freqs))[::-1]
power_spectrum = np.array(per[1][1:])[::-1]
print("max spec = " + str(periods[np.argmax(power_spectrum)]))
plot(periods, power_spectrum)
show()

fig, axes = subplots(3, 1, figsize=(10, 9))
fig.tight_layout()
axes[0].plot(y)
axes[0].title.set_text('sin with err=0, slope=0, variance slope=3/101')
plot_acf(y, ax=axes[1], unbiased=True)
plot_pacf(y, ax=axes[2], method='ywm')
# autocorrelation_plot(df)
show()
