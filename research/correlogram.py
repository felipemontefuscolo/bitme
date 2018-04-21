import sys
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
import matplotlib.pyplot as plt
import random as rd
import scipy.signal as sg  # for scipy.signal.welch
from statsmodels.tsa.stattools import adfuller
from research.detect_peaks import strong_peak


# how to do forecast and detect rolling mean http://www.seanabu.com/2016/03/22/time-series-seasonal-ARIMA-model-in-python/
# auto arima : https://stackoverflow.com/questions/22770352/auto-arima-equivalent-for-python

# every filter applied has to be undone later

# step 1.: dectet trend:
#       a) fit the data to eq E(y_t) = b0 + b1*t + b2*t**2
#       b) subtract it out from the orignal data
#       OR
#       a) differentiate the data
# step 2.: remove seasonality (I'm skeptical about this one)
#       a) detect season: https://stats.stackexchange.com/questions/16117/what-method-can-be-used-to-detect-seasonality-in-data 
#          construct periodogram using scipy.signal.welch(df, fs=1, nperseg=N) and take the frequency with higher sampling
#       b) detect peaks with https://stackoverflow.com/questions/31910524/calculate-histogram-peaks-in-python
#           or with https://github.com/MonsieurV/py-findpeaks/blob/master/tests/libs/detect_peaks.py
#          (see also realtime peak detection: https://stackoverflow.com/questions/22583391/peak-signal-detection-in-realtime-timeseries-data
# step 3.:



def test_stationarity(timeseries, period):
    # Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=period)
    rolstd = pd.rolling_std(timeseries, window=period)

    # Plot rolling statistics:
    fig = plt.figure(figsize=(12, 8))
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()

    # Perform Dickey-Fuller test:
    print 'Results of Dickey-Fuller Test:'
    dftest = adfuller(timeseries, autolag='AIC', regression='ctt')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print dfoutput


rd.seed(9001)


def diff(x, d):
    # don't confuse np.diff(n) with pd.diff(n) -- the former calculate the n-th derivative, the latter differentiate n times
    return np.array([float('nan')] * d + list(np.diff(x, d)))


def white(N):
    return np.array([rd.gauss(0.0, 1.0) for i in range(N)])


def Lop(y, lag):
    lag = np.sign(lag) * min(abs(lag), len(y))
    x = list(y)
    N = len(x)
    if lag < 0:
        x[-lag:N] = x[0:N + lag]
        x[0:-lag] = [float('nan')] * (-lag)
    else:
        x[0:N - lag] = x[lag:N]
        x[N - lag:N] = [float('nan')] * (lag)
    return np.array(x)


def arima(a, d, b, cte, base, err_mag=1):
    w = err_mag * np.array(white(len(base)))
    base = diff(np.array(base), d)
    r = np.array(base)
    for i in range(len(a)):
        r += a[i] * Lop(base, -i - 1)
    for i in range(len(b)):
        r += b[i] * Lop(w, i + 1)
    r += cte + w
    return r


def stepf(x, L, v):
    return v if 0 <= x < L else 0


def create1(N):
    return [x + 3 for x in range(N)]


def create2(N, err_mag=0.):  # sin func with 4 periods  8*pi/(N-1) * T = 2*pi  -> T = (N-1)/4
    return [np.sin(8. * np.pi / (N - 1) * x) + err_mag * rd.gauss(0.0, 1.0) for x in range(N)]


def create3(N, err_mag=0., slope=0., vari_slope=0.):
    return [slope * float(x) + (1. + vari_slope * float(x)) * np.sin(8. * np.pi / (N - 1) * x) + err_mag * rd.gauss(0.0,
                                                                                                                    1.0)
            for x in range(N)]


def create_cos(num_points=101, num_periods=5, err_mag=0., mean_slope=0., vari_slope=0., freq_slope=0., normalize=False):
    x = np.arange(num_points).astype('float64')
    if normalize:
        x = x / float(num_points)
    freq = freq_slope * x + 2.*np.pi*float(num_periods)/(num_points - 1)
    vari = (1. + vari_slope * x)
    err = err_mag * np.array([rd.gauss(0., 1.) for i in range(num_points)])
    mean = mean_slope * x

    return mean + vari * np.cos(freq * x) + err


def create4(N):
    a = [0 for i in range(N)]
    w = N / 10
    for i in range(N):
        a[i] = stepf(i - N / 10, w, 1) + stepf(i - N * 3 / 10, w, 1) + 0 * stepf(i - N / 2, w, 1) + stepf(
            i - N * 7 / 10, w, 1) + stepf(i - N * 9 / 10, w, 1)
        a[i] += stepf(i - N * 2 / 5, w, .3)
    return a


def downstairs(N=100, n_steps=5):
    step_len = N / n_steps
    d = -1
    a = [n_steps for i in range(N)]
    for i in range(N):
        if i % step_len == 0:
            d += 1
        a[i] -= d
    return a


# some predefined curvers
r001 = arima(a=[], d=0, b=[], cte=0, base=[float(i) for i in range(100)], err_mag=5)  # growing straight line with error
r002 = arima(a=[], d=0, b=[], cte=0, base=[float(i) for i in range(100)], err_mag=5)  # growing straight line with error
r005 = arima(a=[], d=0, b=[], cte=0, base=create2(100), err_mag=0.7)  # noisy sin


def filter1(x):  # remove trend (make it stationary on mean)
    return x.diff(1).dropna()


def filter2(x):  # make series stationary on variance and mean
    return np.log(x.diff(1).dropna())


def filter3(x, T):  # remove season
    return (x - x.shift(T)).dropna()


# df = pd.DataFrame()
# df['t'] = create3(N)
# df = filter3(pd.Series(create2(101)), 25)
df = pd.Series(create3(101, err_mag=.9, slope=5. / 101., vari_slope=3. / 101.))

N = len(df)

Nh = N - 1


def max_peak(x, y):
    # return None if peak is not relevant    
    return


def to_clip(x):
    print "["
    for i in range(len(x)):
        s = str(x[i])
        if i != len(x) - 1:
            s += ','
        print s
    print ']'


def experiment1():
    n = 301
    x = [float(i) for i in range(n)]
    #y = np.array(downstairs(n, 9)) + 0.6 * white(n)
    y = create_cos(num_points=n, num_periods=5, err_mag=0., mean_slope=10./n, vari_slope=0./n, freq_slope=.0/n)
    # y[300:] = 5
    # y = create3(n+1)
    # y = [ np.exp(i/(float(n)) + 0.2*np.sin(8.*np.pi / (n-1) * i) for i in range(n) ]
    # y = [ (i - float(n/2))**2/(float(n/2)**2) + 0.2*np.sin(16.*np.pi / (n-1) * i) for i in range(n) ]

    pars = np.polyfit(x, y, 2)
    fit = np.poly1d(pars)(x)
    y_orig = y
    y = np.array(y) - fit
    print("fitted y: mean={}".format(y.mean()))

    per = sg.welch(y, fs=1., nperseg=len(y))
    # per = sg.welch(y, fs=1., nperseg=len(y), detrend='linear')
    # remove 0 freqs
    freqs = per[0][1:]
    periods = (1. / np.array(freqs))[::-1]
    power_spectrum = np.array(per[1][1:])[::-1]
    imax = strong_peak(periods, power_spectrum, min_hight=2., show=True, verbose=True, ax=plt)

    #if imax:
    #    plt.plot(periods[imax], power_spectrum[imax], 'ro')
    # plt.plot(per[0], per[1])
    plt.figure()
    ly, = plt.plot(y, label='de-trended')
    ly_orig, = plt.plot(y_orig, label='original')
    plt.legend(handles=[ly, ly_orig])
    # dydT = pd.Series(y) - pd.Series(y).shift(1)
    # plt.figure(); dydT.plot()
    plt.show()
    if imax:
        print("found period = {}".format(int(periods[imax])))
        dT = np.array((pd.DataFrame(y) - pd.DataFrame(y).shift(int(periods[imax]))).dropna()[0])
        test_stationarity(y, len(y))
    #print to_clip(power_spectrum[::-1])
    sys.exit(0)


def experiment2():

    sys.exit(0)

experiment1()

df.dropna(inplace=True)
# print(df)
print('---')
alpha = 0.09  # tuned value so that sin() function have spike on its period
unbiased = False
fft = True
acf_x, confint = acf(df, alpha=alpha, nlags=Nh, unbiased=unbiased, fft=fft)
pacf_x, pconfint = pacf(df, alpha=alpha, nlags=Nh, method='ywunbiased')
lower_band = confint[:, 0] - acf_x
upper_band = confint[:, 1] - acf_x
z95 = 1.959963984540054
z99 = 2.5758293035489004
print("important features (acf > confidence band) according to statsmodels:")
print(list((np.where(np.abs(acf_x) > upper_band)[0])))
print("important features (acf > confidence band = " + str(z99 / np.sqrt(N)) + ") according to pandas:")
print(list((np.where(np.abs(acf_x) > z99 / np.sqrt(N))[0])))

# plt.figure()
fig, axes = plt.subplots(3, 1, figsize=(10, 9))
fig.tight_layout()
axes[0].plot(df)
axes[0].title.set_text('sin with err=0, slope=0, variance slope=3/101')
plot_acf(df, ax=axes[1], alpha=alpha, lags=Nh, unbiased=unbiased, fft=fft)
plot_pacf(df, ax=axes[2], alpha=alpha, lags=Nh, method='ywm')
# autocorrelation_plot(df)
plt.show()


# print(df)

# def main():
# if __name__ == '__main__':
#    sys.exit(main())


# c = 1/3 * ((3-4)^2 + (5-4)^2)
