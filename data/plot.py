import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack

pi = np.pi


N = len(pd.read_csv('bitmex_1day.csv'))
T = 1  # sample spacing
T = 1.

x = np.linspace(0., N*T, N)
y = np.sin(50. * 2.*pi*x) + 0.5* np.sin(80. * 2.*pi*x)  # for testing purposes
#y = pd.read_csv('bitmex_1day.csv'); y = 0.5*(y['open'] + y['close'])
yf = scipy.fftpack.fft(y)
xf = np.linspace(0, 1./(2.*T), N/2)

fix, ax = plt.subplots()
ax.plot(xf, 2./N * np.abs(yf[:N//2]))
plt.show()
