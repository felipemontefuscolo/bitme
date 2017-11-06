import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import fft

pi = np.pi

data = pd.read_csv('bitmex_1day.csv')
N = len(data)
T = 1  # sample spacing

x = np.linspace(0, N*T, N)
#y = np.sin(.1 * 2*pi*x) + 0.5* np.sin(.3 * 2*pi*x) # for testing purposes
y = 0.5*(data['open'] + data['close']); y = y - np.mean(y)
yf = fft.rfft(y)
xf = np.linspace(0, 1./(2.*T), N/2 + 1)

fix, ax = plt.subplots()
#ax.plot(xf, np.log(2./N * np.abs(yf[:N//2])) + 1)
ax.plot(xf[1:], 2./N * np.abs(yf)[1:])
plt.show()

