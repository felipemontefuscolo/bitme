import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from solver import solve
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random as rn

tf.logging.set_verbosity(tf.logging.ERROR)

# MAKE IT DETERMINISTIC
np.random.seed(42)
rn.seed(12345)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)
from tensorflow.keras import backend as K
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


# CREATE X and Y
x = (np.arange(-10.,11) + np.random.rand(21)).reshape(1,21)
# y = 0.3 * x + np.random.rand(21) + 1
np.random.seed(3)
def fun(x):
    if x < -5:
        return -(x + 5.)
    if x < 5:
        return x + 5.
    return -x + 15.
np_fun = np.vectorize(fun)
y = np_fun(x) + np.random.normal(scale=.5, size=len(x[0]))
A, b = solve(x, y)


def build_model2():
    # leaky = lambda x: tf.nn.leaky_relu(x, alpha=.2)
    model_in = layers.Input(shape=(1,))
    model = keras.Sequential([
        layers.Dense(2, input_shape=(1,), activation=tf.nn.leaky_relu, kernel_initializer='zeros'),
        #layers.Dense(1, input_shape=(1,), kernel_initializer='zeros'),
        layers.Dense(1, kernel_initializer='zeros')
    ])
    optimizer = tf.keras.optimizers.RMSprop(0.1)
    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    return model

model2 = build_model2()
EPOCHS = 1000
class PrintDot(keras.callbacks.Callback):
    def __init__(self):
        self.i = 0
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0 and epoch > 0: print('{}%'.format(int(epoch*100/EPOCHS)))
        print('.', end='')
        self.i = (self.i + 1) % 10
        
early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=30)

history = model2.fit(
    x[0], y[0],
    epochs=EPOCHS,
    verbose=0,
    callbacks=[early_stop, PrintDot()])

if True:
    plt.plot(x[0], y[0], '.-')
    plt.plot(x[0], model2.predict(x[0])[:,0])
    plt.plot(x[0], (A.dot(x) + b)[0])
    plt.show()

print()
for i in model2.get_weights():
    print(i)