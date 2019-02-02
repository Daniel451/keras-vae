from keras.datasets import mnist
from keras import callbacks
import numpy as np

from vae import vae, dec

# MNIST data
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1)
X_train = (X_train / 255.0).astype(np.float32)

# logging
tb = callbacks.TensorBoard(log_dir="./log", write_graph=True)

# train
vae.fit(x=X_train, y=X_train, batch_size=128, epochs=20, callbacks=[tb])

vae.save("vae.h5")
dec.save("dec.h5")
