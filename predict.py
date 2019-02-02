from keras.datasets import mnist
from keras.models import load_model
import numpy as np
from matplotlib import pyplot as plt
from vae import vae_loss

# MNIST data
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_test = X_test.reshape(-1, 28, 28, 1)
X_test = (X_test / 255.0).astype(np.float32)

# model
vae = load_model("vae.h5", custom_objects={"vae_loss": vae_loss})

# predict 8 examples
r = vae.predict(X_test[0:8])

# plot these examples
f, axarr = plt.subplots(2, 8)
for i in range(8):
    # original
    axarr[0, i].imshow(X_test[i].reshape(28, 28), cmap="gray")
    # predictions
    axarr[1, i].imshow(r[i].reshape(28, 28), cmap="gray")
plt.show()
