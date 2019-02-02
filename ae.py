from keras import backend as K
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Lambda, Input
from keras.layers import Dense, Flatten, Reshape
from keras.layers import Conv2D, MaxPool2D, UpSampling2D
from keras.layers import LeakyReLU
from keras.optimizers import Adam

from matplotlib import pyplot as plt


def resize_images(inputs, dims_xy):
    x, y = dims_xy
    return Lambda(lambda im: K.tf.image.resize_images(im, (y, x)))(inputs)


# MNIST data
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1)

# encoder
enc_in = Input(shape=(28, 28, 1), name="enc_in")
x = Conv2D(16, (3, 3))(enc_in)
x = LeakyReLU()(x)
x = MaxPool2D()(x)
x = Conv2D(32, (3, 3))(x)
x = LeakyReLU()(x)
x = Flatten()(x)
z = Dense(100, name="z")(x)

enc = Model(enc_in, z, name="encoder")
enc.summary()

# decoder
dec_in = Input(shape=(100,), name="dec_in")
x = Dense(14 * 14 * 8)(dec_in)
x = LeakyReLU()(x)
x = Reshape((14, 14, 8))(x)
x = Conv2D(32, (3, 3))(x)
x = LeakyReLU()(x)
x = resize_images(x, (14, 14))
x = UpSampling2D()(x)
x = Conv2D(16, (3, 3))(x)
x = LeakyReLU()(x)
x = Conv2D(1, (3, 3), activation="linear")(x)
dec_out = resize_images(x, (28, 28))

dec = Model(dec_in, dec_out, name="decoder")
dec.summary()

# loss
def custom_loss(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred))

# complete model
outputs = dec(enc(enc_in))
ae = Model(enc_in, outputs, name="ae")
ae.compile(optimizer=Adam(lr=1e-4), loss=custom_loss)
ae.summary()

ae.fit(x=X_train, y=X_train, batch_size=256, epochs=10)
r = ae.predict(X_train[0:8])

f, axarr = plt.subplots(2, 8)
for i in range(8):
    axarr[0, i].imshow(X_train[i].reshape(28, 28), cmap="gray")
    axarr[1, i].imshow(r[i].reshape(28, 28), cmap="gray")
plt.show()
