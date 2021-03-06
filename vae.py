from keras import backend as K
from keras.models import Model
from keras.layers import Lambda, Input
from keras.layers import Dense, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, MaxPool2D, UpSampling2D
from keras.layers import LeakyReLU
from keras.losses import mse
from keras.optimizers import Adam


def sampling(z):
    z_mean, z_var = z
    batch_size = K.shape(z_mean)[0]
    z_size = K.int_shape(z_mean)[1]
    eps = K.random_normal(shape=(batch_size, z_size))

    return z_mean + K.exp(0.5 * z_var) * eps


# network parameters
input_shape = (28, 28, 1)
batch_size = 128
latent_dim = 2

# encoder
enc_in = Input(shape=(28, 28, 1), name="enc_in")
x = Conv2D(16, (3, 3), padding="same")(enc_in)
x = LeakyReLU()(x)
x = MaxPool2D()(x)
x = Conv2D(32, (3, 3), padding="same")(x)
x = LeakyReLU()(x)
shape = K.int_shape(x)
x = Flatten()(x)
x = Dense(100)(x)
x = LeakyReLU()(x)
z_mean = Dense(latent_dim, name="z_mean")(x)
z_var = Dense(latent_dim, name="z_var")(x)
z = Lambda(sampling, output_shape=(latent_dim,), name="z")([z_mean, z_var])

enc = Model(enc_in, [z_mean, z_var, z], name="encoder")

# decoder
dec_in = Input(shape=(latent_dim,), name="dec_in")
x = Dense(shape[1] * shape[2] * shape[3])(dec_in)
x = LeakyReLU()(x)
x = Reshape((shape[1], shape[2], shape[3]))(x)
x = Conv2DTranspose(32, (3, 3), padding="same")(x)
x = LeakyReLU()(x)
x = UpSampling2D()(x)
x = Conv2DTranspose(16, (3, 3), padding="same")(x)
x = LeakyReLU()(x)
x = Conv2DTranspose(1, (3, 3), activation="linear", padding="same")(x)
dec_out = x

dec = Model(dec_in, dec_out, name="decoder")

# loss
def vae_loss(y_true, y_pred):
    loss_im = mse(K.flatten(y_true), K.flatten(y_pred))
    loss_im *= 28*28
    loss_kl = -0.5 * K.sum(1 + z_var - K.square(z_mean) - K.exp(z_var), axis=-1)
    loss = K.mean(loss_im + loss_kl)

    return loss

# complete model
outputs = dec(enc(enc_in)[2])
vae = Model(enc_in, outputs, name="vae")
vae.compile(optimizer=Adam(lr=1e-4), loss=vae_loss)



