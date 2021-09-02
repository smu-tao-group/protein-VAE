#!/usr/bin/env python

"""
VAE/AE model architectures
"""

from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras import losses
from tensorflow.keras.optimizers import Adam

# increment of neurons between adjacent layers
POWER = {1: 64, 2: 16, 3: 8, 4: 4, 5: 4}


def get_neuron_layer(original_dim, latent_dim, num_of_hidden_layer):
    """Get the number of neurons in each layer.

    Parameters
    ----------
    original_dim : int
        number of dim in the original space
    latent_dim : int
        number of dim in latent space
    num_of_hidden_layer: int
        number of hidden layers except for the input and latent space

    Returns
    -------
    neuron_layer : list
        number of neurons in each layer in ascending order from
        latent dim to original dim
    """

    assert num_of_hidden_layer in POWER, "Invalid number of hidden layer"

    neuron_layer = [latent_dim]

    # start from 16 in layer 4
    if num_of_hidden_layer == 4:
        neuron_layer.append(16)
    elif num_of_hidden_layer == 5:
        neuron_layer.append(8)

    while neuron_layer[-1] < original_dim:
        neuron_layer.append(
            min(neuron_layer[-1] * POWER[num_of_hidden_layer], original_dim)
            )

    # need to delete the last second layer to have 4 layers
    if num_of_hidden_layer == 4:
        neuron_layer.pop(-2)

    return neuron_layer


def vae_encoder(original_dim, latent_dim=2, num_of_hidden_layer=4):
    """Build VAE encoder model.

    Parameters
    ----------
    original_dim : int
        number of dim in the original space
    latent_dim : int (optional)
        number of dim in latent space
    num_of_hidden_layer: int (optional)
        number of hidden layers except for the input and latent space

    Returns
    -------
    encoder : keras.Model
        constructed encoder model
    z_mean : keras.Model.Dense
    z_log_var : keras.Model.Dense
    encoder_input : keras.Model.Input
        input layer
    """

    input_shape = (original_dim, )
    encoder_input = x = Input(shape=input_shape)

    neuron_layer = get_neuron_layer(
        original_dim, latent_dim, num_of_hidden_layer
        )

    for i in range(len(neuron_layer) - 2, 0, -1):
        x = Dense(neuron_layer[i], activation='relu')(x)

    z_mean = Dense(latent_dim)(x)
    z_log_var = Dense(latent_dim)(x)

    def sampling(args):
        z_mean, z_log_var = args

        epsilon = K.random_normal(
            shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.
            )

        return z_mean + K.exp(z_log_var) * epsilon

    z = Lambda(sampling)([z_mean, z_log_var])

    # create encoder
    encoder = Model(encoder_input, [z_mean, z_log_var, z])

    return encoder, z_mean, z_log_var, encoder_input


def vae_decoder(original_dim, latent_dim=2, num_of_hidden_layer=4):
    """Build VAE/AE decoder model.
    Can be used for both VAE and AE.

    Parameters
    ----------
    original_dim : int
        number of dim in the original space
    latent_dim : int (optional)
        number of dim in latent space
    num_of_hidden_layer: int (optional)
        number of hidden layers except for the input and latent space

    Returns
    -------
    decoder : keras.Model
        constructed decoder model
    """

    latent_inputs = x = Input(shape=(latent_dim,))

    neuron_layer = get_neuron_layer(
        original_dim, latent_dim, num_of_hidden_layer
        )

    for i in range(1, len(neuron_layer) - 1):
        x = Dense(neuron_layer[i], activation='relu')(x)

    x = Dense(original_dim, activation='sigmoid')(x)

    decoder = Model(latent_inputs, x)
    return decoder


def build_vae(original_dim, latent_dim=2, num_of_hidden_layer=4):
    """Build VAE model.

    Parameters
    ----------
    original_dim : int
        number of dim in the original space
    latent_dim : int (optional)
        number of dim in latent space
    num_of_hidden_layer: int (optional)
        number of hidden layers except for the input and latent space

    Returns
    -------
    encoder : keras.Model
        constructed encoder model
    decoder : keras.Model
        constructed decoder model
    vae: keras.Model
        constructed VAE model
    """

    # init encoder and decoder
    encoder, z_mean, z_log_var, encoder_input = vae_encoder(
        original_dim, latent_dim, num_of_hidden_layer
        )

    decoder = vae_decoder(original_dim, latent_dim, num_of_hidden_layer)

    # same as: z_decoded = decoder(z)
    z_decoded = decoder(encoder(encoder_input)[2])

    def vae_loss(x, z_decoded):
        # reconstruction loss
        x = K.flatten(x)

        z_decoded = K.flatten(z_decoded)
        xent_loss = losses.binary_crossentropy(x, z_decoded)

        kl_loss = -5e-4 * K.mean(
            1 + z_log_var - K.square(z_mean) - K.exp(z_log_var),
            axis=-1
            )

        return K.mean(xent_loss + kl_loss)

    # Instantiate the VAE model:
    vae = Model(encoder_input, z_decoded)
    vae.add_loss(vae_loss(encoder_input, z_decoded))

    opt = Adam(learning_rate=0.0001)
    vae.compile(optimizer=opt)

    return encoder, decoder, vae


def ae_encoder(original_dim, latent_dim=2, num_of_hidden_layer=4):
    """Build AE encoder model.

    Parameters
    ----------
    original_dim : int
        number of dim in the original space
    latent_dim : int (optional)
        number of dim in latent space
    num_of_hidden_layer: int (optional)
        number of hidden layers except for the input and latent space

    Returns
    -------
    encoder : keras.Model
        constructed encoder model
    encoder_input : keras.Model.Input
        input layer
    """

    # encoder
    input_shape = (original_dim, )
    encoder_input = x = Input(shape=input_shape)

    neuron_layer = get_neuron_layer(
        original_dim, latent_dim, num_of_hidden_layer
        )

    for i in range(len(neuron_layer) - 2, -1, -1):
        x = Dense(neuron_layer[i], activation='relu')(x)

    # create encoder
    encoder = Model(encoder_input, x, name='encoder')

    return encoder, encoder_input


def build_ae(original_dim, latent_dim=2, num_of_hidden_layer=4):
    """Build AE model.

    Parameters
    ----------
    original_dim : int
        number of dim in the original space
    latent_dim : int (optional)
        number of dim in latent space
    num_of_hidden_layer: int (optional)
        number of hidden layers except for the input and latent space

    Returns
    -------
    encoder : keras.Model
        constructed encoder model
    decoder : keras.Model
        constructed decoder model
    ae: keras.Model
        constructed AE model
    """

    # init encoder and decoder
    encoder, encoder_input = ae_encoder(
        original_dim, latent_dim, num_of_hidden_layer)
    decoder = vae_decoder(original_dim, latent_dim, num_of_hidden_layer)

    # same as: z_decoded = decoder(z)
    z_decoded = decoder(encoder(encoder_input))

    # Instantiate the VAE model:
    ae = Model(encoder_input, z_decoded)

    opt = Adam(learning_rate=0.0001)
    ae.compile(optimizer=opt, loss='binary_crossentropy')

    return encoder, decoder, ae
