import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
import cyclic_esn

#__ create autoencoder model
def model_autoencoder(input_dim, encoding_dim):
    #.. input placeholder
    input_data = Input(shape=(input_dim,))
    #.. encoded representation of the input
    encoded = Dense(encoding_dim, activation='linear')(input_data)
    #.. lossy reconstruction of the input
    decoded = Dense(input_dim, activation='linear')(encoded)

    #.. autoencoder
    #.. this model maps an input to its reconstruction
    autoencoder = Model(input_data, decoded)

    #.. encoder
    #.. this model maps an input to its encoded representation
    encoder = Model(input_data, encoded)

    #.. create a placeholder for an encoded input
    encoded_input = Input(shape=(encoding_dim,))
    #.. retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]

    #.. decoder
    #.. create the decoder model
    decoder = Model(encoded_input, decoder_layer(encoded_input))

    autoencoder.compile(optimizer='nadam', loss='mean_squared_error')

    return encoder, decoder, autoencoder

#__ dimension reduction by standard autoencoder
def standard_AE(weights, dim_latent_space):
    #.. dimension of input data space
    input_dim = weights.shape[0]
    #.. size of our encoded representations
    encoding_dim = dim_latent_space
    
    #.. get autoencoder model
    encoder, decoder, autoencoder = \
    model_autoencoder(input_dim, encoding_dim)

    #.. fit autoencoder
    autoencoder.fit(weights.T, weights.T,
                    epochs=100,
                    batch_size=None,
                    shuffle=False)

    #.. encoded test data
    reduced_weights = encoder.predict(weights.T)

    #.. reconstructed test data
    # decoded_data = decoder.predict(encoded_data)

    return reduced_weights

if __name__ == "__main__":
    pass
