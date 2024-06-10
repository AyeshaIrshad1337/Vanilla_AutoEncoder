import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Input, Reshape

from tensorflow.keras.models import Model
def create_model(input_img):
    latent_dim = 64
    # Encoder
    
    encoded = Dense(128, activation='relu')(input_img)
    encoded = Dense(64, activation='relu')(encoded)
    encoded= Dense(latent_dim, activation='relu')(encoded)
    # Decoder
    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(784, activation='sigmoid')(decoded)
    return decoded
# Define the encoder
def build_encoder(input_shape):
    encoder_input = Input(shape=input_shape)
    x = Flatten()(encoder_input)
    x = Dense(128, activation='relu')(x)
    encoder_output = Dense(64, activation='relu')(x)
    encoder = Model(encoder_input, encoder_output, name="encoder")
    return encoder

# Define the decoder
def build_decoder(encoded_shape):
    decoder_input = Input(shape=encoded_shape)
    x = Dense(128, activation='relu')(decoder_input)
    x = Dense(28 * 28, activation='sigmoid')(x)
    decoder_output = Reshape((28, 28, 1))(x)
    decoder = Model(decoder_input, decoder_output, name="decoder")
    return decoder