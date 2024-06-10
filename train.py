from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from models.model import build_encoder, build_decoder
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
# reshaping images into channel
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

input_shape = (28, 28, 1)
encoded_shape = 64

encoder = build_encoder(input_shape)
decoder = build_decoder(encoded_shape)

autoencoder_input = Input(shape=input_shape)
encoded = encoder(autoencoder_input)
decoded = decoder(encoded)

autoencoder = Model(autoencoder_input, decoded, name="autoencoder")

# Compile the autoencoder
autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the autoencoder
autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True, validation_data=(x_test, x_test))

# Evaluate the autoencoder
loss = autoencoder.evaluate(x_test, x_test)
print(f'Test loss: {loss}')

# Save the encoder and decoder models
encoder.save("encoder.h5")
decoder.save("decoder.h5")    
