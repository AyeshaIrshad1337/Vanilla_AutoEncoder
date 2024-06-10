from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from models.model import create_model
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Flatten the 28x28 images into vectors of size 784
x_train = x_train.reshape((len(x_train), -1))
x_test = x_test.reshape((len(x_test), -1))
input_img = Input(shape=(784,))
decoder=create_model(input_img)
autoencoder = Model(input_img, decoder)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True, validation_data=(x_test, x_test))

