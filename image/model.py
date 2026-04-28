import tensorflow
from keras import datasets
from keras.datasets import cifar10
# Load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape)  
print(x_test.shape)

x_train = x_train / 255.0
x_test = x_test / 255.0

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential()

# Convolution layers
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))

# Fully connected
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))


# Save model
model.save("cifar10_model.keras")