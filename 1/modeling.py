import keras
import tensorflow as tf
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
print(x_train.shape)
print(x_test.shape)

# view the first image
import matplotlib.pyplot as plt
plt.imshow(x_train[0], cmap='gray')
print(y_train[0])

# mormalize the data
x_train = x_train / 255.0
x_test = x_test / 255.0

# since MLP flatten the input, we need to flatten the input
x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)


network = keras.models.Sequential()
network.add(keras.layers.Dense(512, activation='relu', input_shape=(28*28,)))
network.add(keras.layers.Dense(512, activation='relu'))
network.add(keras.layers.Dense(10, activation='softmax'))

network.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

network.fit(x_train, y_train, epochs=20, batch_size=128,)


# Epoch 1/5
# 469/469 [==============================] - 4s 6ms/step - loss: 0.4750 - accuracy: 0.8305
# Epoch 2/5
# 469/469 [==============================] - 3s 6ms/step - loss: 0.3488 - accuracy: 0.8717
# Epoch 3/5
# 469/469 [==============================] - 3s 7ms/step - loss: 0.3116 - accuracy: 0.8854
# Epoch 4/5
# 469/469 [==============================] - 3s 7ms/step - loss: 0.2919 - accuracy: 0.8912
# Epoch 5/5
# 469/469 [==============================] - 3s 7ms/step - loss: 0.2732 - accuracy: 0.8986

# plot loss and accuracy
history = network.history.history
print(history)
plt.plot(history['loss'], label='loss')
plt.plot(history['accuracy'], label='accuracy')
plt.legend()

# evaluate the model
test_loss, test_acc = network.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)
# 313/313 [==============================] - 0s 998us/step - loss: 0.3401 - accuracy: 0.8711
# Test accuracy: 0.8711000084877014


# see what images are misclassified
predictions = network.predict(x_test)
predicted_classes = np.argmax(predictions, axis=1)
misclassified = np.where(predicted_classes != y_test)[0]
print(misclassified)

# plot the first 25 misclassified images
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_test[misclassified[i]].reshape(28,28), cmap='gray')
    plt.xlabel(f'Actual: {y_test[misclassified[i]]}, Predicted: {predicted_classes[misclassified[i]]}')
plt.show()
