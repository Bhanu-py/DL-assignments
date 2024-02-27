import keras
import tensorflow as tf
import numpy as np
from keras.layers import Dropout

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
network.add(Dropout(0.2))
network.add(keras.layers.Dense(128, activation='relu'))
network.add(Dropout(0.2))
network.add(keras.layers.Dense(512, activation='relu'))
network.add(keras.layers.Dense(10, activation='softmax'))

network.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

network.fit(x_train, y_train, epochs=20, batch_size=128,)

## output
    # Epoch 1/20
    # 469/469 [==============================] - 4s 8ms/step - loss: 0.5329 - accuracy: 0.8077
    # Epoch 2/20
    # 469/469 [==============================] - 3s 7ms/step - loss: 0.3768 - accuracy: 0.8638
    # Epoch 3/20
    # 469/469 [==============================] - 3s 7ms/step - loss: 0.3393 - accuracy: 0.8760
    # Epoch 4/20
    # 469/469 [==============================] - 3s 7ms/step - loss: 0.3130 - accuracy: 0.8847
    # Epoch 5/20
    # 469/469 [==============================] - 3s 7ms/step - loss: 0.2961 - accuracy: 0.8911
    # Epoch 6/20
    # 469/469 [==============================] - 3s 7ms/step - loss: 0.2793 - accuracy: 0.8965
    # Epoch 7/20
    # 469/469 [==============================] - 3s 7ms/step - loss: 0.2725 - accuracy: 0.8988
    # Epoch 8/20
    # 469/469 [==============================] - 3s 7ms/step - loss: 0.2593 - accuracy: 0.9034
    # Epoch 9/20
    # 469/469 [==============================] - 3s 7ms/step - loss: 0.2507 - accuracy: 0.9065
    # Epoch 10/20
    # 469/469 [==============================] - 3s 7ms/step - loss: 0.2395 - accuracy: 0.9100
    # Epoch 11/20
    # 469/469 [==============================] - 3s 7ms/step - loss: 0.2296 - accuracy: 0.9132
    # Epoch 12/20
    # 469/469 [==============================] - 3s 7ms/step - loss: 0.2252 - accuracy: 0.9154
    # Epoch 13/20
    # 469/469 [==============================] - 3s 7ms/step - loss: 0.2186 - accuracy: 0.9179
    # Epoch 14/20
    # 469/469 [==============================] - 3s 7ms/step - loss: 0.2115 - accuracy: 0.9208
    # Epoch 15/20
    # 469/469 [==============================] - 3s 7ms/step - loss: 0.2035 - accuracy: 0.9224
    # Epoch 16/20
    # 469/469 [==============================] - 3s 7ms/step - loss: 0.1995 - accuracy: 0.9249
    # Epoch 17/20
    # 469/469 [==============================] - 3s 7ms/step - loss: 0.1968 - accuracy: 0.9251
    # Epoch 18/20
    # 469/469 [==============================] - 3s 7ms/step - loss: 0.1863 - accuracy: 0.9286
    # Epoch 19/20
    # 469/469 [==============================] - 3s 7ms/step - loss: 0.1794 - accuracy: 0.9320
    # Epoch 20/20
    # 469/469 [==============================] - 3s 7ms/step - loss: 0.1774 - accuracy: 0.9316


# plot loss and accuracy
history = network.history.history
print(history)
plt.plot(history['loss'], label='loss')
plt.plot(history['accuracy'], label='accuracy')
plt.legend()
plt.show()
## output
    #{'loss': [0.53292316198349, 0.37675461173057556, 0.3392738401889801, 0.31296929717063904, 0.29611217975616455, 0.27929654717445374, 0.27249088883399963, 0.2593202590942383, 0.25074219703674316, 0.23946473002433777, 0.22963565587997437, 0.22518783807754517, 0.2185746282339096, 0.21149609982967377, 0.2035059630870819, 0.19947710633277893, 0.19679787755012512, 0.18633471429347992, 0.17943862080574036, 0.17738743126392365], 'accuracy': [0.8076666593551636, 0.8637999892234802, 0.8760499954223633, 0.8847000002861023, 0.8910833597183228, 0.8965333104133606, 0.8988000154495239, 0.9033833146095276, 0.9065499901771545, 0.9099666476249695, 0.9131500124931335, 0.9153833389282227, 0.9179333448410034, 0.9208333492279053, 0.9224333167076111, 0.9248999953269958, 0.9251000285148621, 0.9285500049591064, 0.9320166707038879, 0.9315500259399414]}


# evaluate the model
test_loss, test_acc = network.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)
## output
    # 313/313 [==============================] - 0s 1ms/step - loss: 0.3373 - accuracy: 0.8971
    # Test accuracy: 0.8970999717712402


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
