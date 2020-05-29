from keras import layers
from keras import models
model = models.Sequential()
model.add(layers.Conv2D(32,(5,5),activation='relu',input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2,2)))

# adding one more Conv2D layer
model.add(layers.Conv2D(64,(5,5),activation='relu'))
# adding one more MaxPooling2D layer
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Flatten())
model.add(layers.Dense(10, activation='softmax'))
from keras.datasets import mnist
from keras.utils import to_categorical
(train_images, train_labels),(test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000,28,28,1))
train_images = train_images.astype('float32')/255
test_images = test_images.reshape((10000,28,28,1))
test_images = test_images.astype('float32')/255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(train_images, train_labels, batch_size=100, epochs=3, verbose=1)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test Accuracy:', test_acc)
