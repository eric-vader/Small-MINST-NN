import numpy as np
import mnist
from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, Reshape, Dropout, Conv2D, Flatten
from keras.utils import to_categorical

train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

# Normalize the images.
train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5

# Flatten the images.
'''
train_images = train_images.reshape((-1, 784))
test_images = test_images.reshape((-1, 784))
input_shape=(784,)
'''
print(train_images.shape)
img_rows, img_cols = 28, 28
train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, 1)
test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

# Build the model.
'''
model = Sequential([
  MaxPooling2D(pool_size=(4, 4), input_shape=input_shape),
  Reshape((49,)),
  Dense(16, activation='relu'),
  Dense(10, activation='relu'),
  Dense(10, activation='softmax'),
])
'''

'''
model = Sequential([
  MaxPooling2D(pool_size=(2, 2), input_shape=input_shape),
  Conv2D(10, kernel_size=(3, 3), activation='relu',padding='same'),
  
  MaxPooling2D(pool_size=(4, 4)),
  Conv2D(10, kernel_size=(2, 2), activation='relu',padding='same'),
  
  MaxPooling2D(pool_size=(2, 2)),
  Conv2D(10, kernel_size=(1, 1), activation='softmax',padding='same'),

  Flatten(),
])
Very good
'''

model = Sequential([
  MaxPooling2D(pool_size=(2, 2), input_shape=input_shape),
  Conv2D(6, kernel_size=(3, 3), activation='relu',padding='same'),
  
  MaxPooling2D(pool_size=(4, 4)),
  Conv2D(8, kernel_size=(2, 2), activation='relu',padding='same'),
  
  MaxPooling2D(pool_size=(2, 2)),
  Conv2D(10, kernel_size=(1, 1), activation='softmax',padding='same'),

  Flatten(),
])

# Compile the model.
model.compile(
  optimizer='adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

model.summary()

# Train the model.
model.fit(
  train_images,
  to_categorical(train_labels),
  epochs=12,
  batch_size=16,
)

test_result = model.evaluate(
  test_images,
  to_categorical(test_labels)
)

print(test_result)


