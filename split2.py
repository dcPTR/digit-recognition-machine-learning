import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tools import visualize_history, normalize_data, augment_data
import time

name = "MODEL2"

(x_train, y_train), (test_x, test_y) = mnist.load_data()
x_train = normalize_data(x_train)

x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
values = [y_train.tolist().count(i) for i in range(10)]
print(f"Categories number: {values}")

# split images based on category
category_x = [[] for i in range(10)]
category_y = [[] for i in range(10)]
for i in range(len(x_train)):
    category_x[y_train[i]].append(x_train[i])
    category_y[y_train[i]].append(y_train[i])

max_cat_number = max(values)
last_digit = max_cat_number % 10
new_max_count = max_cat_number + 10 - last_digit
for cat in range(10):
    category_x[cat] = np.array(category_x[cat]).reshape(-1, 28, 28, 1).astype('float32') / 255.
    k = new_max_count - values[cat]
    batch = augment_data(category_x[cat], category_y[cat], k)[0]
    print(f"Category: {cat}  -  {len(batch[0])} images added")
    x_train = np.concatenate([x_train, batch[0]])
    y_train = np.append(y_train, batch[1])

x_train, y_train = shuffle(x_train, y_train)

values = [y_train.tolist().count(i) for i in range(10)]
print(f"Categories after augmentation: {values}")

train_size, val_size, test_size = 0.8, 0.1, 0.1

x_train1, x_test1, y_train1, y_test1 = \
    train_test_split(x_train, y_train, test_size=test_size, shuffle=True, stratify=y_train)
x_train1, x_val1, y_train1, y_val1 = \
    train_test_split(x_train1, y_train1, test_size=val_size/(train_size+val_size), shuffle=True, stratify=y_train1)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten())  # it converts an N-dimensional layer to a 1D layer

# hidden layers
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Dense(256, activation='sigmoid'))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Dense(128, activation='relu'))

# output layer - the number of neurons must be equal to the number of classes
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint(f"{name}/{name}.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

start_time = time.time()
history = model.fit(x_train1, y_train1, epochs=5, validation_data=(x_val1, y_val1), callbacks=[checkpoint])
end_time = time.time()

visualize_history(history, name)

model.save(name)

print(f"{name} saved\n")

model = tf.keras.models.load_model(f"{name}/{name}.h5")

val_loss, val_acc = model.evaluate(x_val1, y_val1, verbose=2)
print(f"\nVal accuracy:  {val_acc*100} %")
test_loss, test_acc = model.evaluate(x_test1, y_test1, verbose=2)
print(f"\nTest accuracy: {test_acc*100} %")

print(f"\nTime: {(end_time - start_time):.2f} s")