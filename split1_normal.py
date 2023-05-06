import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from tools import visualize_history

name = "MODEL1"

(train_x, train_y), (test_x, test_y) = mnist.load_data()
train_size, val_size, test_size = 0.8, 0.1, 0.1

x_train1, x_test1, y_train1, y_test1 = \
    train_test_split(train_x, train_y, test_size=test_size, shuffle=True, stratify=train_y)
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

checkpoint = ModelCheckpoint(f"{name}/{name}.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')

history = model.fit(x_train1, y_train1, epochs=5, validation_data=(x_val1, y_val1), callbacks=[checkpoint])

visualize_history(history, name)

model.save(name)

print(f"{name} saved\n")

model = tf.keras.models.load_model(name)

val_loss, val_acc = model.evaluate(x_val1, y_val1, verbose=2)
print(f"\nVal accuracy:  {val_acc*100} %")
test_loss, test_acc = model.evaluate(x_test1, y_test1, verbose=2)
print(f"\nTest accuracy: {test_acc*100} %")

