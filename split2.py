import numpy as np
import pandas as pd
import tensorflow as tf
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def visualize_history(history: tf.keras.callbacks.History) -> None:
    """
    Visualize history of the training model.

    Parameters
    ----------
    history : tf.keras.callbacks.History
    """
    df_hist = pd.DataFrame(history.history)

    fig, axs = plt.subplots(1, 2)
    fig.set_figheight(9)
    fig.set_figwidth(16)

    axs[0].plot(df_hist["loss"], label="zbiór uczący")
    axs[0].set_title('Wartość funkcji kosztu podczas uczenia modelu')
    axs[0].set_xlabel('epoka')
    axs[0].set_ylabel('wartość')
    axs[0].legend()

    axs[1].plot(df_hist["accuracy"], label='zbiór uczący')
    axs[1].set_title('Skuteczności modelu podczas uczenia')
    axs[1].set_xlabel('epoka')
    axs[1].set_ylabel('wartość')
    axs[1].legend()

    plt.tight_layout()
    plt.show()


def normalize_data(data):
    data = data.astype(np.float32)
    for image_id in range(len(data)):
        data[image_id] = data[image_id] / 255. * 2. - 1.
    return data


def augment_data(x, y, batch_size):
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
    )
    return train_datagen.flow(x, y, batch_size=batch_size)


(x_train, y_train), (test_x, test_y) = mnist.load_data()
x_train = normalize_data(x_train)

x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
values = [0] * 10
for i in range(len(x_train)):
    values[y_train[i]] = values[y_train[i]] + 1
print("Categories number: " + str(values))

# split images based on category
category_x = [0] * 10
category_y = [0] * 10
for i in range(10):
    category_x[i] = [0] * values[i]
    category_y[i] = [0] * values[i]
cat_counter = [0] * 10
for i in range(len(x_train)):
    category_x[y_train[i]][cat_counter[y_train[i]]] = x_train[i]
    category_y[y_train[i]][cat_counter[y_train[i]]] = y_train[i]
    cat_counter[y_train[i]] += 1

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

values = [0] * 10
for i in range(len(y_train)):
    values[y_train[i]] = values[y_train[i]] + 1
print("Categories after augmentation: " + str(values))

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

history = model.fit(x_train1, y_train1, epochs=5)

visualize_history(history)

model.save('MODEL2')

print("MODEL2 saved\n")

model = tf.keras.models.load_model('MODEL2')

predictions = model.predict(x_val1)

error = 0
for i in range(len(x_val1)):
    guess = (np.argmax(predictions[i]))
    actual = y_val1[i]
    if guess != actual:
        error += 1

print("\nAccuracy = " + str(((len(predictions)-error)/len(predictions)) * 100) + "%")
