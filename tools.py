import pandas as pd
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import numpy as np

def visualize_history(history: tf.keras.callbacks.History, title = "") -> None:
    """
    Visualize history of the training model.

    Parameters
    ----------
    history : tf.keras.callbacks.History
    title : str, optional
    """
    df_hist = pd.DataFrame(history.history)

    fig, axs = plt.subplots(1, 2)
    fig.set_figheight(9)
    fig.set_figwidth(16)
    fig.suptitle(title)

    axs[0].plot(df_hist["loss"], label="zbiór uczący")
    axs[0].plot(df_hist["val_loss"], label="zbiór walidacyjny")
    axs[0].set_title('Wartość funkcji kosztu podczas uczenia modelu')
    axs[0].set_xlabel('epoka')
    axs[0].set_ylabel('wartość')
    axs[0].legend()

    axs[1].plot(df_hist["accuracy"], label='zbiór uczący')
    axs[1].plot(df_hist["val_accuracy"], label='zbiór walidacyjny')
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

