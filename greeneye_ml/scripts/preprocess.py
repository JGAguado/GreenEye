import tensorflow as tf


def load_datasets(data_dir, img_size=(224,224), batch_size=32, validation_split=0.2):
    """
    Loads training and validation datasets from a directory.
    Expects subdirectories for each class within data_dir.
    Returns prefetch-enabled tf.data.Dataset objects.
    """
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        labels='inferred',
        label_mode='categorical',
        validation_split=validation_split,
        subset='training',
        seed=123,
        image_size=img_size,
        batch_size=batch_size
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        labels='inferred',
        label_mode='categorical',
        validation_split=validation_split,
        subset='validation',
        seed=123,
        image_size=img_size,
        batch_size=batch_size
    )
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds