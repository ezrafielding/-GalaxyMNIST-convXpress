import h5py
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

def get_data_from_h5(filename):
    """Gets Data from Galaxy MNIST hdf5 files.

    Args:
        filename: Path to file.

    Returns:
        images and labels as numpy arrays.
    """
    with h5py.File(filename, 'r') as f:
        # Get images
        images = np.array(f['images'])
        labels = np.array(f['labels'])
        
        # Convert labels to 10 categorical classes
        labels = tf.keras.utils.to_categorical(labels, 4)
    return images, labels

def image_prep(x):
    """Pre-processes Images.

    Args:
        x: Dictionary containing image and label.

    Returns:
        augmented image and label.
    """
    # Convert Images to tf.float32 data type
    image = tf.cast(x['image'], tf.float32)
    # Normalize pixel values
    image = image / 255
    # Remove clour by taking mean over channels
    grey_image = tf.reduce_mean(input_tensor=image, axis=2, keepdims=True)
    # Ensuring dimensions are correct
    assert grey_image.shape[0] == 64
    assert grey_image.shape[1] == 64
    assert grey_image.shape[2] == 1
    # Ensure new tensor is returned
    aug_image = tf.identity(grey_image)
    return aug_image, x['label']

def make_tf_Dataset(train_images, train_labels, test_images, test_labels):
    """Packs data into a tensorflow dataset.

    Args:
        train_images: Images used for training.
        train_labels: Labels used for training.
        test_images: Images used for testing.
        test_lables: Labels used for testing.

    Returns:
        Train and Test tensorflow datasets.
    """
    # Pack Images and Labels into dataset
    train = tf.data.Dataset.from_tensor_slices({"image":train_images, "label":train_labels})
    test = tf.data.Dataset.from_tensor_slices({"image":test_images, "label":test_labels})

    # Chache, shuffle and batch dataset. Image pre-processing also done
    train = train.map(
        lambda x: image_prep(x)
    ).cache().shuffle(100).batch(128)
    test = test.map(
        lambda x: image_prep(x)
    ).cache().batch(128)

    return train,test

def get_data(train_filename, test_filename):
    """Gets the dataset data.

    Args:
        filename: Path to dataset file.

    Returns:
        Train and Test tensorflow datasets.
    """
    # Fetch Images and labels
    train_images, train_labels = get_data_from_h5(train_filename)
    test_images, test_labels = get_data_from_h5(test_filename)
    # Pack data into datasets and pre-process images
    train, test = make_tf_Dataset(train_images, train_labels, test_images, test_labels)
    return train, test
