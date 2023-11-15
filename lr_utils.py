import numpy as np
import h5py


def load_dataset():
    # Load the training dataset
    train_dataset = h5py.File('/path/to/data/datasets/train_catvnoncat.h5', "r")

    # Extract training set features and labels
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # train set labels

    # Load the testing dataset
    test_dataset = h5py.File('/path/to/data/datasets/test_catvnoncat.h5', "r")

    # Extract test set features and labels
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # test set labels

    # Extract the list of classes
    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    # Reshape labels to match the expected format
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    # Return the loaded dataset
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
