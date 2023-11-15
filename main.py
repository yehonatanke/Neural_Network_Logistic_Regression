"""
Image Recognition Program

This Python script is part of an image recognition program designed to classify images as 'cat' or 'non-cat'.
It prepares a dataset for training and testing a machine learning model.

Features:
- Loads a dataset containing cat and non-cat images for training and testing.
- Reshapes and processes the data to be suitable for machine learning.
- Standardizes image data by flattening and normalizing pixel values.

Usage:
- Run this script in conjunction with other modules and utility files.
- Ensure that the necessary data files (e.g., train_catvnoncat.h5, test_catvnoncat.h5) are available.

Dependencies:
- numpy
- matplotlib
- scipy
- h5py
- PIL (Pillow)
- lr_utils
- util.py

For detailed usage and examples, refer to the README.md file in the repository.

Note: Before running the script, make sure to have the required dependencies installed.
"""

import h5py
import scipy
from util import *
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

# The dataset ("data.h5"):
# - a training set of m_train images labeled as cat (y=1) or non-cat (y=0)
# - a test set of m_test images labeled as cat or non-cat
# - each image is of shape (num_px, num_px, 3) where 3 is for the 3 channels (RGB).
#   Thus, each image is square (height = num_px) and (width = num_px).

# ----------------------
# Part 1: Reshape the training and test data sets

# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# Example of a picture (change the index for different pictures).
# If Y= [1] it's a cat, if Y = [0] it's not a cat.
index = 29
plot_example_image(index, train_set_x_orig, train_set_y, classes)

# The values of:
# - m_train (number of training examples)
# - m_test (number of test examples)
# - num_px (= height = width of a training image)
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

print_dataset_info(m_train, m_test, num_px, train_set_x_orig, train_set_y, test_set_x_orig, test_set_y)

# Reshape the training and test data sets so that images of size (num_px, num_px, 3)
# are flattened into single vectors of shape (num_px ∗ num_px ∗ 3, 1).
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

# Check that the first 10 pixels of the second image are in the correct place
assert np.alltrue(train_set_x_flatten[0:10, 1] == [196, 192, 190, 193, 186, 182, 188, 179, 174,
                                                   213]), "Wrong solution."
assert np.alltrue(test_set_x_flatten[0:10, 1] == [115, 110, 111, 137, 129, 129, 155, 146, 145,
                                                  159]), "Wrong solution."

print("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print("train_set_y shape: " + str(train_set_y.shape))
print("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print("test_set_y shape: " + str(test_set_y.shape), "\n")

# To standardize a color image dataset, dividing every pixel value by 255,
# the maximum value for a pixel channel. This simple normalization scales
# the pixel values to the range [0, 1], making it a convenient alternative
# to more complex mean subtraction and standard deviation division.
train_set_x = train_set_x_flatten / 255.
test_set_x = test_set_x_flatten / 255.

# Part 1 Summary:
# Common steps for pre-processing a new dataset are:
# Figure out the dimensions and shapes of the problem (m_train, m_test, num_px, ...)
# Reshape the datasets such that each example is now a vector of size (num_px * num_px * 3, 1)
# "Standardize" the data
# ---------------------- End of part 1

# ----------------------
# Part 2: General Architecture of the learning algorithm

# Mathematical expression of the algorithm:
# For one example, 𝑥⁽ⁱ⁾:
# 𝑧⁽ⁱ⁾ = 𝑤ᵀ𝑥⁽ⁱ⁾ + 𝑏
# ŷ⁽ⁱ⁾ = 𝑎⁽ⁱ⁾ = 𝑠𝑖𝑔𝑚𝑜𝑖𝑑(𝑧⁽ⁱ⁾)
# ℒ⁽ⁱ⁾(𝑎⁽ⁱ⁾, 𝑦⁽ⁱ⁾) = -𝑦⁽ⁱ⁾log(𝑎⁽ⁱ⁾) - (1-𝑦⁽ⁱ⁾)log(1-𝑎⁽ⁱ⁾)
# The cost is then computed by summing over all training examples: 𝐽 = 1/m ∑ⁱ=1ᵐ 𝓁(𝑎⁽ⁱ⁾, 𝑦⁽ⁱ⁾)

# Key steps:
# - Initialize the parameters of the model
# - Learn the parameters for the model by minimizing the cost
# - Use the learned parameters to make predictions (on the test set)
# - Analyse the results and conclude

# Building the parts of our algorithm:
# The main steps for building a Neural Network are:
# 1. Define the model structure (such as number of input features)
# 2. Initialize the model's parameters
# 3. Loop:
#   - Calculate current loss (forward propagation)
#   - Calculate current gradient (backward propagation)
#   - Update parameters (gradient descent)
# We often build 1-3 separately and integrate them into one function we call model().

# Train a logistic regression model on the given training and test sets
logistic_regression_model = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000,
                                  learning_rate=0.005, print_cost=True)

# Display an image from the test set along with its true and predicted labels (change the index).
index = 2
print_image(index, test_set_x, test_set_y, logistic_regression_model, classes, num_px)

# Plot the learning curve (with costs)
learning_curve_analysis(logistic_regression_model)

# Define a list of learning rates (Change the values)
learning_rates = [0.01, 0.001, 0.0001]
learning_rates_analysis(learning_rates, train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=1500)
