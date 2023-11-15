import numpy as np
import copy
import matplotlib.pyplot as plt


def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    - z (numpy.ndarray or float): A scalar or numpy array of any size.

    Returns:
    numpy.ndarray or float: The sigmoid of the input z.

    Formula:
    - σ(x) = 1 / (1 + e^(-x))
    """
    return 1 / (1 + np.exp(-z))


def initialize_with_zeros(dim):
    """
    Initializes parameters w and b for a linear model with zero values.

    Parameters:
    - dim (int or float): Size of the parameter vector w (number of parameters).

    Returns:
    tuple: A tuple containing the initialized weight vector w and bias scalar b.

    Raises:
    ValueError: If dim is not a positive integer or convertible to an integer.

    Returns:
    tuple:
        - w (numpy.ndarray): Initialized weight vector of shape (dim, 1).
        - b (float): Initialized bias scalar.

    Example:
    >>> initialize_with_zeros(3)
    (array([[0.],
           [0.],
           [0.]]), 0.0)
    """
    if not (isinstance(dim, int) or (isinstance(dim, float) and dim.is_integer())):
        raise ValueError("dim must be a positive integer or convertible to an integer.")

    # Convert dim to an integer if it's a float
    dim = int(dim)

    if dim <= 0:
        raise ValueError("dim must be a positive integer.")

    # Initialize weight vector w with zeros and bias scalar b to 0.0
    w = np.zeros((dim, 1))
    b = 0.0

    return w, b


def propagate(w, b, X, Y):
    """
    Compute the cost function and its gradient for logistic regression.

    Parameters:
    - w (numpy.ndarray): Weights, a numpy array of shape (num_px * num_px * 3, 1).
    - b (float): Bias, a scalar.
    - X (numpy.ndarray): Data of shape (num_px * num_px * 3, number of examples).
    - Y (numpy.ndarray): True label vector (0 for non-cat, 1 for cat) of shape (1, number of examples).

    Returns:
    tuple: A tuple containing:
        - grads (dict): Dictionary containing the gradients of the weights and bias.
            - "dw" (numpy.ndarray): Gradient of the loss with respect to w, same shape as w.
            - "db" (float): Gradient of the loss with respect to b, a scalar.
        - cost (float): Negative log-likelihood cost for logistic regression.
    """

    m = X.shape[1]

    # Check dimensions
    if w.shape != (X.shape[0], 1) or Y.shape != (1, m):
        raise ValueError("Invalid dimensions for w, X, or Y.")

    # FORWARD PROPAGATION
    Z = np.dot(w.T, X) + b
    A = sigmoid(Z)

    # Clip values to avoid numerical instability
    A = np.clip(A, 1e-15, 1 - 1e-15)

    cost = (-1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

    # BACKWARD PROPAGATION
    dw = (1 / m) * np.dot(X, (A - Y).T)
    db = (1 / m) * np.sum(A - Y)

    grads = {"dw": dw, "db": db}

    return grads, cost


def optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False):
    """
    Optimize the weights and bias parameters using gradient descent.
    The goal is to learn 𝑤 and 𝑏 by minimizing the cost function 𝐽

    Parameters:
    - w (numpy.ndarray): Weights, a numpy array of shape (num_px * num_px * 3, 1).
    - b (float): Bias, a scalar.
    - X (numpy.ndarray): Data of shape (num_px * num_px * 3, number of examples).
    - Y (numpy.ndarray): True label vector (0 for non-cat, 1 for cat) of shape (1, number of examples).
    - num_iterations (int): Number of iterations of the optimization loop.
    - learning_rate (float): Learning rate of the gradient descent update rule.
    - print_cost (bool): True to print the loss every 100 steps.

    Returns:
    tuple: A tuple containing:
        - params (dict): Dictionary containing the optimized weights w and bias b.
        - grads (dict): Dictionary containing the gradients of the weights and bias with respect to the cost function.
        - costs (list): List of all the costs computed during the optimization, used for plotting the learning curve.

    Example:
    >>> w_initial = np.zeros((3, 1))
    >>> b_initial = 0.0
    >>> X_train = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> Y_train = np.array([[1, 0, 1]])
    >>> params, grads, costs = optimize(w_initial, b_initial, X_train, Y_train, num_iterations=500, learning_rate=0.01, print_cost=True)
    """

    w = copy.deepcopy(w)
    b = copy.deepcopy(b)

    costs = []

    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
            # Print the cost every 100 training iterations
            if print_cost:
                print(f"Cost after iteration {i}: {cost}")

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs


def predict(w, b, X):
    """
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b).

    This function applies logistic regression to make predictions on input data X based on
    the learned weights (w) and bias (b).

    There are two steps to computing predictions:
        1. Calculate ŷ = 𝐴 = 𝜎(𝑤𝑇𝑋+𝑏)
        2. Convert the entries of a into 0 (if activation <= 0.5) or 1 (if activation > 0.5),
           stores the predictions in a vector Y_prediction

    Parameters:
    - w (numpy.ndarray): Weights, a numpy array of shape (num_px * num_px * 3, 1).
    - b (float): Bias, a scalar.
    - X (numpy.ndarray): Input data of shape (num_px * num_px * 3, number of examples).

    Returns:
    Y_prediction (numpy.ndarray): Predictions, a numpy array (vector) containing
    all predictions (0/1) for the examples in X.
    """

    # Get the number of examples
    m = X.shape[1]

    # Initialize the prediction vector
    Y_prediction = np.zeros((1, m))

    # Reshape weights for compatibility
    w = w.reshape(X.shape[0], 1)

    # Compute the activation values
    A = sigmoid(np.dot(w.T, X) + b)

    # Convert probabilities to actual predictions
    for i in range(A.shape[1]):
        if A[0, i] > 0.5:
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0

    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    """
    Builds the logistic regression model by calling the function you've implemented previously.

    Parameters:
    - X_train (numpy.ndarray): Training set represented by a numpy array of shape (num_px * num_px * 3, m_train).
    - Y_train (numpy.ndarray): Training labels represented by a numpy array (vector) of shape (1, m_train).
    - X_test (numpy.ndarray): Test set represented by a numpy array of shape (num_px * num_px * 3, m_test).
    - Y_test (numpy.ndarray): Test labels represented by a numpy array (vector) of shape (1, m_test).
    - num_iterations (int): Hyperparameter representing the number of iterations to optimize the parameters.
    - learning_rate (float): Hyperparameter representing the learning rate used in the update rule of optimize().
    - print_cost (bool): Set to True to print the cost every 100 iterations.

    Returns:
    d (dict): Dictionary containing information about the model.
        - 'costs' (list): List of costs computed during training.
        - 'Y_prediction_test' (numpy.ndarray): Predictions on the test set.
        - 'Y_prediction_train' (numpy.ndarray): Predictions on the training set.
        - 'w' (numpy.ndarray): Learned weights.
        - 'b' (float): Learned bias.
        - 'learning_rate' (float): Learning rate used in training.
        - 'num_iterations' (int): Number of iterations used for training.
    """

    # Initialize parameters with zeros
    w, b = initialize_with_zeros(X_train.shape[0])

    # Gradient descent
    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    # Retrieve parameters w and b from dictionary "params"
    w = params["w"]
    b = params["b"]

    # Predict test/train set examples
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # Print train/test errors
    if print_cost:
        train_accuracy = 100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100
        test_accuracy = 100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100
        print("Train accuracy: {} %".format(train_accuracy))
        print("Test accuracy: {} %".format(test_accuracy))

    d = {
        "costs": costs,
        "Y_prediction_test": Y_prediction_test,
        "Y_prediction_train": Y_prediction_train,
        "w": w,
        "b": b,
        "learning_rate": learning_rate,
        "num_iterations": num_iterations
    }

    return d


def learning_rates_analysis(learning_rates, train_set_x, train_set_y, test_set_x, test_set_y,
                            num_iterations=1500):
    """
    Compare the learning curve of our model with several choices of learning rates
    (Train logistic regression models with different learning rates and plot their training costs).

    Parameters:
    - learning_rates (list): List of learning rates to experiment with.
    - train_set_x (numpy.ndarray): Training set features.
    - train_set_y (numpy.ndarray): Training set labels.
    - test_set_x (numpy.ndarray): Test set features.
    - test_set_y (numpy.ndarray): Test set labels.
    - num_iterations (int): Number of iterations for training. Default is 1500.

    Example:
    learning_rates_analysis(
        learning_rates=[0.01, 0.001, 0.0001],
        train_set_x=train_set_x,
        train_set_y=train_set_y,
        test_set_x=test_set_x,
        test_set_y=test_set_y,
        num_iterations=2000)
    """
    print("\n----\nLearning rate analysis:\n")

    # Initialize a dictionary to store trained models
    models = {}

    # Train logistic regression models for each learning rate
    for lr in learning_rates:
        print("Training a model with learning rate: " + str(lr))
        models[str(lr)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=num_iterations,
                                learning_rate=lr, print_cost=False)
        print('\n' + "-------------------------------------------------------" + '\n')

    # Plot the training costs over iterations for each learning rate
    for lr in learning_rates:
        plt.plot(np.squeeze(models[str(lr)]["costs"]), label=str(models[str(lr)]["learning_rate"]))

    # Set labels for the plot
    plt.ylabel('cost')
    plt.xlabel('iterations (hundreds)')
    plt.title('Learning Rate Analysis')

    # Add a legend to the plot
    legend = plt.legend(loc='upper center', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')

    # Display the plot
    plt.show()


def print_image(index, test_set_x, test_set_y, logistic_regression_model, classes, num_px):
    """
    Display an image from the test set along with its true and predicted labels.

    Parameters:
    - index (int): Index of the image in the test set.
    - test_set_x (numpy array): Test set features.
    - test_set_y (numpy array): Test set labels.
    - logistic_regression_model (dict): Dictionary containing model predictions.
    - classes (list): Array of classes (e.g., ['cat', 'non-cat']).
    - num_px (int): Size of the image in pixels (assuming it's a square image).

    Returns:
    None
    """
    # Display the image
    plt.figure()
    plt.imshow(test_set_x[:, index].reshape((num_px, num_px, 3)))
    plt.title(f"Example Image {index + 1}")
    plt.show()

    # Extract true and predicted labels
    true_label = test_set_y[0, index]
    predicted_label = int(logistic_regression_model['Y_prediction_test'][0, index])

    # Print labels
    print(f"True Label: {classes[true_label]}, Predicted Label: {classes[predicted_label]}")


def learning_curve_analysis(logistic_regression_model):
    """
    Plot the learning curve with costs from a logistic regression model.

    Parameters:
    - logistic_regression_model (dict): Dictionary containing model information.

    Returns:
    None
    """
    # Extract costs from the model
    costs = np.squeeze(logistic_regression_model['costs'])

    # Plot learning curve
    plt.figure()
    plt.plot(costs)
    plt.ylabel('Cost')
    plt.xlabel('Iterations (per hundreds)')
    plt.title("Learning Rate =" + str(logistic_regression_model["learning_rate"]))
    plt.show()


def plot_example_image(index, train_set_x_orig, train_set_y, classes):
    """
    Display an example image from the training set along with its true label.

    Parameters:
    - index (int): Index of the image in the training set.
    - train_set_x_orig (numpy array): Original training set features.
    - train_set_y (numpy array): Training set labels.
    - classes (list): Array of classes (e.g., ['cat', 'non-cat']).

    Returns:
    None
    """
    # Display the image
    plt.figure()
    plt.imshow(train_set_x_orig[index])
    plt.title(f"Example Image {index}")
    plt.show()

    # Extract true label
    true_label = train_set_y[0, index]

    print("Y = " + str(train_set_y[:, index]) + ", it's a '" +
          classes[np.squeeze(train_set_y[:, index])].decode("utf-8") + "' picture.")


def print_dataset_info(m_train, m_test, num_px, train_set_x_orig, train_set_y, test_set_x_orig, test_set_y):
    """
    Print information about the dataset.

    Parameters:
    - m_train (int): Number of training examples.
    - m_test (int): Number of testing examples.
    - num_px (int): Height/width of each image.
    - train_set_x_orig (numpy array): Original training set features.
    - train_set_y (numpy array): Training set labels.
    - test_set_x_orig (numpy array): Original test set features.
    - test_set_y (numpy array): Test set labels.

    Returns:
    None
    """
    print("Number of training examples: m_train = " + str(m_train))
    print("Number of testing examples: m_test = " + str(m_test))
    print("Height/Width of each image: num_px = " + str(num_px))
    print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
    print("train_set_x shape: " + str(train_set_x_orig.shape))
    print("train_set_y shape: " + str(train_set_y.shape))
    print("test_set_x shape: " + str(test_set_x_orig.shape))
    print("test_set_y shape: " + str(test_set_y.shape))
