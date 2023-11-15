# Neural_Network_Logistic_Regression
Neural Network Logistic Regression: Python script for image classification, using machine learning to identify cat and non-cat images. Implements logistic regression with neural network architecture.

## Features

- Loads a dataset containing cat and non-cat images for training and testing.
- Reshapes and processes the data to be suitable for machine learning.
- Standardizes image data by flattening and normalizing pixel values.

## Usage

- Run this script in conjunction with other modules and utility files.
- Ensure that the necessary data files (e.g., train_catvnoncat.h5, test_catvnoncat.h5) are available.

## Dependencies

- numpy
- matplotlib
- scipy
- h5py
- PIL (Pillow)
- lr_utils
- util.py

## Acknowledgments

This project is based on the materials provided in the 'Deep Learning Specialization' course on Coursera. The foundational concepts, logic, and inspiration for this project are derived from the course materials created by the DeepLearning.AI team.

Attribution

Course: Neural Networks and Deep Learning
Instructor: Andrew Ng
Platform: Coursera
Please note that while the code structure has been extensively modified, the core ideas and problem-solving approach are based on the educational content provided in the course.

## Installation

```bash
pip install numpy matplotlib scipy h5py Pillow
```
## Configuration

If you need to customize certain paths or settings, follow these steps:

1. Open the `lr_util.py` file located in the `lr_utils/` directory.
2. Look for the following line:

    ```python
    data_path = "/path/to/data/datasets/..."
    ```

3. Replace `"/path/to/data/datasets/"` with the actual path to your data file.

   For example, if your data file is located in the `data/` directory of your project, you can set it as follows:

    ```python
    data_path = "data/your_data_file.h5"
    ```

4. Save the changes.

### License

This project is licensed under the MIT License - see the LICENSE file for details.

### Author

yehonataKe
