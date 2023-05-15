# Handwritten Digits Classification

This is a project that aims to classify handwritten digits using machine learning. The dataset used for this project is the MNIST dataset, which is a well-known benchmark dataset for image classification.

## Dataset

The MNIST dataset contains 70,000 images of handwritten digits, each of size 28x28 pixels. The dataset is split into a training set of 60,000 images and a test set of 10,000 images.

## Model

The model used for this project is a convolutional neural network (CNN). The CNN is a type of neural network that is particularly well-suited for image classification tasks. The model is trained on the training set and evaluated on the test set.

## Requirements

To run this project, you will need the following Python packages:

- TensorFlow
- Keras
- NumPy
- Matplotlib

You can install these packages using pip:

```
pip install tensorflow keras numpy matplotlib
```

## Usage

To train the model, run the `train.py` script:

```
python train.py
```

This will train the model on the MNIST dataset and save the trained weights to a file called `model.h5`.

To evaluate the model on the test set, run the `evaluate.py` script:

```
python evaluate.py
```

This will load the trained weights from the `model.h5` file and evaluate the model on the test set.

## Results

The CNN achieves an accuracy of around 99% on the MNIST test set, which is a very good result for this task.

## Conclusion

This project demonstrates how to use machine learning to classify handwritten digits using the MNIST dataset. The CNN model achieves a high accuracy on the test set and can be used as a basis for more advanced image classification tasks.
