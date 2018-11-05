# Title

Implementation of a fully connected feed-forward neural network using the back propagation algorithm for
stochastic batch gradient descent computations. Network size is adoptive and supports MLP.

# Features

**Activations**: _softmax_, _sigmoid_  

**Loss functions**: log-likelihood, mean-square, cross-entropy(binary equivalent of log-likelihood)

**Regularization**: L2

**Validation** - Takes labels and data as input

**Hyper-parameters**: learning-rate(eta), regularization-parameter(lambda), epochs

# Use case

Used for training and deploying human activity recognition system using wearable sensors.

Accuracy on test set using 6-9-6 neural network for multi-class classification using
softmax layer: **97.62%**

# Dependencies

numpy

matplotlib, sklearn, pandas, keras.utils : for preprocessing

# Author

Prathamesh Mandke - mandkepk97@gmail.com

# License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
