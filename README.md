## 

1) Data generator and visualizer (data_generator.py)

GUI enables the generation and visualization of data samples from two classes. Samples are two-dimensional in order to allow plotting them in x-y space. Each class consist of one or more Gaussian modes with their means and variances generated randomly for specified interval. The interface allows for setting the desired number of modes per class and samples per mode. Class labels are indicates by colors.

2) Single neuron (neuron.py)

Implementation of an artificial neuron. Neuron takes generated samples and predict their class membership at its output. Implemented neuron allows for different activation function for evaluation: Heaviside, sigmoid, sin, tanh, sign, ReLu and leaky Realu.

3) Three-layer neural network 

Implementation of Three-layer fully connected neural network. The output is presented in the form of two values describing the confidence that the network belongs to each class. The number of neurons in the input and hidden layer are configurable. Neurons use the logistics activation function. GUI presents the decision boundary for the network through colouring the corresponding parts of the plot. Model operates on a dataset with multiple modes per class.