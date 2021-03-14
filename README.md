# deep_learning_projects

## Project 1:
The aim of this project is to compare alternative network architectures on a classication task. Nets are fed with two 14x14 downscaled versions of MNIST images and then classify the images based on which represents the greater digit.
We explore the use of auxiliary losses, weight-sharing and noise removal in variants of a principal architecture (described in the following section).



## Project 2:
In this mini-project we aimed to implement a mini-framework mimicking PyTorch. This mini-
framework should have similar usage to PyTorch so that it can be familiar to the user and easy to use. Additionally it should be coded in a straightforward manner so that it can be easily extended. Furthermore, it should efficiently and accurately classify the dummy data we give to it.

The structure of the frame-work is implemented with a single superclass Module which all
other implemented classes inherit. This structure allows all implemented classes to have a familiar structure and usage.

Multiple steps were taken during the coding of this project to ensure that the
implemented framework would be efficient. Importantly, we frequently used matrix operations to ensure that iteration through the data would be minimal. Additionally, training is optimized through a simple algorithm which adapts the learning rate as the training process continues. In order to minimize the amount of data which is passed between layers, activation layers are implemented as attributes of the layers.

In order to demonstrate the efficacy of this frame work we implement a simple network which uses only linear layers. This network has two input neurons, two output neurons, and three hidden layers with 25 neurons each. The framework, of course, is functional with other architectures. Two types of activation layers are implemented: rectied linear unit (ReLU) for the hidden layers, and hyperbolic tangent (Tanh) for the output layer.
