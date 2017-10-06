# NeuralNetworkEigen

Neural Network implementation using the Eigen library for C++.
This Neural Network attempts to solve a -CLASSIFICATION- problem.
It is intended as a dummy guideline to create more complex NN's.
All the operations needed were tested and are implmemented:
Call order is like this :
 -stochastic gradient descent calls -> update mini batch
 which calls back propagation -> which calls feedforward

In this C++ class that implments  neural network.
The sizes are 2 nodes in the input layer,
              3 nodes in the hidden layer
              2 notes in the output layer.
                      
The Eigen library has proven useful in it's creation.
The dynamic matrices Eigen::MatrixXf are handy.

I tried to make it as dynamic as possible (If I wanted to change
the number of nodes per layer, or the number of hidden layers)
but it was hard to do.

For this reason, in the future, if the sizes of my layers change, 
care will have to be taken to keep the matrix dimensions straight.

Efforts will be make towards making it more dynamic so it may be 
created more dynamically - for any number of layers and any number of 
layers.



