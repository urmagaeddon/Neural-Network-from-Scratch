Neural Network for Digit Classification

This project implements a simple 2-layer neural network from scratch to classify handwritten digits from the MNIST dataset. The network achieves approximately 84.4% accuracy on the validation set.

Project Structure

The implementation consists of:
1. A neural network with one hidden layer (10 neurons) and one output layer (10 neurons)
2. ReLU activation for the hidden layer
3. Softmax activation for the output layer
4. Gradient descent optimization

Key Features

- Forward Propagation: Computes predictions through the network
- Backward Propagation: Implements the backpropagation algorithm to compute gradients
- Training Loop: Performs gradient descent to optimize the network parameters
- Evaluation: Includes functions to test predictions and calculate accuracy

Usage

1. Load the MNIST dataset (train.csv)
2. Initialize the network parameters:
   
   W1, b1, W2, b2 = init_params()
   
3. Train the network:
   
   W1, b1, W2, b2 = gradient_descent(X_train, Y_train, learning_rate=0.10, iterations=500)
   
4. Make predictions:
   
   predictions = make_predictions(X, W1, b1, W2, b2)
   
5. Evaluate accuracy:

   accuracy = get_accuracy(predictions, Y)


Example Output

The training process outputs the accuracy every 10 iterations:

Iteration:  0
Accuracy: 0.1076
Iteration:  10
Accuracy: 0.1690
...
Iteration:  490
Accuracy: 0.8396

Final validation accuracy: 84.4%

Dependencies

- NumPy
- Pandas
- Matplotlib

Files

- train.csv: MNIST training data (not included in repository)
- Main Python script containing the neural network implementation

Notes

For production use, consider using deep learning frameworks like TensorFlow or PyTorch which offer more features and optimizations.
