**Autograd** is a core component of PyTorch that provides automatic differentiation for tensor operations. It automatically computes gradients for tensors, which is essential for training machine learning models using optimization algorithms such as Gradient Descent.

PyTorch tracks all operations performed on tensors with **requires_grad=True** and builds a computational graph dynamically during execution.


Why Autograd is Important

Autograd enables neural networks to learn by:
i) Computing gradients automatically
ii) Reducing manual derivative calculations
iii) Supporting backpropagation
iv) Optimizing model parameters efficiently


Training Process includes:
1.	Forward pass - Compute the output of the network given an input.
2.	Calculate loss - Calculate the loss function to quantify the error.
3.	Backward pass - Compute gradients of the loss with respect to the parameters.
4.	Update gradients - Adjust the parameters using an optimization algorithm (e.g., gradient descent).
