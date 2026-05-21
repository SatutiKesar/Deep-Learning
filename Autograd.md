**Autograd** is a core component of PyTorch that provides automatic differentiation for tensor operations. It automatically computes gradients for tensors, which is essential for training machine learning models using optimization algorithms such as Gradient Descent.

PyTorch tracks all operations performed on tensors with **requires_grad=True** and builds a computational graph dynamically during execution.


Why Autograd is Important

Autograd enables neural networks to learn by:
1. Computing gradients automatically
2. Reducing manual derivative calculations
3. Supporting backpropagation
4. Optimizing model parameters efficiently


Training Process includes:
1.	Forward pass - Compute the output of the network given an input.
2.	Calculate loss - Calculate the loss function to quantify the error.
3.	Backward pass - Compute gradients of the loss with respect to the parameters.
4.	Update gradients - Adjust the parameters using an optimization algorithm (e.g., gradient descent).
<img width="776" height="508" alt="autograd" src="https://github.com/user-attachments/assets/6abbd1d8-dc50-4284-8ed3-d7491ae28ea5" />
