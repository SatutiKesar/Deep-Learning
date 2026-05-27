Steps for GPU Training
1.	Check GPU Availability - Before starting, verify if a GPU is available. If available, select it; otherwise use the CPU.
	code:-
		import torch
		#check for GPU
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		print(f"Using device: {device}")
2.  Move the Model to GPU - Move your model to the selected device (cuda for GPU or 	cpu) so that all computations occur on the same device.
	code:-
		model = MyModel()                         # Replace with your model
		model = model.to(device)	   # Move model to GPU
3. Modify the Training Loop by Moving Data to GPU - Ensure that each batch of data 
	(features and labels) is moved to the GPU before processing. This ensures that both 	the model and data are on the same device.
	code:-
	for batch_features, batch_labels in train_loader:
		# Move data to GPU
		batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
4. Modify the Evaluation Loop by Moving Data to GPU - Similarly, ensure test data is 	moved to the GPU during evaluation. Disable gradeint descent calculations using torch.no_grad() for efficiency.
	code:- 
	with torch.no_grad():
	     for batch_features, batch_labels in test_loader:
	     # Move data to GPU
	     batch_features, batch_labels = batch_features.to(device), batch_labels.to(device) 
5. Optimize the GPU Usage - To make the best use of GPU resources, apply the following optimizations:

a. Use Larger Batch Sizes - Larger batch sizes can better utilize GPU memory and reduce computation time per epoch (if memory allows).
	code:- 
  	 train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, pin_memory=True)
  	 test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True, pin_memory=True)
	
b. Enable DataLoader Pinning - Use pin_memory=True in DataLoader to speed up data transfer from CPU to GPU.
	   train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, pin_memory=True)

GPU and Deep Learning
A GPU accelerates the heavy math in neural networks:
1. matrix multiplications
2. convolutions
3. gradient calculations
4. parallel operations on tensors
   
Training modern models without GPUs is usually extremely slow.
The following techniques all run during training, so GPUs execute their computations efficiently.

Dropout
1.	Applied to the hidden layers
2.	Applied after the ReLU activation function
3.	Randomly turns off p% neurons in the hidden layer during each forward pass
4.	This has a regularization effect
5.	During evaluation dropout is not used


Batch Norm
1. Applied to Hidden Layers: Typically applied to the hidden layers of a neural network, but not to the output layer.
2. Applied After Linear Layers and Before Activation Functions: Normalizes the output of the preceding layer(e.g., after nn.Linear) and is usually followed by an activation function(e.g., ReLU).
3. Normalizes Activation: Computes the mean and variance of the activations within a mini-batch and uses these statistics to normalize the activations.
4. Includes Learnable Parameters: Introduces two learnable parameters, gamma(scaling) and beta(shifting), which allow the network to adjust the normalized outputs.
5. Improves Training Stability: Reduces internal covariate shift, stabilizing the training process and allowing the use of higher learning rates.
6. Regularization Effect: Introduces some regularization because the statistics are computed over a mini-batch, adding noise to the training process.
7. Consistent During Evaluation: During evaluation, BatchNorm uses the running mean and variance accumulated during training, rather than recomputing them from the mini-batch.

L2 Regularization
1. Applied to Model Weights: Regularization is applied to the weights of the model to penalize large values and encourage smaller, more generalizable weights.
   
3. Penalizes Large Weights: Encourages the network to distribute learning across multiple parameters, avoiding reliance on a few large weights.













