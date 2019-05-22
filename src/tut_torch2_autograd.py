# Sample Network
#
# (Input Layer)--fully-conn.-->(Single ReLU Layer)--fully-conn.-->(Output Layer)
#
# This version uses the basics of pytorch. Now we use the autograd stuff! yay!
import torch

dtype  = torch.float
device = torch.device("cpu") 

NUM_ITERS = 500
N = 64 # batch size
D_in, H, D_out = 1000, 100, 10 # Input, hidden, output layer dimensions

# Creating Random Data
x = torch.randn(N, D_in,  device=device, dtype=dtype) 
y = torch.randn(N, D_out, device=device, dtype=dtype)

# Layer Weights - First time we see a difference. 
#				- The extra parameter tells torch that we would like it to
#				- compute the gradient/backwards pass for this op/node
w1 = torch.randn(D_in, H,  device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(NUM_ITERS):
	# Forward Pass - No longer need to save intermediate values. Only did that 
	#			   - to ensure we could manually calculate the gradients. Note,
	#			   - same operations, just on one line. 
	y_pred = x.mm(w1).clamp(min=0).mm(w2)

	# Loss
	loss = (y_pred - y).pow(2).sum()
	print(t, loss.item())

	# Backward Pass - No work needed! This tells torch to have the tensor compute
	#				- and store gradients for all nodes that have indicated they
	#				- need to have these calculated.
	loss.backward()	

	# Update weights from gradients
	# Note - This can, apparently, be done many differeny ways. But this seems to 
	# be the 'canonical' way of doing it? Still unsure. 
	with torch.no_grad():
		w1 -= learning_rate * w1.grad
		w2 -= learning_rate * w2.grad 
		# Zero the grads out?
		w1.grad.zero_()
		w2.grad.zero_()