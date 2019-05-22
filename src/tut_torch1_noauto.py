# Sample Network
#
# (Input Layer)--fully-conn.-->(Single ReLU Layer)--fully-conn.-->(Output Layer)
#
# This version uses the basics of pytorch. It's really not much different from the 
# numpy code, but the first few lines show the changes. You explicitly control
# what device, and therefore what kinds of possible optimizations, are being used. 
import torch

dtype  = torch.float
device = torch.device("cpu") # Heres where you say something like:
							 # torch.device("cuda:0") 
							 # to get the code optimzed for CUDA enabled systems.

NUM_ITERS = 500
N = 64 # batch size
D_in, H, D_out = 1000, 100, 10 # Input, hidden, output layer dimensions

# Creating Random Data
x = torch.randn(N, D_in,  device=device, dtype=dtype) # Note the only dif is the device and dtype
y = torch.randn(N, D_out, device=device, dtype=dtype)

# Layer Weights
w1 = torch.randn(D_in, H,  device=device, dtype=dtype) # Almost identical as well. 
w2 = torch.randn(H, D_out, device=device, dtype=dtype)

learning_rate = 1e-6
for t in range(NUM_ITERS):
	# Forward Pass
	h = x.mm(w1)
	h_relu = h.clamp(min=0)
	y_pred = h_relu.mm(w2)

	# Loss
	loss = (y_pred - y).pow(2).sum().item()
	print(t, loss)

	# Backwards Pass - get those grads.
	grad_y_pred = 2.0*(y_pred - y)
	grad_w2     = h_relu.t().mm(grad_y_pred)
	grad_h_relu = grad_y_pred.mm(w2.t())
	grad_h      = grad_h_relu.clone()
	grad_h[h < 0] = 0
	grad_w1     = x.t().mm(grad_h)

	# Update weights from gradients
	w1 -= learning_rate * grad_w1
	w2 -= learning_rate * grad_w2