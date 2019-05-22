# Sample Network
#
# (Input Layer)--fully-conn.-->(Single ReLU Layer)--fully-conn.-->(Output Layer)
#
# This version only uses numpy
import numpy as np

NUM_ITERS = 500
N = 64 # batch size
D_in, H, D_out = 1000, 100, 10 # Input, hidden, output layer dimensions

# Creating random data
x = np.random.randn(N, D_in)  # Creates N vectors of dimension D_in with random values.
y = np.random.randn(N, D_out) # same but with D_out

# Layer weights
w1 = np.random.randn(D_in, H)  # For each of the D_in nodes in the input layer, 
							   # we need H-many weights
w2 = np.random.randn(H, D_out) # For each of the H nodes in the hidden layer, 
							   # we need D_out-many weights

learning_rate = 1e-6
for t in range(NUM_ITERS):
	# Forward Pass
	h = x.dot(w1)
	h_relu = np.maximum(h, 0)
	y_pred = h_relu.dot(w2)

	# Loss
	loss = np.square(y_pred - y).sum()
	print(t, loss)

	# Backward Pass - Note, just manually here.
	grad_y_pred = 2.0*(y_pred - y)
	grad_w2 = h_relu.T.dot(grad_y_pred)
	grad_h_relu = grad_y_pred.dot(w2.T)
	grad_h = grad_h_relu.copy()
	grad_h[h < 0] = 0 				# Coooool
	grad_w1 = x.T.dot(grad_h)

	# Update weights from gradients
	w1 -= learning_rate*grad_w1
	w2 -= learning_rate*grad_w2