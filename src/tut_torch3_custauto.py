# Sample Network
#
# (Input Layer)--fully-conn.-->(Single ReLU Layer)--fully-conn.-->(Output Layer)
#
# Same as 2, but now with a custom reLU op to show how to implement auto grad of your own.
import torch

#### Implementing a autograd enabled layer operation
# Must accept and return a tensor.
class MyReLU(torch.autograd.Function):
	"""
	Subclassing the autograd.Function class and overriding the
	forward and backward methods allows us to play with the
	rest of the ecosystem easily. 
	"""

	@staticmethod
	def forward(ctx, input):
		"""
		During forward pass you are given a ctx (context) object
		and a reference to the input data. The ctx object can be
		used to save the input, or the result of any other op 
		you need. This is useful because we can access it inside 
		the backward method, as it will be passed to that as well.
		
		in our ReLU, we will save the input for the back pass and
		return the proper clamped output.
		"""
		ctx.save_for_backward(input)
		return input.clamp(min=0)

	@staticmethod
	def backward(ctx, grad_output):
		"""
		The ctx is the context object described before. 

		We are also given the gradient of the loss with respect to the
		output, and we need to compute the gradient of the loss with 
		respect to the input. Yay, chain rule!
		"""
		input, = ctx.saved_tensors 		 # Unpacking the saved stuff
		grad_input = grad_output.clone()
		grad_input[input < 0] = 0 # Here we actually implement the grad.
		return grad_input
# The above will be used later in place of the typical relu op.
##### More or less the same code now.

dtype  = torch.float
device = torch.device("cpu") 

NUM_ITERS = 10
N = 64 # Batch Size
D_in, H, D_out = 1000, 100, 10 # Input, hidden, output layer dimensions

# Creating Random Data
x = torch.randn(N, D_in,  device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# Layer Weights 
w1 = torch.randn(D_in, H,  device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(NUM_ITERS):
	# Forward Pass - using MyReLU
	relu = MyReLU.apply # The Function.apply method turns this into a torch op? 
						# Recall that during this op, the class is saving the 
						# input of the forward pass to use later. 
	y_pred = relu(x.mm(w1)).mm(w2) # Replace the inline relu with our op.

	# Loss
	loss = (y_pred - y).pow(2).sum()
	print(t, loss.item())

	# Backward Pass 
	loss.backward()	# This is using our Function behind the scenes now.

	# Update weights from gradients
	with torch.no_grad():
		w1 -= learning_rate * w1.grad
		w2 -= learning_rate * w2.grad 
		# Zero the grads out?
		w1.grad.zero_()
		w2.grad.zero_()