import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

NUM_EPOCHS = 100
EMBEDDING_DIM = 6
HIDDEN_DIM = 6

# Need this to be "dictionary agnostic" bc we will use it for 
# both the sentence-word sequences and the tag sequences. 
def prepare_sequence(seq, to_ix):
	idxs = [to_ix[w] for w in seq]
	return torch.tensor(idxs, dtype=torch.long)

training_data = [
	("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
	("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]

word_to_ix = {}
for sent, tags in training_data:
	for word in sent:
		if word not in word_to_ix:
				word_to_ix[word] = len(word_to_ix)

# Just hard code the parts of speech we care about. 
tag_to_ix = {"DET": 0, "NN": 1, "V": 2}

# The Actual POS Tagging Model:
class LSTMTagger(nn.Module):
	def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
		super(LSTMTagger, self).__init__()
		self.hidden_dim = hidden_dim
		# Same as other examples, we want to learn some similarity 
		# preserving embeddings for the words we've seen.
		self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

		# LSTM projects from embedding dim to the hidden dim.
		# Each time step yields a new embedding from the next word
		# and the hidden state vector from the last state.
		self.lstm = nn.LSTM(embedding_dim, hidden_dim)

		# Finally, the hidden state of each step is projected
		# to the tagset space. This is done with a linear layer.
		self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

	def forward(self, sentence):
		embeds = self.word_embeddings(sentence)
		# Note, we are doing the sequence 'all at once'. Also, we 
		# only care about the entire set of outputs to pass to the
		# next stage of the model.
		lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
		# Note for both of these steps in the layer we need to ensure
		# we are looking at the correct shape view of the outputs. 
		# TODO - Write out the shape of these so you can track whats
		#        happening. 
		tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))	
		tag_scores = F.log_softmax(tag_space, dim=1)
		return tag_scores

# Now we actually train it. 

model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Tutorial has you print out the initial scores, form before training.
# I guess I know what the no_grad thing does. It just ignores all of the
# gradient calcs/saving when modules are excuted with it. So we can use 
# it when doing things like seeing these untrained values w.o wasting
# the cycles on the grad stuff. 
with torch.no_grad():
	inputs = prepare_sequence(training_data[0][0], word_to_ix)
	tag_scores = model(inputs)
	print(tag_scores)

# Actual Training:
for epoch in range(NUM_EPOCHS):
	total_loss = 0
	for sentence, tags in training_data:
		# Zero Grads - Again.
		model.zero_grad()

		# Get the inputs and targets.
		sentence_in = prepare_sequence(sentence, word_to_ix)
		targets     = prepare_sequence(tags, tag_to_ix)

		# Forward pass
		tag_scores = model(sentence_in)

		# Backward Pass, Update parameters (through optimizer)
		loss = loss_function(tag_scores, targets)
		loss.backward() # Wooo autograd
		optimizer.step()

		total_loss += loss.item()
	# print(total_loss)

# Now at the end we can compute scores again to compare:
# Note - Lazy copy and paste :(
with torch.no_grad():
	inputs = prepare_sequence(training_data[0][0], word_to_ix)
	tag_scores = model(inputs)
	print(tag_scores)

