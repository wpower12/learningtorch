import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

NUM_EPOCHS = 1
WORD_EMBEDDING_DIM = 6
CHAR_EMBEDDING_DIM = 3
W_HIDDEN_DIM = 6
C_HIDDEN_DIM = 3
NUM_CHARS = 26


def prepare_sequence(seq, to_ix):
	idxs = [to_ix[w] for w in seq]
	return torch.tensor(idxs, dtype=torch.long)

training_data = [
	("the dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
	("everybody read that book".split(), ["NN", "V", "DET", "NN"])
]

word_to_ix = {}
for sent, tags in training_data:
	for word in sent:
		if word not in word_to_ix:
				word_to_ix[word] = len(word_to_ix)

tag_to_ix  = {"DET": 0, "NN": 1, "V": 2}

# I could make this deal with capitols now, or just change the data to 
# only lower case. I did the second thing lol.
chars = "abcdefghijklmnopqrstuvwyxz"
char_to_ix = {}
for c in chars:
	char_to_ix[c] = len(char_to_ix)

# The Actual POS Tagging Model:
class LSTMTagger(nn.Module):
	def __init__(self, w_embedding_dim, 
					   c_embedding_dim, 
					   w_hidden_dim, 
					   c_hidden_dim, 
					   vocab_size, 
					   tagset_size):
		super(LSTMTagger, self).__init__()
		
		self.c_embedding_dim = c_embedding_dim
		self.w_embedding_dim = w_embedding_dim
		self.word_embeddings = nn.Embedding(vocab_size, w_embedding_dim)
		self.char_embeddings = nn.Embedding(NUM_CHARS, c_embedding_dim)

		# Char LSTM
		self.dim_clstm_in  = c_embedding_dim
		self.dim_clstm_out = c_hidden_dim
		self.c_lstm = nn.LSTM(c_embedding_dim, c_hidden_dim)
		
		# Word LSTM
		self.dim_wlstm_in  = w_embedding_dim+c_hidden_dim
		self.dim_wlstm_out = w_hidden_dim
		self.w_lstm = nn.LSTM(self.dim_wlstm_in, self.dim_wlstm_out)

		# Same as before, output of word LSTM goes to tag layer.
		self.hidden2tag = nn.Linear(w_hidden_dim, tagset_size)

	def forward(self, sentence):
		w_hidden = (torch.randn(1, 1, self.dim_wlstm_in),		# In to w_lstm  
				    torch.randn(1, 1, self.dim_wlstm_out))      # w_lstm to out

		for w in sentence:
			# Get Tensor of Character Embeddings
			char_seq      = prepare_sequence(w, char_to_ix)
			char_seq_in   = torch.tensor(char_seq, dtype=torch.long) 
			c_embeds      = self.char_embeddings(char_seq_in)
			c_embeds_in   = c_embeds.view(-1, 1, self.c_embedding_dim)
			print(c_embeds_in.shape)

			# Run the Character LSTM to get the Character Representation
			c_hidden = ((torch.randn(1, 1, self.dim_clstm_in)),
				        (torch.randn(1, 1, self.dim_clstm_out)))
			_ , c_lstm_out = self.c_lstm(c_embeds_in, c_hidden)
			char_rep = c_lstm_out[1] # Want the final hidden state OUT of the LSTM

			# Word Embedding for the Word Representation
			word_rep = self.word_embeddings(torch.tensor(word_to_ix[w]))
			word_rep = word_rep.view(1, 1, self.w_embedding_dim)

			concat_in = torch.cat((word_rep, char_rep), 2)
			print(concat_in.shape)

			w_lstm_out, w_hidden = self.w_lstm(concat_in, w_hidden)

		print("should have a leading dimension == length of sentence??")
		print(w_lstm_out.shape)
		
		tag_space = self.hidden2tag(w_lstm_out.view(len(sentence), -1))	
		tag_scores = F.log_softmax(tag_space, dim=1)

		return tag_scores	

# Now we actually train it. 

model = LSTMTagger(WORD_EMBEDDING_DIM, 
				   CHAR_EMBEDDING_DIM,
				   W_HIDDEN_DIM,
				   C_HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Tutorial has you print out the initial scores, form before training.
# I guess I know what the no_grad thing does. It just ignores all of the
# gradient calcs/saving when modules are excuted with it. So we can use 
# it when doing things like seeing these untrained values w.o wasting
# the cycles on the grad stuff. 
with torch.no_grad():
	inputs = training_data[0][0]
	tag_scores = model(inputs)
	print(tag_scores)

# Actual Training:
for epoch in range(NUM_EPOCHS):
	total_loss = 0
	for sentence, tags in training_data:
		# Zero Grads - Again.
		model.zero_grad()

		# Get the inputs and targets.
		targets     = prepare_sequence(tags, tag_to_ix)

		## For each word, we get a sequence
		# char_sequences = [prepare_sequence(w, char_to_ix) for w in sentence]
		# # for word in sentence:
		# # 	char_sequences.append(prepare_sequence(word, char_to_ix))
		# # 	# print(word)
		
		# char_seq_in = torch.tensor(char_sequences, dtype=torch.long)
		# print(char_seq_in.shape)

		# Forward pass
		print(sentence)
		tag_scores = model(sentence)

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

