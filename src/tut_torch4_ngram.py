"""
NGram Based Word Embeddings

Building a simple ngram based, latent feature space embedding.
Context Indexes --> Embedding Space--> Latent Space --> Vocab Space
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10
LATENT_SPACE_SIZE = 128 # Not used in actual tut, but I like it explicit.
NUM_EPOCHS = 100

# We will use Shakespeare Sonnet 2
test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()

# Building the trigram - the actual data set
#  - python list comprehension. 
#  - Note: The range for i has to 'stop early' so we aren't trying to index
#          the test_sentence out of bounds.
#  - Also note, that this is hardcoded for CONTEXT_SIZE = 2, but the
#    rest of the code assumes/uses the variable. This would be pretty
#    annoying to convert, I guess. 
#  - TODO: Convert to a generic version that uses the CONTEXT_SIZE constant.
trigrams = [([test_sentence[i], test_sentence[i + 1]], test_sentence[i + 2])
            for i in range(len(test_sentence) - 2)]
print(trigrams[:3])

vocab = set(test_sentence) # Makes a set, which we use to build a map
word_to_ix = {word: i for i, word in enumerate(vocab)}

# The actual model will be created as a subclass/extension of the nn.Module 
# class this lets it play with all the expected toys in torch, like optimizers 
# and automatic backwards propogation/gradient calcs.
class NGramLangaugeModeler(nn.Module):
	
	def __init__(self, vocab_size, embedding_dim, context_size):
		super(NGramLangaugeModeler, self).__init__()
		self.embedding = nn.Embedding(vocab_size, embedding_dim)
		self.linear1   = nn.Linear(context_size * embedding_dim, LATENT_SPACE_SIZE)
		self.linear2   = nn.Linear(LATENT_SPACE_SIZE, vocab_size) 

	def forward(self, inputs):
		embeds = self.embedding(inputs).view((1,-1))
		out = F.relu(self.linear1(embeds))
		out = self.linear2(out)
		log_probs = F.log_softmax(out, dim=1)
		return log_probs

losses = []
loss_function = nn.NLLLoss()
model = NGramLangaugeModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(NUM_EPOCHS):
	total_loss = 0
	for context, target in trigrams:
		# Formulate the input, which is a tensor of the indexes for each word
		# in the context.
		context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)

		model.zero_grad() # b/c torch will accumulate gradients unless you zero them.

		# Forward pass:
		log_probs = model(context_idxs)

		# Loss
		target = torch.tensor([word_to_ix[target]], dtype=torch.long)
		loss = loss_function(log_probs, target)

		# Backwards pass
		loss.backward()
		optimizer.step()

		total_loss += loss.item()
	print(total_loss)
	losses.append(total_loss)
# print(losses)

