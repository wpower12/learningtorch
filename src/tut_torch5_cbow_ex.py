import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right
EMBEDDING_DIM = 10
NUM_EPOCHS = 100

raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()

# By deriving a set from `raw_text`, we deduplicate the array
vocab = set(raw_text)
vocab_size = len(vocab)

word_to_ix = {word: i for i, word in enumerate(vocab)}
data = []

# Similar to the last exercise, the tutorial 'hard codes' building
# the data set instead of using the constant for CONTEXT_SIZE
# TODO - Make this general wrt CONTEXT_SIZE 
for i in range(2, len(raw_text) - 2):
    context = [raw_text[i - 2], raw_text[i - 1],
               raw_text[i + 1], raw_text[i + 2]]
    target = raw_text[i]
    data.append((context, target))
print(data[:5])

#### Create your model and train.  

# Implements the linear function over the sum of the embedded context wordsQ
class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # Note - Linear is LINEAR, it is learning A,b for x(A.T)+b
        self.linear    = nn.Linear(embedding_dim, vocab_size, bias=True)

    def forward(self, inputs):
        embed = self.embedding(inputs).sum(dim=0).view((1,-1)) # Sum all the embeded context words
                                                  # into one input vector.
        out = self.linear(embed)
        return F.log_softmax(out, dim=1)

def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)

loss_function = nn.NLLLoss()
model = CBOW(len(vocab), EMBEDDING_DIM)
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(NUM_EPOCHS):
    total_loss = 0
    for context, target in data:
        ctx_idxs = make_context_vector(context, word_to_ix)
        target_tensor = torch.tensor([word_to_ix[target]], dtype=torch.long)

        model.zero_grad()

        # Forward pass
        log_probs = model(ctx_idxs)

        # Loss
        loss = loss_function(log_probs, target_tensor)

        # Backwards
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(total_loss)




