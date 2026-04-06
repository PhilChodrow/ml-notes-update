# type: ignore
# flake8: noqa
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
import style
#
#
#
#
#
#
#
#| code-fold: true
import torch 
import torch
from torchinfo import summary
from torch import nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import urllib
import pandas as pd
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

# for appearance
import plotly.express as px
import plotly.io as pio
pio.templates.default = "plotly_white"
pio.renderers.default = "notebook_connected"
#
#
#
#
#
url = "https://raw.githubusercontent.com/PhilChodrow/ml-notes-update/refs/heads/main/data/seuss.txt"
text = "\n".join([line.decode('utf-8').strip() for line in urllib.request.urlopen(url)])
#
#
#
#
#
print(text[0:132])
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
from tokenizers import Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM

# GPT2 tokenizer and model
checkpoint = "openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
#
#
#
#
#
#
#
vocab_df = pd.DataFrame(list(tokenizer.vocab.items()), columns=["Token", "ID"])
vocab_df.head(10)
#
#
#
#
#
#
#
#
#
#
#
#---
sentence = "I do not like green eggs and ham."
tokens = tokenizer.encode(sentence)
#---

token_df = pd.DataFrame({"Token": tokens})
token_df["Text"] = token_df["Token"].apply(lambda x: tokenizer.decode(x))
token_df.head(10)
#
#
#
#
#
#---
decoded = tokenizer.decode(tokens)
print(decoded)
#---
#
#
#
#
#
print("Vocabulary size:", tokenizer.vocab_size)
#
#
#
#
#
#---
tokens = tokenizer.encode(text)
unique_tokens = set(tokens)
#---
print("Total number of tokens in text:", len(tokens))
print("Number of unique tokens in text:", len(unique_tokens))
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#---
def cosine_similarity(u, v):
    return u@v / (torch.norm(u) * torch.norm(v))
#---
#
#
#
#
#
#
#
#
#
#
#
#
#
example = "I do not like green eggs and ham."

#---
example_tokens = tokenizer.encode(example)

context_length = 2
data = []
for i in range(context_length, len(example_tokens)-context_length):
    for j in range(-context_length, context_length+1):
        if j != 0:
            data.append((example_tokens[i+j], example_tokens[i]))
#---

for context, target in data:
    print(f"{tokenizer.decode(context):<6} --> {tokenizer.decode(target)}")
#
#
#
#
#
#
#
from torch.utils.data import Dataset, DataLoader
class CBOWDataset(Dataset):
    def __init__(self, tokens, context_length):
        self.tokens = tokens
        
        # dict to remap tokens to contiguous integers
        self.token_to_idx = {token: i for i, token in enumerate(set(tokens))}
        self.idx_to_token = {i: token for token, i in self.token_to_idx.items()}
        
        # context length is the number of tokens on either side of the target token
        self.context_length = context_length
        self.data = []
        for i in range(context_length, len(tokens)-context_length):
            for j in range(-context_length, context_length+1):
                if j != 0:
                    self.data.append((self.token_to_idx[tokens[i+j]], self.token_to_idx[tokens[i]]))
                    self.data.append((self.token_to_idx[tokens[i+j]], self.token_to_idx[tokens[i]]))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx][0]), torch.tensor(self.data[idx][1])
#
#
#
#
#
context_length = 5
dataset = CBOWDataset(tokens, context_length=context_length)
x, y = dataset[0]

print(f"{x.item():<6} --> {y.item()}")
print(f"{tokenizer.decode(dataset.idx_to_token[x.item()]):<6} --> {tokenizer.decode(dataset.idx_to_token[y.item()])}")
#
#
#
#
#
data = CBOWDataset(tokens, context_length)
dataloader = DataLoader(data, batch_size=32, shuffle=True)
#
#
#
#
#
#
#
#---
class CBOW(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, d_embedding)
        self.linear = nn.Linear(d_embedding, vocab_size)
        
    def forward(self, x):
        embedded = self.embeddings(x) 
        output = self.linear(embedded)
        return output
#---
#
#
#
#
#
vocab_size = len(data.token_to_idx)
d_embedding = 4
model = CBOW(vocab_size, d_model=d_embedding)

summary(model, input_size=(32, context_length), device = device, dtypes = [torch.long])
#
#
#
#
#
#
#
#
#
def visualize_embeddings(model, tokenizer, data):

    # extract the embedding weights
    weights = model.embeddings.weight

    # use t-SNE to reduce the dimensionality of the embeddings to 2D
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(weights.detach().cpu().numpy())

    # make a dataframe for plotly 
    embedding_df = pd.DataFrame(embeddings_2d, columns=["Dim1", "Dim2"])
    embedding_df["Token"] = [tokenizer.decode(data.idx_to_token[i]) for i in range(len(embeddings_2d))]

    # labeled scatter plot of the embeddings
    fig = px.scatter(embedding_df, x="Dim1", y="Dim2", text="Token", title="2D Visualization of Token Embeddings")
    fig.update_traces(textposition='top center')
    fig.update_layout(xaxis_title="Dimension 1", yaxis_title="Dimension 2")
    fig.show()
#
#
#
#
#
#
#
visualize_embeddings(model, tokenizer, data)
#
#
#
#
#
#
#
#
#
#
#
#
#---
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()
model.to(device)
loss_history = []
#---
#
#
#
#---
for epoch in range(5):
    total_loss = 0
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        opt.zero_grad()
        output = model(X_batch)
        loss = loss_fn(output, y_batch)
        loss.backward()
        opt.step()

        total_loss += loss.item()
#---
    loss_history.append(total_loss / len(dataloader))
#
#
#
#
#
#
#
#| code-fold: true
fig, ax = plt.subplots()
ax.plot(loss_history, marker='o', color = "steelblue")
ax.set_title("Training Loss Over Epochs")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
plt.show()
#
#
#
#
#
#
#
#
#
#
#
visualize_embeddings(model, tokenizer, data)
#
#
#
#
#
#
#
#
#
#
#
#
base_token = "Fox"

comparison_tokens = [
    " socks",     # from Fox in Socks
    " box",       # from Fox in Socks
    " broom",     # from Fox in Socks
    " ham",       # from Green Eggs and Ham
    " Christmas", # from How the Grinch Stole Christmas
    " hat"        # from The Cat in the Hat
    ]

for token in comparison_tokens:
    idx = data.token_to_idx[tokenizer.encode(token)[0]]
    embedding = model.embeddings.weight[idx]
    print(f"Cosine similarity between '{base_token}' and '{token}': {cosine_similarity(model.embeddings.weight[data.token_to_idx[tokenizer.encode(base_token)[0]]], embedding):.4f}")
#
#
#
#
#
#
#
#
#
#
#
