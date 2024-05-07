import torch
import torch.nn as nn
import torchtext
from torchtext.data.utils import get_tokenizer

tokenizer = get_tokenizer("basic_english")


class RNNModel(nn.Module):
    def __init__(self, input_dim=20000, embedding_dim=64, hidden_dim=32, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.LSTM(
            embedding_dim, hidden_dim, num_layers=num_layers, dropout=0.5
        )
        self.fc = nn.Linear(hidden_dim * num_layers, 1)
        self.init_weights()

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.5, 0.5)
        self.fc.weight.data.uniform_(-0.5, 0.5)

    def forward(self, x):
        x = x.permute(1, 0)
        emb = self.embedding(x)
        # output will not be used because we have a many-to-one rnn
        output, (hidden, cell) = self.rnn(emb)
        hidden.squeeze_(0)
        hidden = hidden.transpose(0, 1)
        hidden = hidden.reshape(-1, self.hidden_dim * self.num_layers)
        out = self.fc(hidden)
        return out


model = RNNModel()
model.load_state_dict(torch.load("app/model_1.pt", map_location=torch.device("cpu")))

vocab = torch.load("app/vocab.pt", map_location=torch.device("cpu"))
text_pipeline = lambda x: vocab(tokenizer(x))


def predict(text):
    with torch.no_grad():
        text = torch.tensor(text_pipeline(text), dtype=torch.int64)
        text = torch.unsqueeze(text, 0)
        result = model(text).squeeze()
        result = (round(result.item(), 2)) * 10000
        return "{:20,.2f}".format(result)
