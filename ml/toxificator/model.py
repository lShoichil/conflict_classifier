import torch
import torch.nn as nn
from navec import Navec
from slovnet.model.emb import NavecEmbedding


class ToxicClassifier(nn.Module):
    def __init__(self, navec):
        super(ToxicClassifier, self).__init__()
        self.hidden_size = 256
        drp = 0.1
        self.embedding = NavecEmbedding(navec)
        self.lstm = nn.LSTM(
            300,
            hidden_size=self.hidden_size,
            num_layers=3,
            batch_first=True,
            dropout=.4,
        )
        self.linear = nn.Linear(
            self.hidden_size * 2,
            64
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drp)
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        h_embedding = self.embedding(x)
        h_lstm, _ = self.lstm(h_embedding)
        avg_pool = torch.mean(h_lstm, 1)
        max_pool, _ = torch.max(h_lstm, 1)
        concat = torch.cat((avg_pool, max_pool), 1)
        concat = self.relu(self.linear(concat))
        concat = self.dropout(concat)
        out = self.out(concat)
        out = torch.sigmoid(out)
        return out
